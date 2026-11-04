"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a low-rank exact tensor decomposition for 4x4 x 4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Start with schoolbook rank 112
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Try ranks from 108 down to 80 (skip 112 since that's schoolbook)
    # Focus on the range where improvements are most likely
    for target_rank in range(108, 79, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try multiple random seeds for each rank
        for seed in range(15):
            if time.time() > deadline - 15:
                break
            result = _try_decomposition(n, m, p, target_rank, seed, deadline)
            if result is not None:
                best_rank = target_rank
                best_result = result
                found = True
                break
        
        if found:
            continue
    
    return best_result


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_decomposition(n, m, p, target_rank, seed, deadline):
    """
    Attempt to find a rank-`target_rank` decomposition using accelerated ALS.
    """
    np.random.seed(seed)
    
    # Build the target tensor's nonzero structure
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # For each a=i*m+j: nonzeros at (j*p+k, k*n+i) for k=0..p-1
    # For each b=j*p+k: nonzeros at (i*m+j, k*n+i) for i=0..n-1
    # For each c=k*n+i: nonzeros at (i*m+j, j*p+k) for j=0..m-1
    
    # Precompute index mappings
    a_to_nonzeros = {}  # a -> list of (b, c)
    b_to_nonzeros = {}  # b -> list of (a, c)
    c_to_nonzeros = {}  # c -> list of (a, b)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if a not in a_to_nonzeros:
                    a_to_nonzeros[a] = []
                a_to_nonzeros[a].append((b, c))
                if b not in b_to_nonzeros:
                    b_to_nonzeros[b] = []
                b_to_nonzeros[b].append((a, c))
                if c not in c_to_nonzeros:
                    c_to_nonzeros[c] = []
                c_to_nonzeros[c].append((a, b))
    
    # Random initialization with structured perturbation
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # ALS iterations
    max_iters = 80
    prev_loss = np.inf
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Update U
        for a in range(n * m):
            nz = a_to_nonzeros.get(a, [])
            if not nz:
                continue
            # Build design matrix: each row is V[b,:] * W[c,:]
            M = np.array([V[b, :] * W[c, :] for b, c in nz])  # (len(nz), target_rank)
            t = np.ones(len(nz))
            # Solve M^T @ u = t via lstsq
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
            except np.linalg.LinAlgError:
                pass
        
        # Update V
        for b in range(m * p):
            nz = b_to_nonzeros.get(b, [])
            if not nz:
                continue
            M = np.array([U[a, :] * W[c, :] for a, c in nz])
            t = np.ones(len(nz))
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
            except np.linalg.LinAlgError:
                pass
        
        # Update W
        for c in range(p * n):
            nz = c_to_nonzeros.get(c, [])
            if not nz:
                continue
            M = np.array([U[a, :] * V[b, :] for a, b in nz])
            t = np.ones(len(nz))
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
            except np.linalg.LinAlgError:
                pass
        
        # Check convergence and validity periodically
        if it % 8 == 7:
            if _check_validity(U, V, W, n, m, p, a_to_nonzeros, b_to_nonzeros, c_to_nonzeros, tol=1e-3):
                # Try to snap to half-integers
                U_snap, V_snap, W_snap = _snap_to_half_integers(U, V, W)
                if _check_validity(U_snap, V_snap, W_snap, n, m, p, a_to_nonzeros, b_to_nonzeros, c_to_nonzeros, tol=1e-8):
                    return _to_lists(U_snap, V_snap, W_snap)
                elif _check_validity(U, V, W, n, m, p, a_to_nonzeros, b_to_nonzeros, c_to_nonzeros, tol=1e-8):
                    return _to_lists(U, V, W)
                else:
                    # Try a polish pass with tighter tolerance
                    U_p, V_p, W_p = _polish(U, V, W, n, m, p, a_to_nonzeros, b_to_nonzeros, c_to_nonzeros, deadline)
                    if _check_validity(U_p, V_p, W_p, n, m, p, a_to_nonzeros, b_to_nonzeros, c_to_nonzeros, tol=1e-8):
                        U_s, V_s, W_s = _snap_to_half_integers(U_p, V_p, W_p)
                        if _check_validity(U_s, V_s, W_s, n, m, p, a_to_nonzeros, b_to_nonzeros, c_to_nonzeros, tol=1e-8):
                            return _to_lists(U_s, V_s, W_s)
                        return _to_lists(U_p, V_p, W_p)
    
    return None


def _check_validity(U, V, W, n, m, p, a_to_nonzeros, b_to_nonzeros, c_to_nonzeros, tol=1e-6):
    """Check if the decomposition is valid."""
    # Check all valid positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a] * V[b], W[c])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check invalid positions (sample)
    np.random.seed(999)
    for _ in range(300):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        
        # Check if this is a valid triple
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        
        val = np.dot(U[a] * V[b], W[c])
        if abs(val) > tol:
            return False
    
    return True


def _polish(U, V, W, n, m, p, a_to_nonzeros, b_to_nonzeros, c_to_nonzeros, deadline):
    """High-precision polish pass."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(30):
        if time.time() > deadline - 3:
            break
        
        for a in range(n * m):
            nz = a_to_nonzeros.get(a, [])
            if not nz:
                continue
            M = np.array([V[b, :] * W[c, :] for b, c in nz])
            t = np.ones(len(nz))
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=1e-12)
            except np.linalg.LinAlgError:
                pass
        
        for b in range(m * p):
            nz = b_to_nonzeros.get(b, [])
            if not nz:
                continue
            M = np.array([U[a, :] * W[c, :] for a, c in nz])
            t = np.ones(len(nz))
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=1e-12)
            except np.linalg.LinAlgError:
                pass
        
        for c in range(p * n):
            nz = c_to_nonzeros.get(c, [])
            if not nz:
                continue
            M = np.array([U[a, :] * V[b, :] for a, b in nz])
            t = np.ones(len(nz))
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=1e-12)
            except np.linalg.LinAlgError:
                pass
    
    return U, V, W


def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer if close."""
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    return U_snap, V_snap, W_snap


def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
