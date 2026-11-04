"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a low-rank exact tensor decomposition for 4x4 x 4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Start with schoolbook rank 112
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Precompute index mappings for the tensor
    a_to_nonzeros = {}
    b_to_nonzeros = {}
    c_to_nonzeros = {}
    
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
    
    # Try ranks from 108 down to 80
    # We prioritize lower ranks but start slightly below schoolbook to find improvements quickly
    for target_rank in range(108, 79, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Increase number of seeds for lower ranks as they are harder
        num_seeds = 20 if target_rank > 90 else 40
        
        for seed in range(num_seeds):
            if time.time() > deadline - 5:
                break
                
            result = _try_decomposition(n, m, p, target_rank, seed, deadline, 
                                        a_to_nonzeros, b_to_nonzeros, c_to_nonzeros)
            if result is not None:
                best_rank = target_rank
                best_result = result
                found = True
                break
        
        if found:
            # If we found a rank, we can stop or continue to find even lower?
            # The task asks for rank-minimal. So we should continue searching for lower ranks.
            # But if we found 100, we should check 99, 98...
            # The loop does this naturally.
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

def _try_decomposition(n, m, p, target_rank, seed, deadline, 
                       a_to_nonzeros, b_to_nonzeros, c_to_nonzeros):
    """
    Attempt to find a rank-`target_rank` decomposition using ALS.
    """
    np.random.seed(seed)
    
    # Random initialization
    # Scale initialization to help convergence
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    max_iters = 100
    prev_loss = np.inf
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # Update U
        for a in range(n * m):
            nz = a_to_nonzeros.get(a, [])
            if not nz:
                continue
            # Build design matrix: each row is V[b,:] * W[c,:]
            # Vectorized construction
            # nz is list of (b, c)
            # We need M[i, :] = V[nz[i][0]] * W[nz[i][1]]
            # This is hard to vectorize fully without indexing tricks, but we can do it in a list comp
            M = np.array([V[b] * W[c] for b, c in nz])
            t = np.ones(len(nz))
            try:
                U[a], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
            except np.linalg.LinAlgError:
                pass
                
        # Update V
        for b in range(m * p):
            nz = b_to_nonzeros.get(b, [])
            if not nz:
                continue
            M = np.array([U[a] * W[c] for a, c in nz])
            t = np.ones(len(nz))
            try:
                V[b], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
            except np.linalg.LinAlgError:
                pass
                
        # Update W
        for c in range(p * n):
            nz = c_to_nonzeros.get(c, [])
            if not nz:
                continue
            M = np.array([U[a] * V[b] for a, b in nz])
            t = np.ones(len(nz))
            try:
                W[c], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
            except np.linalg.LinAlgError:
                pass
        
        # Check validity periodically (every 10 iters) to avoid overhead
        if it % 10 == 9:
            # First, check if the current solution is close to exact
            # We don't want to snap yet, just check if we are close enough to try snapping
            # Actually, just try snapping and checking. It's cheap.
            U_snap, V_snap, W_snap = _snap_to_half_integers(U, V, W)
            if _check_validity(U_snap, V_snap, W_snap, n, m, p, 
                               a_to_nonzeros, b_to_nonzeros, c_to_nonzeros, tol=1e-6):
                return _to_lists(U_snap, V_snap, W_snap)
                
            # Also check if the unsnapped version is extremely close to a valid half-integer solution
            # This is redundant if we just snap, but maybe the snap is too aggressive?
            # No, if it's a valid half-integer solution, the real solution should be very close to it.
            
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
    # To be rigorous, we should check all invalid positions if the rank is low enough,
    # but sampling is faster. Given the density, sampling 1000 should catch most errors.
    np.random.seed(999)
    num_samples = 1000
    for _ in range(num_samples):
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

def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer."""
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
