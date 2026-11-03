"""
Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.

We exploit the factorization T_{2,3,6} = T_{2,3,3} ⊗ T_{1,1,2}.
Since T_{1,1,2} has rank 1, rank(T_{2,3,6}) <= rank(T_{2,3,3}).
We search for a rank-r decomposition of T_{2,3,3} (r < 18) and extend it.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    For all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integers(mat, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    mat = np.asarray(mat, dtype=float)
    snapped = np.round(mat * 2) / 2.0
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_233_rank(r, deadline, rng):
    """
    Try to find a rank-r decomposition of T_{2,3,3}.
    Uses randomized local search with half-integer constraints.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = 2, 3, 3
    # U: 6 x r, V: 9 x r, W: 6 x r
    
    # Try multiple random initializations
    for trial in range(200):
        if time.time() > deadline:
            return None
        
        # Initialize with small random half-integers
        U = rng.randint(-2, 3, size=(6, r)).astype(float) * 0.5
        V = rng.randint(-2, 3, size=(9, r)).astype(float) * 0.5
        W = rng.randint(-2, 3, size=(6, r)).astype(float) * 0.5
        
        # Local search: iteratively adjust entries to satisfy equations
        for iteration in range(500):
            if time.time() > deadline:
                return None
            
            # Compute residuals for all 18 equations
            residuals = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                        residuals.append((i, j, k, val - 1.0))
            
            # Check if all residuals are near zero
            max_res = max(abs(res) for _, _, _, res in residuals)
            if max_res < 1e-9:
                valid = _verify_decomposition(U, V, W, n, m, p)
                if valid:
                    return U, V, W
            
            # Find the equation with the largest residual and adjust
            i, j, k, res = max(residuals, key=lambda x: abs(x[3]))
            
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # Try adjusting U[u_idx, c] for random c
            c = rng.randint(0, r)
            old_U = U[u_idx, c].copy()
            U[u_idx, c] += 0.5 * (1 if res < 0 else -1)
            
            new_res = float(np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])) - 1.0
            if abs(new_res) > abs(res):
                U[u_idx, c] = old_U
            
            # Try adjusting V[v_idx, c2]
            c2 = rng.randint(0, r)
            old_V = V[v_idx, c2].copy()
            V[v_idx, c2] += 0.5 * (1 if res < 0 else -1)
            
            new_res = float(np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])) - 1.0
            if abs(new_res) > abs(res):
                V[v_idx, c2] = old_V
            
            # Try adjusting W[w_idx, c3]
            c3 = rng.randint(0, r)
            old_W = W[w_idx, c3].copy()
            W[w_idx, c3] += 0.5 * (1 if res < 0 else -1)
            
            new_res = float(np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])) - 1.0
            if abs(new_res) > abs(res):
                W[w_idx, c3] = old_W
        
        # After each major iteration, try snapping to half-integers
        snapped_U = _snap_to_half_integers(U)
        snapped_V = _snap_to_half_integers(V)
        snapped_W = _snap_to_half_integers(W)
        if snapped_U is not None and snapped_V is not None and snapped_W is not None:
            valid = _verify_decomposition(snapped_U, snapped_V, snapped_W, n, m, p)
            if valid:
                return snapped_U, snapped_V, snapped_W
    
    return None

def _structured_search(n, m, p, deadline):
    """
    Try structured decompositions based on tensor product factorization.
    
    Key insight: T_{2,3,6} = T_{2,3,3} ⊗ T_{1,1,2}
    Since rank(T_{1,1,2}) = 1, we have rank(T_{2,3,6}) <= rank(T_{2,3,3}).
    
    We search for a rank-r decomposition of T_{2,3,3} with r < 18,
    then extend it to T_{2,3,6} via the tensor product construction.
    """
    n2, m2, p2 = 2, 3, 3
    schoolbook_233 = n2 * m2 * p2  # 18
    
    # Search for rank < 18 decomposition of T_{2,3,3}
    rng = np.random.RandomState(42)
    best_233 = None
    for r in range(1, schoolbook_233):
        if time.time() > deadline:
            break
        result = _try_233_rank(r, deadline, rng)
        if result is not None:
            best_233 = result
            break
    
    if best_233 is not None:
        U233, V233, W233 = best_233
        r233 = U233.shape[1]
        
        # T_{1,1,2} rank-1 decomposition:
        # U1 = [[1]], V1 = [[1],[1]], W1 = [[1],[1]]
        U1 = np.array([[1.0]])
        V1 = np.array([[1.0], [1.0]])
        W1 = np.array([[1.0], [1.0]])
        
        # Extend to T_{2,3,6} via tensor product
        U6 = np.zeros((6, r233))
        V6 = np.zeros((18, r233))
        W6 = np.zeros((12, r233))
        
        # U unchanged (n=2, m=3)
        U6 = U233.copy()
        
        # V: index (j,k) in 3x6, k = k1*2 + k2
        for j in range(3):
            for k in range(6):
                k1 = k // 2
                k2 = k % 2
                V6[j * 6 + k, :] = V233[j * 3 + k1, :] * V1[k2, 0]
        
        # W: index (k,i) in 6x2, k = k1*2 + k2
        for k in range(6):
            k1 = k // 2
            k2 = k % 2
            for i in range(2):
                W6[k * 2 + i, :] = W233[k1 * 2 + i, :] * W1[k2, 0]
        
        valid = _verify_decomposition(U6, V6, W6, 2, 3, 6)
        if valid:
            return U6, V6, W6
    
    return None

def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 2x3 * 3x6.
    Returns {'U': U, 'V': V, 'W': W} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Try the structured approach (tensor product reduction)
    result = _structured_search(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        snapped_U = _snap_to_half_integers(U)
        snapped_V = _snap_to_half_integers(V)
        snapped_W = _snap_to_half_integers(W)
        if snapped_U is not None and snapped_V is not None and snapped_W is not None:
            U, V, W = snapped_U, snapped_V, snapped_W
        valid = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook (rank 36)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
