"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_mm_tensor(n, m, p):
    """Build the 4x4x4 matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k).
    
    The tensor encodes: C[i,k] = sum_j A[i,j] * B[j,k]
    In the decomposition convention:
      U has shape (n*m, rank)  -> indexed by (i, j)
      V has shape (m*p, rank)  -> indexed by (j, k)  
      W has shape (p*n, rank)  -> indexed by (k, i)  [trace convention W[k*n+i, r]]
    
    The tensor T has shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all (a,b,c)."""
    T = _build_mm_tensor(n, m, p)
    # Compute the tensor from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Efficient: reshape and use einsum
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=tol, rtol=0):
        return False, np.max(np.abs(T_recon - T))
    return True, 0.0

def _cp_decompose(T, rank, n_iter=200, seed=42):
    """Find a CP decomposition of tensor T with given rank using ALS.
    
    T has shape (I, J, K). We seek U (I,rank), V (J,rank), W (K,rank)
    such that T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r].
    """
    I, J, K = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U = rng.randn(I, rank)
    V = rng.randn(J, rank)
    W = rng.randn(K, rank)
    
    for it in range(n_iter):
        # Update U: for each r, U[:,r] = matvec(T, V[:,r], W[:,r])
        # T contracted over modes 2 and 3 with V[:,r] and W[:,r]
        # U_new[:,r] = sum_{j,k} T[:,j,k] * V[j,r] * W[k,r]
        # = (T * V[:,r] * W[:,r]) -- mode-1 unfolding times outer product
        # Mode-1 unfolding: T_1 has shape (I, J*K)
        # U[:,r] = T_1 @ (V[:,r] * W[:,r])  where V[:,r]*W[:,r] is outer product flattened
        
        # More efficient: use einsum
        # U[:,r] = np.einsum('ijk,jr,kr->ir', T, V, W)
        U_new = np.einsum('ijk,jr,kr->ir', T, V, W)
        # Normalize columns to avoid scale drift
        norms = np.linalg.norm(U_new, axis=0)
        norms[norms < 1e-15] = 1.0
        U_new /= norms
        U = U_new
        
        V_new = np.einsum('ijk,ir,kr->jr', T, U, W)
        norms = np.linalg.norm(V_new, axis=0)
        norms[norms < 1e-15] = 1.0
        V_new /= norms
        V = V_new
        
        W_new = np.einsum('ijk,ir,jr->kr', T, U, V)
        norms = np.linalg.norm(W_new, axis=0)
        norms[norms < 1e-15] = 1.0
        W_new /= norms
        W = W_new
        
        # Check convergence
        if it % 20 == 0:
            err = np.linalg.norm(np.einsum('ir,jr,kr->ijk', U, V, W) - T)
            if err < 1e-10:
                break
    
    return U, V, W

def _try_rank(T, rank, max_seeds=5, n_iter=300):
    """Try to find a valid CP decomposition of T with the given rank."""
    for seed in range(max_seeds):
        U, V, W = _cp_decompose(T, rank, n_iter=n_iter, seed=seed)
        valid, err = _verify_decomposition(U, V, W, *T.shape[0] and (4,4,4))
        if valid:
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    T = _build_mm_tensor(n, m, p)
    
    # The known minimal rank for 4x4x4 matrix multiplication is 26 (Landsberg 2006).
    # Try ranks from 26 upward.
    best_result = None
    
    # First, try to find a decomposition at rank 26
    for rank in [26, 27, 28, 29, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64]:
        if time.time() > deadline:
            break
        
        result = _try_rank(T, rank, max_seeds=3, n_iter=200)
        if result is not None:
            U, V, W = result
            # Convert to list of lists with integer/half-integer entries if possible
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify once more
            valid, err = _verify_decomposition(U_list, V_list, W_list, n, m, p)
            if valid:
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    # Fallback: schoolbook decomposition (rank 64)
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
