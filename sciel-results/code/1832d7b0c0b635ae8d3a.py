"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
# 150s limit. We want to be safe.
_DEADLINE = 145

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W form a valid rank-r decomposition for n x m * m x p.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Vectorized check
    # We need to check T[i,j,k] = 1
    # T_comp[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Construct index arrays
    i_idx = np.arange(n)
    j_idx = np.arange(m)
    k_idx = np.arange(p)
    
    # This loop is small (2*3*4 = 24 iterations), so it's fast
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j, :]
                v_val = V[j*p + k, :]
                w_val = W[k*n + i, :]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    return U, V, W

def _try_rank_23_structured(n, m, p):
    """
    Attempt to find a rank 23 decomposition.
    Strategy: Start with schoolbook rank 24, and try to express one column as a linear combination of the others.
    This is a heuristic. We assume the tensor lies in a 23-dimensional subspace.
    We solve for the coefficients of the linear dependency.
    """
    if time.time() > _DEADLINE - 5:
        return None

    # Get schoolbook factors
    U24, V24, W24 = _construct_schoolbook(n, m, p)
    
    # We want to see if we can remove one column, say column 23.
    # This implies that the 24th term is a linear combination of the first 23.
    # T = sum_{r=0}^{22} u_r v_r w_r + u_23 v_23 w_23
    # If we can find alpha such that u_23 v_23 w_23 = sum_{r=0}^{22} alpha_r (u_r v_r w_r) ... this is not quite right because the factors are coupled.
    
    # A more robust approach: 
    # The space of matrices U, V, W of rank 23 is a subset of rank 24.
    # We can try to minimize the residual of the tensor with rank 23 using a few iterations of ALS starting from a truncated schoolbook.
    
    R = 23
    # Initialize with first 23 columns of schoolbook
    U = U24[:, :R].copy()
    V = V24[:, :R].copy()
    W = W24[:, :R].copy()
    
    # Add small random noise to break symmetry and allow ALS to adjust
    rng = np.random.RandomState(42)
    U += rng.randn(*U.shape) * 1e-4
    V += rng.randn(*V.shape) * 1e-4
    W += rng.randn(*W.shape) * 1e-4
    
    # ALS for rank 23
    # Target tensor T is implicit (1s at indices)
    # We minimize || T - sum_r U_r V_r W_r ||^2
    
    max_iters = 500
    tol = 1e-8
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            return None
            
        # Update U
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] is the variable
                # We want sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])_k = 1 for all k
                # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then A @ U[i*m+j, :]^T = ones(p)
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    # Solve least squares
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=1e-12)
                except np.linalg.LinAlgError:
                    return None
                    
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=1e-12)
                except np.linalg.LinAlgError:
                    return None
                    
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=1e-12)
                except np.linalg.LinAlgError:
                    return None
                    
        # Check convergence
        if it % 10 == 0:
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                return U, V, W

    return None

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Immediately construct the known valid rank 24 solution
    U24, V24, W24 = _construct_schoolbook(n, m, p)
    
    # Verify it (should always pass)
    if not _verify_decomposition(U24, V24, W24, n, m, p):
        # This should never happen
        pass
    
    # 2. Attempt to find a lower rank (23) if time permits
    # Given the small size, this is a "free" check.
    result_23 = _try_rank_23_structured(n, m, p)
    
    if result_23 is not None:
        U, V, W = result_23
        # Final strict verification
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # 3. Fallback to rank 24
    return {
        "U": U24.tolist(),
        "V": V24.tolist(),
        "W": W24.tolist()
    }
# EVOLVE-BLOCK-END
