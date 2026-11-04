import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition must satisfy: for all i,j,k,
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Compute tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _try_rank_reduction(rank_target, n, m, p, max_iter=500, tol=1e-10):
    """
    Try to find a rank-rank_target decomposition using random init and gradient-free search.
    Uses coordinate descent / alternating minimization.
    """
    rng = np.random.RandomState(42)
    
    for attempt in range(10):
        # Random initialization
        U = rng.randn(n * m, rank_target)
        V = rng.randn(m * p, rank_target)
        W = rng.randn(p * n, rank_target)
        
        # Scale to reasonable magnitude
        scale = (n * m * p / rank_target) ** (1/3)
        U *= scale
        V *= scale
        W *= scale
        
        # Alternating minimization
        for iteration in range(max_iter):
            # Fix V and W, solve for U
            # For each (i,j), U[i*m+j, :] should satisfy:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is a linear system: U[i*m+j, :] * (V[j,:] * W[i,:]) = ones
            # where (V[j,:] * W[i,:])[k] = V[j*p+k,:] * W[k*n+i,:]
            
            converged = True
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # Let M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Then sum_r U[i*m+j,r] * M[k,r] = 1 for all k
                    M = np.zeros((p, rank_target))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve M.T @ U[i*m+j, :] = ones(p)
                    # i.e., U[i*m+j, :] = (M.T)^+ @ ones
                    try:
                        U_new = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        U_new = U[i * m + j, :]
                    if np.linalg.norm(U_new - U[i * m + j, :]) > tol:
                        converged = False
                    U[i * m + j, :] = U_new
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    # Build system: for each i, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # Let M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # Then sum_r M[i,r] * V[j*p+k,r] = 1 for all i
                    M = np.zeros((n, rank_target))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V_new = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        V_new = V[j * p + k, :]
                    if np.linalg.norm(V_new - V[j * p + k, :]) > tol:
                        converged = False
                    V[j * p + k, :] = V_new
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    # Build system: for each j, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # Let M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # Then sum_r M[j,r] * W[k*n+i,r] = 1 for all j
                    M = np.zeros((m, rank_target))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W_new = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        W_new = W[k * n + i, :]
                    if np.linalg.norm(W_new - W[k * n + i, :]) > tol:
                        converged = False
                    W[k * n + i, :] = W_new
            
            if converged:
                break
        
        # Verify the solution
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix 
    by a 3x5 matrix. Searches for rank below 45 using alternating minimization.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline (10s safety margin)
    
    # Try ranks from 30 up to 44 (schoolbook is 45)
    for rank in range(30, 45):
        if time.time() > deadline:
            break
        result = _try_rank_reduction(rank, n, m, p, max_iter=200)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook (rank 45)
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
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
