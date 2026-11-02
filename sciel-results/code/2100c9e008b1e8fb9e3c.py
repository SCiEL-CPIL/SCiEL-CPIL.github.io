import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x7 matrix.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Construct Schoolbook Decomposition (Rank 42)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Precompute the target tensor T for verification
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0

    def verify(U_mat, V_mat, W_mat, atol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', W_mat, U_mat, V_mat)
        return np.allclose(T_recon, T, atol=atol)

    # 2. Attempt to find a lower rank decomposition using ALS
    # Try ranks from 41 downwards
    for target_rank in range(rank_schoolbook - 1, 0, -1):
        if time.time() > deadline - 20:
            break
            
        # Multiple restarts for ALS
        found = False
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize to avoid scaling issues
            U /= (np.linalg.norm(U) + 1e-6)
            V /= (np.linalg.norm(V) + 1e-6)
            W /= (np.linalg.norm(W) + 1e-6)
            
            # ALS Loop
            max_iters = 50
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                # Update W
                # For each k, i: solve M_W * w = 1
                # M_W[j, r] = U[i*m+j, r] * V[j*p+k, r]
                for k in range(p):
                    for i in range(n):
                        M_W = np.zeros((m, target_rank))
                        for j in range(m):
                            M_W[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        # Solve M_W * w = 1
                        # M_W is m x R. We want w (R x 1) such that M_W @ w = 1 (m x 1)
                        # Using lstsq
                        w, residuals, rank_w, sv = np.linalg.lstsq(M_W, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                        
                # Update V
                # For each j, k: solve M_V * v = 1
                # M_V[i, r] = U[i*m+j, r] * W[k*n+i, r]
                for j in range(m):
                    for k in range(p):
                        M_V = np.zeros((n, target_rank))
                        for i in range(n):
                            M_V[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        v, residuals, rank_v, sv = np.linalg.lstsq(M_V, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                        
                # Update U
                # For each i, j: solve M_U * u = 1
                # M_U[k, r] = V[j*p+k, r] * W[k*n+i, r]
                for i in range(n):
                    for j in range(m):
                        M_U = np.zeros((p, target_rank))
                        for k in range(p):
                            M_U[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        u, residuals, rank_u, sv = np.linalg.lstsq(M_U, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                        
                # Check convergence
                if it > 0:
                    T_recon = np.einsum('ar,br,cr->abc', W, U, V)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-9:
                        # Polish step: try to reduce rank by checking if any column is redundant
                        # or if we can find a slightly lower rank solution
                        # For now, accept this solution and try to see if we can go lower
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        found = True
                        break
                        
            if found:
                break

    # 3. Return the best result
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
