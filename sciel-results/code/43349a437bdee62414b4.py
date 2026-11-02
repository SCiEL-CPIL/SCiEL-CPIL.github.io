import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition (rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def alternating_minimize(U, V, W, max_iter=300, tol=1e-10):
        """
        Alternating minimization for the trilinear tensor decomposition.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for iteration in range(max_iter):
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    W_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = W_row
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    coeff = np.zeros(r)
                    for k in range(p):
                        coeff += V[j * p + k, :] * W[k * n + i, :]
                    if np.linalg.norm(coeff) > 1e-12:
                        U_row = np.linalg.lstsq(coeff.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                        U[i * m + j, :] = U_row
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    coeff = np.zeros(r)
                    for i in range(n):
                        coeff += U[i * m + j, :] * W[k * n + i, :]
                    if np.linalg.norm(coeff) > 1e-12:
                        V_row = np.linalg.lstsq(coeff.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                        V[j * p + k, :] = V_row
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        error = max(error, abs(val - 1.0))
            
            if error < tol:
                break
        
        return U, V, W, error
    
    # Try different ranks using alternating minimization
    target_ranks = [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 25:
            break
        
        found = False
        
        for seed in range(20):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(seed * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            U, V, W, error = alternating_minimize(U, V, W, max_iter=150, tol=1e-10)
            
            if error < 1e-9:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_decomposition(U_r, V_r, W_r, tol=1e-8):
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
                else:
                    if check_decomposition(U, V, W, tol=1e-8):
                        best_rank = target_rank
                        best_U = U
                        best_V = V
                        best_W = W
                        found = True
                        break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
