import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if it's a valid multiplication term
    # Actually, the tensor is defined by the indices.
    # T[i, j, k] is 1 for all i, j, k? No.
    # The tensor is the structure of the multiplication.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has components T[i, j, k] which is 1.
    # Wait, the tensor is the map (A, B) -> C.
    # The decomposition is sum_c U_c V_c W_c.
    # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c].
    # For matrix multiplication, T[i, j, k] is 1 for all i, j, k?
    # No. The tensor is the "multiplication tensor".
    # It is 1 if the indices correspond to a term in the sum.
    # In the schoolbook, every combination (i, j, k) is a term.
    # So T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    T = np.ones((n, m, p))
    
    # Initialize with Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try ALS search for lower rank
    # We try ranks from 40 down to 26
    for target_rank in range(40, 25, -1):
        if time.time() > deadline:
            break
            
        # Random initialization
        np.random.seed(42) # Fixed seed for reproducibility in this run? Or random?
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # ALS Iterations
        for it in range(50):
            if time.time() > deadline:
                break
                
            # Update W
            # For each k, solve for W[k*n + i, :]
            # Equation: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # This is linear in W for fixed k, i?
            # No, W[k*n+i, c] is the variable.
            # For fixed k, we have variables W[k*n+0, :], W[k*n+1, :], W[k*n+2, :]
            # We have 9 equations (for all i, j).
            # Let's form the matrix A for this system.
            # Variables are x_c = W[k*n+i, c] for all c and i.
            # It's easier to solve for each i separately?
            # No, the equations mix i and j.
            # T[i, j, k] = sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
            # For fixed k, let M[i, j, c] = U[i*m+j, c] * V[j*p+k, c]
            # We want to find W_vec (size n * R) such that:
            # for all i, j: sum_c M[i, j, c] * W_vec[i*R + c] = T[i, j, k]
            
            # Construct A (9 x n*R) and b (9)
            A_w = np.zeros((n * m, n * target_rank))
            b_w = np.zeros(n * m)
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    b_w[idx] = T[i, j, k]
                    for c in range(target_rank):
                        A_w[idx, i * target_rank + c] = U[i * m + j, c] * V[j * p + k, c]
            
            # Solve A_w @ W_vec = b_w
            W_vec, residuals, rank_w, s = np.linalg.lstsq(A_w, b_w, rcond=None)
            W[k * n * target_rank : (k+1) * n * target_rank] = W_vec.reshape(n, target_rank).flatten()
            
            # Update U
            # For each i, solve for U[i*m+j, :]
            # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # Variables: U[i*m+0, :], U[i*m+1, :], U[i*m+2, :]
            # Equations: for all j, k. (m * p = 15 equations)
            
            A_u = np.zeros((m * p, m * target_rank))
            b_u = np.zeros(m * p)
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    b_u[idx] = T[i, j, k]
                    for c in range(target_rank):
                        A_u[idx, j * target_rank + c] = V[j * p + k, c] * W[k * n + i, c]
                        
            U_vec, residuals, rank_u, s = np.linalg.lstsq(A_u, b_u, rcond=None)
            U[i * m * target_rank : (i+1) * m * target_rank] = U_vec.reshape(m, target_rank).flatten()
            
            # Update V
            # For each j, solve for V[j*p+k, :]
            # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # Variables: V[j*p+0, :], ..., V[j*p+4, :]
            # Equations: for all i, k. (n * p = 15 equations)
            
            A_v = np.zeros((n * p, p * target_rank))
            b_v = np.zeros(n * p)
            for i in range(n):
                for k in range(p):
                    idx = i * p + k
                    b_v[idx] = T[i, j, k]
                    for c in range(target_rank):
                        A_v[idx, k * target_rank + c] = U[i * m + j, c] * W[k * n + i, c]
                        
            V_vec, residuals, rank_v, s = np.linalg.lstsq(A_v, b_v, rcond=None)
            V[j * p * target_rank : (j+1) * p * target_rank] = V_vec.reshape(p, target_rank).flatten()
            
        # Check validity
        # Reconstruct tensor
        T_recon = np.zeros((n, m, p))
        for c in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        
        if np.allclose(T_recon, T, atol=1e-9):
            # Valid decomposition
            # Prune zero columns
            mask = (np.any(np.abs(U) > 1e-9, axis=0) | 
                    np.any(np.abs(V) > 1e-9, axis=0) | 
                    np.any(np.abs(W) > 1e-9, axis=0))
            # Actually, a column is useless if U[:,c]=0 OR V[:,c]=0 OR W[:,c]=0
            # Wait, if U[:,c]=0, the term is 0.
            # So we keep columns where U[:,c] != 0 AND V[:,c] != 0 AND W[:,c] != 0?
            # No, if U[:,c]=0, the term contributes 0. We can drop it.
            # So we keep c if U[:,c] is not zero AND V[:,c] is not zero AND W[:,c] is not zero.
            mask = (np.all(np.abs(U) > 1e-9, axis=0) & 
                    np.all(np.abs(V) > 1e-9, axis=0) & 
                    np.all(np.abs(W) > 1e-9, axis=0))
            
            if np.sum(mask) < best_rank:
                best_rank = np.sum(mask)
                best_U = U[:, mask]
                best_V = V[:, mask]
                best_W = W[:, mask]
                
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
