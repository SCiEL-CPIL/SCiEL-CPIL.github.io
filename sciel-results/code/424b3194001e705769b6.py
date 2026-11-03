import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize schoolbook
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    def check_fast(U, V, W):
        # Vectorized check
        # T_recon[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can compute this using einsum
        # But the indices are coupled, so we need to be careful.
        
        # Let's compute it in a vectorized way
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # We want to compute T_recon of shape (n, m, p)
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # T_recon[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # This is a bit tricky because the indices are coupled.
        
        # Let's use a loop over i, j, k but vectorize over c
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i,j,k] = np.sum(U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:])
        
        target = np.ones((n, m, p), dtype=float)
        return np.allclose(T_recon, target, atol=1e-9)
    
    start_time = time.time()
    deadline = start_time + 140
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to reduce rank by removing columns
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        if best_rank <= 1:
            break
        
        # Try to remove a random column
        c_remove = random.randint(0, best_rank - 1)
        
        # Create new matrices without column c_remove
        U_new = np.delete(best_U, c_remove, axis=1)
        V_new = np.delete(best_V, c_remove, axis=1)
        W_new = np.delete(best_W, c_remove, axis=1)
        
        # Check if the new decomposition is valid
        if check_fast(U_new, V_new, W_new):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank -= 1
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
