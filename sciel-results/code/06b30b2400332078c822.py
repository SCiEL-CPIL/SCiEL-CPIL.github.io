import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_u = n * m  # 20
    dim_v = m * p  # 25
    dim_w = p * n  # 20
    
    # Target Tensor T
    # T is zero everywhere except T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
            
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=np.int64)
        V = np.zeros((dim_v, rank), dtype=np.int64)
        W = np.zeros((dim_w, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Try to find lower rank
    # Lower bound is max(n*m, m*p, p*n) = 25? No, lower bound is max(n, m, p) * min(n, m, p)?
    # Actually, rank >= max(n*m, m*p, p*n) is false. Rank >= n*m is false.
    # Rank >= max(n, m, p) is true? No.
    # Let's just try from 40 upwards.
    
    for rank in range(40, 101):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts with local search
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random init with integers -1, 0, 1
            U = np.random.choice([-1, 0, 1], size=(dim_u, rank))
            V = np.random.choice([-1, 0, 1], size=(dim_v, rank))
            W = np.random.choice([-1, 0, 1], size=(dim_w, rank))
            
            # Local search
            # We want to minimize sum (T - U V W)^2
            # Since T is 0/1, and we use integers, we want exact match.
            
            # Compute error
            def compute_error(U, V, W):
                # T_hat = U @ V @ W? No.
                # T_hat[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
                # Use einsum: 'ak,bk,ck->abc'
                T_hat = np.einsum('ak,bk,ck->abc', U, V, W)
                return np.sum((T - T_hat) ** 2)
            
            current_error = compute_error(U, V, W)
            if current_error == 0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            # Simple local search: flip random element
            # To speed up, we can precompute contributions?
            # But dim is small.
            
            for step in range(2000):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick random matrix and index
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    mat = U
                    dim1, dim2 = dim_u, rank
                elif mat_idx == 1:
                    mat = V
                    dim1, dim2 = dim_v, rank
                else:
                    mat = W
                    dim1, dim2 = dim_w, rank
                    
                idx1 = np.random.randint(0, dim1)
                idx2 = np.random.randint(0, dim2)
                
                # Try changing value by +1 or -1
                old_val = mat[idx1, idx2]
                
                # Try +1
                mat[idx1, idx2] = old_val + 1
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                    if current_error == 0:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    mat[idx1, idx2] = old_val
                    
                # Try -1
                mat[idx1, idx2] = old_val - 1
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                    if current_error == 0:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    mat[idx1, idx2] = old_val

    return get_schoolbook()
