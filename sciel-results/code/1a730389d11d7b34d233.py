import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    num_a = n * m  # 6
    num_b = m * p  # 21
    num_c = p * n  # 14
    
    # Build tensor T
    T = np.zeros((num_a, num_b, num_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from 21 upward
    min_rank = max(num_a, num_b, num_c)  # 21
    
    for rank in range(min_rank, min_rank + 8):
        if time.time() > deadline - 5:
            break
        
        # Multiple random restarts
        found = False
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(num_a, rank) * 0.5
            V = np.random.randn(num_b, rank) * 0.5
            W = np.random.randn(num_c, rank) * 0.5
            
            # Alternating least squares
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U
                new_U = np.zeros_like(U)
                for a in range(num_a):
                    M_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(num_b * num_c, rank)
                    b_vec = T[a, :, :].reshape(-1)
                    new_U[a, :] = np.linalg.lstsq(M_a, b_vec, rcond=None)[0]
                U = new_U
                
                # Solve for V
                new_V = np.zeros_like(V)
                for b in range(num_b):
                    M_b = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(num_a * num_c, rank)
                    b_vec = T[:, b, :].reshape(-1)
                    new_V[b, :] = np.linalg.lstsq(M_b, b_vec, rcond=None)[0]
                V = new_V
                
                # Solve for W
                new_W = np.zeros_like(W)
                for c in range(num_c):
                    M_c = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(num_a * num_b, rank)
                    b_vec = T[:, :, c].reshape(-1)
                    new_W[c, :] = np.linalg.lstsq(M_c, b_vec, rcond=None)[0]
                W = new_W
            
            # Check exactness
            T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            max_err = np.max(np.abs(T_recon - T))
            if max_err < 1e-8:
                found = True
                break
        
        if found:
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
