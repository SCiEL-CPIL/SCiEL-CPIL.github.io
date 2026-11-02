import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Indices
    # U is (n*m, rank)
    # V is (m*p, rank)
    # W is (p*n, rank)
    
    # Initialize with zeros
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time
                if time.time() - start_time > timeout:
                    break
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
        
        if time.time() - start_time > timeout:
            break
        
    if time.time() - start_time > timeout:
        # If we broke early, we might have an incomplete matrix.
        # But for n=3, m=5, p=5, loop is 75 iterations, very fast.
        # This guard is for safety.
        pass

    # Convert to lists of lists as per seed interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
