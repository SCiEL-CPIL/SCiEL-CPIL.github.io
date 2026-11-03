"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = 1 for all i,j,k
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank is n*m*p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Optional: Validation to be safe, though schoolbook is always valid
    # if not _validate_decomposition(U, V, W, n, m, p):
    #     raise ValueError("Schoolbook decomposition failed validation")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
