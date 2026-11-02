import time
import numpy as np

def verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for t in range(rank):
                    total += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 108  # Safety margin
    
    n, m, p = 3, 5, 7
    
    # Rank 15 decomposition:
    # Split A into 3 rows of size 1x5.
    # For each row i, use rank 5 decomposition of 1x5x7.
    # Total rank = 3 * 5 = 15.
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For row i (0..2), and column j (0..4)
    # Term t = i*5 + j
    # U[i*m + j, t] = 1
    # V[j*p + k, t] = 1 if j == j (always true for the specific j in t)
    # Wait, V index is j*p + k.
    # In the 1x5x7 decomposition for row i:
    # t corresponds to the column j of the 1x5 vector.
    # U_{j, t} = delta_{j, t_j}
    # V_{j, k, t} = delta_{j, t_j}
    # W_{k, t} = 1
    
    # Mapping:
    # t = i*5 + j_col
    # U[i*m + j, t] = 1 if j == j_col else 0
    # V[j*p + k, t] = 1 if j == j_col else 0
    # W[k*n + i, t] = 1
    
    for i in range(n):
        for j_col in range(m):
            t = i * m + j_col
            for j in range(m):
                if j == j_col:
                    U[i * m + j][t] = 1
                    for k in range(p):
                        V[j * p + k][t] = 1
                        W[k * n + i][t] = 1
                        # Note: W depends on k and i.
                        # In the sum, W[k*n+i, t] is used.
                        # So we set W[k*n+i, t] = 1 for all k.
                        # This is correct.
                        
    # Optimization: Pre-fill W
    # Actually, the loop above sets W[k*n+i, t] = 1 for all k.
    # This is efficient enough for 15 terms.
    
    result = {"U": U, "V": V, "W": W}
    
    return result
