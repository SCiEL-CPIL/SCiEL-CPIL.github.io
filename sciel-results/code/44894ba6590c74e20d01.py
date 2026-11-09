"""Matrix multiplication 3x4x5 with rank-60 exact tensor decomposition."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return an exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    
    The output factor W follows the trace convention W[k*n+i, r].
    
    We implement the schoolbook decomposition with rank n*m*p = 60.
    Each term corresponds to a single (i, j, k) product in the standard
    matrix multiplication formula C[i,k] = sum_j A[i,j] * B[j,k].
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook decomposition of rank n*m*p = 60
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over the inner dimension j first, then row i, then col k
    # to ensure proper mapping. Each (i, j, k) triple gets one rank slot.
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                # U[i*m + j, col] = 1  (selects A[i,j])
                U[i * m + j][col] = 1
                # V[j*p + k, col] = 1  (selects B[j,k])
                # W[k*n + i, col] = 1  (writes to C[i,k] at position k*n+i)
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
