"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    
    The decomposition satisfies: for all i,j,k, the contribution A[i,j]*B[j,k] to C[i,k]
    is captured by the tensor product structure.
    
    Using schoolbook: rank = n*m*p = 48
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 48)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 48)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 48)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k  
                # W index: k*n + i (per the trace convention W[k*n+i, r])
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
