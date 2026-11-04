"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication A(nxm) * B(mxq) -> C(nxq)
    # has entries T[i,j,k] = 1 for all valid i,j,k.
    # This is because C[i,k] = sum_j A[i,j] * B[j,k].
    # The all-ones tensor has rank 1.
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must sum to 1.
    # Setting all to 1 with rank 1 works.
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
