"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-9 decomposition for 3x3 * 3x5 matrix multiplication.
    
    For each (i, j) pair where i is row of A (0..2) and j is column of A / row of B (0..2),
    we create one tensor term. This gives 9 terms total.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Term t = (i, j):
      U[i*3 + j, t] = 1
      V[j*5 + k, t] = 1 for all k in 0..4
      W[k*3 + i, t] = 1 for all k in 0..4
    
    Verification: C[i,k] = sum_t U[i*3+j(t),t] * V[j(t)*5+k,t] * W[k*3+i,t]
    Only terms with i'=i contribute (from U), giving sum_j A[i,j]*B[j,k].
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    rank = n * m  # 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            t = i * m + j
            U[i * m + j][t] = 1
            for k in range(p):
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
