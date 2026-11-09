"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for M_{2,3,9}.
    
    The matrix multiplication tensor for (n,m,p) = (2,3,9) has a trivial
    decomposition of rank n*m*p = 54. We return this canonical decomposition.
    
    The trace convention: W[k*n + i, r] where the output element at position
    (i, k) in the n x p result is computed as sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r].
    Actually, the standard convention for matrix multiplication C = A * B where
    A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T_{i,j,k} = delta_{i,j,k} (in the appropriate sense) is decomposed as:
    T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the trivial decomposition, each term in the sum corresponds to a unique (i,j,k):
    U[i*m+j, r] = 1 when r encodes (i,j,k)
    V[j*p+k, r] = 1 when r encodes (i,j,k)  
    W[k*n+i, r] = 1 when r encodes (i,j,k)
    
    This gives rank = n*m*p = 54.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
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
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
