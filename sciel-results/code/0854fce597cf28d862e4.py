"""Matrix multiplication 2x2x12: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return a rank-48 exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x12 matrix.
    
    The decomposition uses the schoolbook structure.
    Index conventions:
      U: (n*m, rank) -- U[i*m + j, c]
      V: (m*p, rank) -- V[j*p + k, c]  
      W: (p*n, rank) -- W[k*n + i, c]
    
    We use a unique column index c for each (i, j, k) triple to avoid
    collisions that create spurious non-zero entries in the tensor.
    c = i * m * p + j * p + k
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Unique column index for the term A[i,j] * B[j,k] -> C[i,k]
                c = i * m * p + j * p + k
                
                U[i * m + j][c] = 1  # A[i][j]
                V[j * p + k][c] = 1  # B[j][k]
                W[k * n + i][c] = 1  # C[i][k]
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
