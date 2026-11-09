import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication 3x4 by 4x7.
    Rank is 3*4*7 = 84.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Using lists of lists as per seed, but we can use numpy for speed if needed.
    # However, the output interface expects lists of lists or similar iterable.
    # The seed uses lists. I will stick to lists for compatibility unless numpy is required.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Schoolbook term: A[i,j] * B[j,k] contributes to C[i,k]
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (per trace convention W[k*n+i])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check time
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
