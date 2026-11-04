import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x4 by 4x5.
    Uses the schoolbook algorithm which yields a rank of n*m*p.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices U, V, W with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i
                W[k * n + i][column] = 1
                column += 1
                if time.time() > deadline:
                    # If we run out of time, return what we have (should not happen for O(1) size)
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
