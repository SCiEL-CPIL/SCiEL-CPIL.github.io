<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # 150s limit, keep margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists for exact integer arithmetic
    # Using lists of lists allows for easy serialization and exact arithmetic
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time
                if time.time() > deadline - 10:
                    # Return current state if close to deadline (though construction is fast)
                    break
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    result = {"U": U, "V": V, "W": W}
    return result
# EVOLVE-BLOCK-END
=======
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # 150s limit, keep margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    
    # Initialize U, V, W as NumPy arrays for exact integer arithmetic
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time
                if time.time() > deadline - 10:
                    break
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    result = {"U": U, "V": V, "W": W}
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
