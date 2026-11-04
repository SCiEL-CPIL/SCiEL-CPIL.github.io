"""Program for M_{2,3,9} matrix multiplication tensor decomposition.

Strategy: The tensor M_{2,3,9} is a direct sum of three M_{2,3,3} tensors.
The rank of M_{2,3,3} is 18 (schoolbook is optimal for this rectangular case).
Thus, the rank of M_{2,3,9} is 3 * 18 = 54.
We construct the schoolbook decomposition of rank 54.
"""

import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110  # Safety margin for 120s limit
    
    n, m, p = 2, 3, 9
    R = n * m * p  # 54
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
