"""
Program for 5x5 Matrix Multiplication Tensor Decomposition.

The goal is to find a rank-minimal exact tensor decomposition for multiplying
a 5x5 matrix by a 5x5 matrix.
The schoolbook algorithm provides a decomposition of rank n*m*p = 125.
No decomposition of rank < 125 is currently known.

Interface:
construct() -> {'U', 'V', 'W'}
U: shape (n*m, rank)
V: shape (m*p, rank)
W: shape (p*n, rank)
Entries are integers.
"""

import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the reference implementation style
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j
                # V row index: j*p + k
                # W row index: k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Safety check (not strictly necessary for this small O(1) task)
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
