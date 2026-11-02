import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 60.
    # Split p=8 into two blocks of size 4.
    # Each block of size 4 is split into 3 and 1.
    # Block 3: Schoolbook M_{3,3,3} rank 27.
    # Block 1: Schoolbook M_{3,3,1} rank 3.
    # Total per block: 30.
    # Two blocks: 60.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add schoolbook terms for M_{n,m,p_sub}
    # offset_k: starting k index
    # offset_col: starting column index in U,V,W
    def add_schoolbook(n, m, p_sub, offset_k, offset_col):
        col = offset_col
        for i in range(n):
            for j in range(m):
                for k in range(p_sub):
                    # U index: i*m + j
                    U[i * m + j][col] = 1
                    # V index: j*p + (offset_k + k)
                    V[j * p + offset_k + k][col] = 1
                    # W index: (offset_k + k)*n + i
                    W[(offset_k + k) * n + i][col] = 1
                    col += 1
        return col
    
    # Block 1: k in [0, 3]
    # Sub-block 1.1: k in [0, 2] (M_{3,3,3})
    col = add_schoolbook(n, m, 3, 0, 0)
    # Sub-block 1.2: k in [3, 3] (M_{3,3,1})
    col = add_schoolbook(n, m, 1, 3, col)
    
    # Block 2: k in [4, 7]
    # Sub-block 2.1: k in [4, 6] (M_{3,3,3})
    col = add_schoolbook(n, m, 3, 4, col)
    # Sub-block 2.2: k in [7, 7] (M_{3,3,1})
    col = add_schoolbook(n, m, 1, 7, col)
    
    return {"U": U, "V": V, "W": W}
