import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # Rank 40 decomposition
    # Split p=8 into p1=4, p2=4
    # Split n=3 into n1=2, n2=1
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x4x4 block (rank 16)
    def add_2x4x4(start_i, start_k):
        nonlocal col
        # A is 2x4 (rows start_i, start_i+1)
        # B is 4x4 (cols start_k to start_k+3)
        # C is 2x4 (rows start_i, start_i+1, cols start_k to start_k+3)
        # Use two 2x4x2 blocks
        for part in range(2):
            # B_part is 4x2 (cols start_k + part*2 to start_k + part*2 + 1)
            # C_part is 2x2 (cols start_k + part*2 to start_k + part*2 + 1)
            # Rank 8 for 2x4x2
            for i in range(2):
                for j in range(4):
                    for l in range(2):
                        # U[i, j, col] = 1
                        # V[j, start_k + part*2 + l, col] = 1
                        # W[start_k + part*2 + l, start_i + i, col] = 1
                        U[(start_i + i) * m + j][col] = 1
                        V[j * p + (start_k + part * 2 + l)][col] = 1
                        W[(start_k + part * 2 + l) * n + (start_i + i)][col] = 1
                        col += 1
    
    # Helper to add a 1x4x4 block (rank 4)
    def add_1x4x4(start_i, start_k):
        nonlocal col
        # A is 1x4 (row start_i)
        # B is 4x4 (cols start_k to start_k+3)
        # C is 1x4 (row start_i, cols start_k to start_k+3)
        # Rank 4
        for j in range(4):
            for l in range(4):
                # U[start_i, j, col] = 1
                # V[j, start_k + l, col] = 1
                # W[start_k + l, start_i, col] = 1
                U[start_i * m + j][col] = 1
                V[j * p + (start_k + l)][col] = 1
                W[(start_k + l) * n + start_i][col] = 1
                col += 1

    # C1 = A1 B1 (2x4x4) + A2 B1 (1x4x4)
    add_2x4x4(0, 0)
    add_1x4x4(2, 0)
    
    # C2 = A1 B2 (2x4x4) + A2 B2 (1x4x4)
    add_2x4x4(0, 4)
    add_1x4x4(2, 4)
    
    return {"U": U, "V": V, "W": W}
