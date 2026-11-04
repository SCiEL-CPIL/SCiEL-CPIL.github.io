import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # Rank 49 (for 4x4x4) + 8 (for 4x4x2) = 57
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 steps
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Part 1: 4x4x4 (cols 0-3 of B, cols 0-3 of C)
    col_start_1 = 0
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = col_start_1 + s * 7 + s2
            
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 (cols 4-5 of B, cols 4-5 of C)
    # Rank 4 for each of the 2 columns.
    col_start_2 = 49
    
    # For column 4 (k=0 relative to part 2, global col 4)
    # C_{i,4} = sum_j A_{i,j} B_{j,4}
    # Use 4 terms, one for each j?
    # Term j=0:
    # U[ (i,0) ] = 1, V[ (0,4) ] = 1, W[ (4, i) ] = 1
    # This captures A[i,0] B[0,4] -> C[i,4]
    
    for j in range(m): # j=0..3
        # Term for j
        col = col_start_2 + j
        
        # U: A[i,j]
        for i in range(n):
            r = i * m + j
            U[r][col] = 1
            
        # V: B[j,4]
        r = j * p + 4
        V[r][col] = 1
        
        # W: C[4,i]
        for i in range(n):
            r = 4 * n + i
            W[r][col] = 1
            
    # For column 5 (k=1 relative to part 2, global col 5)
    for j in range(m): # j=0..3
        col = col_start_2 + 4 + j
        
        # U: A[i,j]
        for i in range(n):
            r = i * m + j
            U[r][col] = 1
            
        # V: B[j,5]
        r = j * p + 5
        V[r][col] = 1
        
        # W: C[5,i]
        for i in range(n):
            r = 5 * n + i
            W[r][col] = 1

    return {"U": U, "V": V, "W": W}
