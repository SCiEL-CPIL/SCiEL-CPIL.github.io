import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen components for 2x2x2
    # u, v, w are lists of 4 coefficients corresponding to indices 00, 01, 10, 11
    strassen_components = [
        # (u, v, w)
        ([(1,0,0,1), (1,0,0,1), (1,1,1,1)], (1,0,0,1), (1,0,0,1), (1,1,1,1)), # This format is wrong, let's use lists of lists
    ]
    
    # Correct Strassen components (u, v, w) as tuples of 4 floats
    S = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 1, 1, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 1)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
        ((0, 0, 0, 1), (-1, 0, 1, 0), (-1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, -1)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]
    
    # Helper to add a Strassen block at offset (oi, oj, ok)
    def add_strassen_block(oi, oj, ok):
        nonlocal col
        for u, v, w in S:
            # u corresponds to A indices (li, lj)
            # v corresponds to B indices (lj, lk)
            # w corresponds to C indices (lk, li)
            
            # Map local indices to global
            # A: (oi+li, oj+lj)
            # B: (oj+lj, ok+lk)
            # C: (ok+lk, oi+li)
            
            for li in range(2):
                for lj in range(2):
                    idx_u = (oi + li) * m + (oj + lj)
                    U[idx_u][col] = u[li * 2 + lj]
                    
            for lj in range(2):
                for lk in range(2):
                    idx_v = (oj + lj) * p + (ok + lk)
                    V[idx_v][col] = v[lj * 2 + lk]
                    
            for lk in range(2):
                for li in range(2):
                    idx_w = (ok + lk) * n + (oi + li)
                    W[idx_w][col] = w[lk * 2 + li]
            
            col += 1
            
    # Add Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    
    # Add Block 2: i in {2,3}, j in {2,3}, k in {2,3}
    add_strassen_block(2, 2, 2)
    
    # Add Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if in S1 or S2
                in_s1 = (i < 2 and j < 2 and k < 2)
                in_s2 = (i >= 2 and i < 4 and j >= 2 and j < 4 and k >= 2 and k < 4)
                
                if not in_s1 and not in_s2:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
