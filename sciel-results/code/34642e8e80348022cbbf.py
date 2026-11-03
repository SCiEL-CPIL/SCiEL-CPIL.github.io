import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculated as 27 + 27 + 2 + 2 + 2 = 60
    rank = 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook 3x3x3 block
    def add_333_block(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i, start_col):
        # u_offset_i: row offset in A (0)
        # u_offset_j: col offset in A (0 or 3)
        # v_offset_j: row offset in B (0 or 3)
        # v_offset_k: col offset in B (0 or 3)
        # w_offset_k: col offset in C (0 or 3)
        # w_offset_i: row offset in C (0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U index: (i + u_offset_i) * m + (j + u_offset_j)
                    ui = (i + u_offset_i) * m + (j + u_offset_j)
                    # V index: (j + v_offset_j) * p + (k + v_offset_k)
                    vi = (j + v_offset_j) * p + (k + v_offset_k)
                    # W index: (k + w_offset_k) * n + (i + w_offset_i)
                    wi = (k + w_offset_k) * n + (i + w_offset_i)
                    
                    U[ui][start_col] = 1
                    V[vi][start_col] = 1
                    W[wi][start_col] = 1
                    start_col += 1
        return start_col

    # Helper to add an optimal 3x2x2 block (Rank 2)
    def add_322_block(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i, start_col):
        # u_offset_i: row offset in A (0)
        # u_offset_j: col offset in A (3)
        # v_offset_j: row offset in B (3)
        # v_offset_k: col offset in B (0, 2, or 4)
        # w_offset_k: col offset in C (0, 2, or 4)
        # w_offset_i: row offset in C (0)
        
        # Term 1: j_local=0
        for i in range(3):
            for k in range(2):
                ui = (i + u_offset_i) * m + (0 + u_offset_j)
                vi = (0 + v_offset_j) * p + (k + v_offset_k)
                wi = (k + w_offset_k) * n + (i + w_offset_i)
                
                U[ui][start_col] = 1
                V[vi][start_col] = 1
                W[wi][start_col] = 1
        start_col += 1
        
        # Term 2: j_local=1
        for i in range(3):
            for k in range(2):
                ui = (i + u_offset_i) * m + (1 + u_offset_j)
                vi = (1 + v_offset_j) * p + (k + v_offset_k)
                wi = (k + w_offset_k) * n + (i + w_offset_i)
                
                U[ui][start_col] = 1
                V[vi][start_col] = 1
                W[wi][start_col] = 1
        start_col += 1
        
        return start_col

    # 1. A1 B11: 3x3 A part (cols 0-2), 3x3 B part (rows 0-2, cols 0-2)
    # A indices: i=0..2, j=0..2. B indices: j=0..2, k=0..2. C indices: k=0..2, i=0..2.
    col = add_333_block(0, 0, 0, 0, 0, 0, col)
    
    # 2. A1 B12: 3x3 A part (cols 0-2), 3x3 B part (rows 0-2, cols 3-5)
    # A indices: i=0..2, j=0..2. B indices: j=0..2, k=3..5. C indices: k=3..5, i=0..2.
    col = add_333_block(0, 0, 0, 3, 3, 0, col)
    
    # 3. A2 B21: 3x2 A part (cols 3-4), 2x2 B part (rows 3-4, cols 0-1)
    # A indices: i=0..2, j=3..4. B indices: j=3..4, k=0..1. C indices: k=0..1, i=0..2.
    col = add_322_block(0, 3, 3, 0, 0, 0, col)
    
    # 4. A2 B22: 3x2 A part (cols 3-4), 2x2 B part (rows 3-4, cols 2-3)
    # A indices: i=0..2, j=3..4. B indices: j=3..4, k=2..3. C indices: k=2..3, i=0..2.
    col = add_322_block(0, 3, 3, 2, 2, 0, col)
    
    # 5. A2 B23: 3x2 A part (cols 3-4), 2x2 B part (rows 3-4, cols 4-5)
    # A indices: i=0..2, j=3..4. B indices: j=3..4, k=4..5. C indices: k=4..5, i=0..2.
    col = add_322_block(0, 3, 3, 4, 4, 0, col)
    
    return {"U": U, "V": V, "W": W}
