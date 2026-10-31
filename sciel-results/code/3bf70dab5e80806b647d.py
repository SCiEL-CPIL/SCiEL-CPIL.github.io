"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Recursive Strassen decomposition for 4x4 matrix multiplication, rank 49."""
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    # Outer Strassen: 4x4 as 2x2 blocks of 2x2
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2
    # C = [[C00, C01], [C10, C11]] where Cij are 2x2
    
    # Outer M1 = (A00+A11)(B00+B11)
    # Outer M2 = (A10+A11)B00
    # Outer M3 = A00(B01-B11)
    # Outer M4 = A11(B10-B00)
    # Outer M5 = (A00+A01)B11
    # Outer M6 = (A10-A00)(B00+B01)
    # Outer M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 outer Strassen terms
    # Each term: X (2x2 of A-entry linear combos), Y (2x2 of B-entry linear combos)
    # X[i][j] = list of (a_row, a_col, coeff)
    # Y[i][j] = list of (b_row, b_col, coeff)
    
    outer_terms = []
    
    # M1: X = A00+A11, Y = B00+B11
    X = [[[(i, j, 1), (2+i, 2+j, 1)] for j in range(2)] for i in range(2)]
    Y = [[[(i, j, 1), (2+i, 2+j, 1)] for j in range(2)] for i in range(2)]
    outer_terms.append((X, Y))
    
    # M2: X = A10+A11, Y = B00
    X = [[[(2+i, j, 1), (2+i, 2+j, 1)] for j in range(2)] for i in range(2)]
    Y = [[[(i, j, 1)] for j in range(2)] for i in range(2)]
    outer_terms.append((X, Y))
    
    # M3: X = A00, Y = B01-B11
    X = [[[(i, j, 1)] for j in range(2)] for i in range(2)]
    Y = [[[(i, 2+j, 1), (2+i, 2+j, -1)] for j in range(2)] for i in range(2)]
    outer_terms.append((X, Y))
    
    # M4: X = A11, Y = B10-B00
    X = [[[(2+i, 2+j, 1)] for j in range(2)] for i in range(2)]
    Y = [[[(2+i, j, 1), (i, j, -1)] for j in range(2)] for i in range(2)]
    outer_terms.append((X, Y))
    
    # M5: X = A00+A01, Y = B11
    X = [[[(i, j, 1), (i, 2+j, 1)] for j in range(2)] for i in range(2)]
    Y = [[[(2+i, 2+j, 1)] for j in range(2)] for i in range(2)]
    outer_terms.append((X, Y))
    
    # M6: X = A10-A00, Y = B00+B01
    X = [[[(2+i, j, 1), (i, j, -1)] for j in range(2)] for i in range(2)]
    Y = [[[(i, j, 1), (i, 2+j, 1)] for j in range(2)] for i in range(2)]
    outer_terms.append((X, Y))
    
    # M7: X = A01-A11, Y = B10+B11
    X = [[[(i, 2+j, 1), (2+i, 2+j, -1)] for j in range(2)] for i in range(2)]
    Y = [[[(2+i, j, 1), (2+i, 2+j, 1)] for j in range(2)] for i in range(2)]
    outer_terms.append((X, Y))
    
    # C targets for each outer term
    c_targets = [
        [(0,0,1), (1,1,1)],       # M1 -> C00, C11
        [(1,0,1), (1,1,-1)],      # M2 -> C10, C11
        [(0,1,1), (1,1,1)],       # M3 -> C01, C11
        [(0,0,1), (1,0,1)],       # M4 -> C00, C10
        [(0,0,-1), (0,1,1)],      # M5 -> C00, C01
        [(1,1,1)],                 # M6 -> C11
        [(0,0,1)],                 # M7 -> C00
    ]
    
    def w_idx(block_r, block_c, local_r, local_c):
        row = block_r * 2 + local_r
        col = block_c * 2 + local_c
        return col * n + row
    
    def add_combos(c1, c2):
        result = {}
        for (r, c, v) in c1:
            result[(r,c)] = result.get((r,c), 0) + v
        for (r, c, v) in c2:
            result[(r,c)] = result.get((r,c), 0) + v
        return [(r, c, v) for (r,c), v in result.items() if v != 0]
    
    def sub_combos(c1, c2):
        return add_combos(c1, [(r, c, -v) for (r,c,v) in c2])
    
    # For each outer term, apply inner Strassen to compute X*Y
    col = 0
    for t in range(7):
        X = outer_terms[t][0]
        Y = outer_terms[t][1]
        
        # Inner Strassen: compute X*Y using 7 multiplications
        # Inner M1 = (X00+X11)(Y00+Y11)
        # Inner M2 = (X10+X11)Y00
        # Inner M3 = X00(Y01-Y11)
        # Inner M4 = X11(Y10-Y00)
        # Inner M5 = (X00+X01)Y11
        # Inner M6 = (X10-X00)(Y00+Y01)
        # Inner M7 = (X01-X11)(Y10+Y11)
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # Inner M1: (X00+X11)(Y00+Y11)
        X_in = add_combos(X[0][0], X[1][1])
        Y_in = add_combos(Y[0][0], Y[1][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] += coeff
            W[w_idx(br, bc, 1, 1)][col] += coeff
        col += 1
        
        # Inner M2: (X10+X11)Y00
        X_in = add_combos(X[1][0], X[1][1])
        Y_in = Y[0][0]
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 1, 0)][col] += coeff
            W[w_idx(br, bc, 1, 1)][col] -= coeff
        col += 1
        
        # Inner M3: X00(Y01-Y11)
        X_in = X[0][0]
        Y_in = sub_combos(Y[0][1], Y[1][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 1)][col] += coeff
            W[w_idx(br, bc, 1, 1)][col] += coeff
        col += 1
        
        # Inner M4: X11(Y10-Y00)
        X_in = X[1][1]
        Y_in = sub_combos(Y[1][0], Y[0][0])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] += coeff
            W[w_idx(br, bc, 1, 0)][col] += coeff
        col += 1
        
        # Inner M5: (X00+X01)Y11
        X_in = add_combos(X[0][0], X[0][1])
        Y_in = Y[1][1]
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] -= coeff
            W[w_idx(br, bc, 0, 1)][col] += coeff
        col += 1
        
        # Inner M6: (X10-X00)(Y00+Y01)
        X_in = sub_combos(X[1][0], X[0][0])
        Y_in = add_combos(Y[0][0], Y[0][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 1, 1)][col] += coeff
        col += 1
        
        # Inner M7: (X01-X11)(Y10+Y11)
        X_in = sub_combos(X[0][1], X[1][1])
        Y_in = add_combos(Y[1][0], Y[1][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] += coeff
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
