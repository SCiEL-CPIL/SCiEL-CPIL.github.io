"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    This uses the recursive Strassen algorithm (7 outer Strassen steps, 
    each using 7 inner Strassen steps, total 49 multiplications).
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    rank = 49
    # Initialize matrices with zeros
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add combinations of indices
    # A entry is (row, col, coeff)
    def add_combos(c1, c2):
        result = {}
        for (r, c, v) in c1:
            result[(r,c)] = result.get((r,c), 0) + v
        for (r, c, v) in c2:
            result[(r,c)] = result.get((r,c), 0) + v
        return [(r, c, v) for (r,c), v in result.items() if v != 0]
    
    def sub_combos(c1, c2):
        return add_combos(c1, [(r, c, -v) for (r,c,v) in c2])
    
    # Define the 7 outer Strassen terms for 4x4 (blocks of 2x2)
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # Each Aij, Bij is 2x2.
    # Indices for A: (0,0) to (1,1) is A00, (0,2) to (1,3) is A01, etc.
    
    # We represent linear combinations of A entries as lists of (row, col, coeff)
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # c_targets[t] defines how the result of outer term t contributes to C blocks
    # Each entry is (block_row, block_col, coeff)
    # Block indices: (0,0)=C00, (0,1)=C01, (1,0)=C10, (1,1)=C11
    c_targets = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1), (1,1,-1)],      # M2
        [(0,1,1), (1,1,1)],       # M3
        [(0,0,1), (1,0,1)],       # M4
        [(0,0,-1), (0,1,1)],      # M5
        [(1,1,1)],                 # M6
        [(0,0,1)],                 # M7
    ]
    
    # Define the 7 outer terms X, Y
    # X is a 2x2 matrix of linear combos of A entries
    # Y is a 2x2 matrix of linear combos of B entries
    
    # Helper to get A block entries
    def a_block(r, c):
        return [[(r+i, c+j, 1) for j in range(2)] for i in range(2)]
        
    def b_block(r, c):
        return [[(r+i, c+j, 1) for j in range(2)] for i in range(2)]

    A00 = a_block(0,0); A01 = a_block(0,2)
    A10 = a_block(2,0); A11 = a_block(2,2)
    
    B00 = b_block(0,0); B01 = b_block(0,2)
    B10 = b_block(2,0); B11 = b_block(2,2)
    
    outer_terms = [
        (add_combos(A00, A11), add_combos(B00, B11)), # M1
        (add_combos(A10, A11), B00),                  # M2
        (A00, sub_combos(B01, B11)),                  # M3
        (A11, sub_combos(B10, B00)),                  # M4
        (add_combos(A00, A01), B11),                  # M5
        (sub_combos(A10, A00), add_combos(B00, B01)), # M6
        (sub_combos(A01, A11), add_combos(B10, B11)), # M7
    ]
    
    # Helper to map C block (br, bc) and local (lr, lc) to W index
    # W index is col * n + row
    # C is 4x4.
    # Block (br, bc) covers rows br*2 to br*2+1, cols bc*2 to bc*2+1
    def w_idx(br, bc, lr, lc):
        row = br * 2 + lr
        col = bc * 2 + lc
        return col * n + row

    col = 0
    for t in range(7):
        if time.time() - start_time > timeout:
            break
            
        X = outer_terms[t][0]
        Y = outer_terms[t][1]
        
        # Inner Strassen on X (2x2) and Y (2x2)
        # X00, X01, X10, X11 are 2x2 blocks of linear combos
        # Wait, X itself is a 2x2 matrix of linear combos.
        # So X[0][0] is the top-left element of the 2x2 matrix X.
        # This element is itself a linear combo of A entries.
        
        # Inner M1 = (X00+X11)(Y00+Y11)
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
        
        # Inner M2 = (X10+X11)Y00
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
        
        # Inner M3 = X00(Y01-Y11)
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
        
        # Inner M4 = X11(Y10-Y00)
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
        
        # Inner M5 = (X00+X01)Y11
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
        
        # Inner M6 = (X10-X00)(Y00+Y01)
        X_in = sub_combos(X[1][0], X[0][0])
        Y_in = add_combos(Y[0][0], Y[0][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 1, 1)][col] += coeff
        col += 1
        
        # Inner M7 = (X01-X11)(Y10+Y11)
        X_in = sub_combos(X[0][1], X[1][1])
        Y_in = add_combos(Y[1][0], Y[1][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] += coeff
        col += 1
        
    # If we timed out, we might have partial results. 
    # But the algorithm is deterministic and fast, so this shouldn't happen.
    # We return the full rank 49 solution.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
