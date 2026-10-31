import time
import numpy as np

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen: 7 outer multiplications of 2x2 blocks,
    each computed via 7 inner scalar multiplications.
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to create a 2x2 block identity or zero matrix of linear combos
    # A 2x2 matrix is represented as a list of lists of (global_row, global_col, coeff)
    
    def get_block_A(br, bc):
        """Returns 2x2 matrix of linear combos for A block at (br, bc)"""
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = [(br*2+i, bc*2+j, 1)]
        return res

    def get_block_B(br, bc):
        """Returns 2x2 matrix of linear combos for B block at (br, bc)"""
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = [(br*2+i, bc*2+j, 1)]
        return res

    def add_mat(X, Y):
        res = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                d = {}
                for r, c, v in X[i][j]:
                    d[(r,c)] = d.get((r,c), 0) + v
                for r, c, v in Y[i][j]:
                    d[(r,c)] = d.get((r,c), 0) + v
                res[i][j] = [(r,c,v) for (r,c),v in d.items() if v != 0]
        return res

    def sub_mat(X, Y):
        neg_Y = [[[(r, c, -v) for (r,c,v) in Y[i][j]] for j in range(2)] for i in range(2)]
        return add_mat(X, neg_Y)

    # Define the 7 Strassen terms for the 4x4 multiplication
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    A00 = get_block_A(0, 0)
    A01 = get_block_A(0, 1)
    A10 = get_block_A(1, 0)
    A11 = get_block_A(1, 1)
    
    B00 = get_block_B(0, 0)
    B01 = get_block_B(0, 1)
    B10 = get_block_B(1, 0)
    B11 = get_block_B(1, 1)

    # Outer Strassen linear combinations
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    outer_X = [
        add_mat(A00, A11),
        add_mat(A10, A11),
        A00,
        A11,
        add_mat(A00, A01),
        sub_mat(A10, A00),
        sub_mat(A01, A11)
    ]
    
    outer_Y = [
        add_mat(B00, B11),
        B00,
        sub_mat(B01, B11),
        sub_mat(B10, B00),
        B11,
        add_mat(B00, B01),
        add_mat(B10, B11)
    ]

    # For each outer term, we need to decompose the 2x2 * 2x2 multiplication
    # into 7 scalar multiplications.
    # The result of each scalar multiplication contributes to the C blocks.
    
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 7 inner Strassen terms for a generic 2x2 multiplication X * Y
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X * Y
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # m1 = (x00+x11)*(y00+y11)
    # m2 = (x10+x11)*y00
    # m3 = x00*(y01-y11)
    # m4 = x11*(y10-y00)
    # m5 = (x00+x01)*y11
    # m6 = (x10-x00)*(y00+y01)
    # m7 = (x01-x11)*(y10+y11)
    
    # We will iterate over the 7 outer terms.
    # For each outer term t, we have X_t and Y_t which are 2x2 matrices of linear combos.
    # We apply the inner Strassen logic to X_t and Y_t.
    
    col = 0
    for t in range(7):
        X = outer_X[t]
        Y = outer_Y[t]
        
        # Extract entries for inner Strassen
        # X is a 2x2 grid of lists of (row, col, coeff)
        
        def entry_sum(X, i1, j1, i2, j2, sign=1):
            # Sum entries (i1,j1) and (i2,j2) with sign
            res = []
            d = {}
            for r, c, v in X[i1][j1]:
                d[(r,c)] = d.get((r,c), 0) + v
            if i2 != i1 or j2 != j1:
                for r, c, v in X[i2][j2]:
                    d[(r,c)] = d.get((r,c), 0) + sign * v
            # If i2==i1 and j2==j1, we just take X[i1][j1]
            # But usually in Strassen, we add distinct blocks.
            # Here X is already a matrix of linear combos.
            # The "entries" of X are the linear combos.
            # So X[i][j] is the linear combo for the (i,j) element of the 2x2 block.
            # The inner Strassen operates on these linear combos.
            # e.g. x00 = X[0][0], x11 = X[1][1].
            # m1_x = x00 + x11. This means we sum the linear combos in X[0][0] and X[1][1].
            pass
            return d

        # Define helper to get linear combo for a specific inner Strassen term
        def get_inner_X_term(k):
            # Returns dict {(global_r, global_c): coeff}
            d = {}
            if k == 0: # x00 + x11
                for r,c,v in X[0][0]: d[(r,c)] = d.get((r,c),0) + v
                for r,c,v in X[1][1]: d[(r,c)] = d.get((r,c),0) + v
            elif k == 1: # x10 + x11
                for r,c,v in X[1][0]: d[(r,c)] = d.get((r,c),0) + v
                for r,c,v in X[1][1]: d[(r,c)] = d.get((r,c),0) + v
            elif k == 2: # x00
                for r,c,v in X[0][0]: d[(r,c)] = d.get((r,c),0) + v
            elif k == 3: # x11
                for r,c,v in X[1][1]: d[(r,c)] = d.get((r,c),0) + v
            elif k == 4: # x00 + x01
                for r,c,v in X[0][0]: d[(r,c)] = d.get((r,c),0) + v
                for r,c,v in X[0][1]: d[(r,c)] = d.get((r,c),0) + v
            elif k == 5: # x10 - x00
                for r,c,v in X[1][0]: d[(r,c)] = d.get((r,c),0) + v
                for r,c,v in X[0][0]: d[(r,c)] = d.get((r,c),0) - v
            elif k == 6: # x01 - x11
                for r,c,v in X[0][1]: d[(r,c)] = d.get((r,c),0) + v
                for r,c,v in X[1][1]: d[(r,c)] = d.get((r,c),0) - v
            return d

        def get_inner_Y_term(k):
            d = {}
            if k == 0: # y00 + y11
                for r,c,v in Y[0][0]: d[(r,c)] = d.get((r,c),0) + v
                for r,c,v in Y[1][1]: d[(r,c)] = d.get((r,c),0) + v
            elif k == 1: # y00
                for r,c,v in Y[0][0]: d[(r,c)] = d.get((r,c),0) + v
            elif k == 2: # y01 - y11
                for r,c,v in Y[0][1]: d[(r,c)] = d.get((r,c),0) + v
                for r,c,v in Y[1][1]: d[(r,c)] = d.get((r,c),0) - v
            elif k == 3: # y10 - y00
                for r,c,v in Y[1][0]: d[(r,c)] = d.get((r,c),0) + v
                for r,c,v in Y[0][0]: d[(r,c)] = d.get((r,c),0) - v
            elif k == 4: # y11
                for r,c,v in Y[1][1]: d[(r,c)] = d.get((r,c),0) + v
            elif k == 5: # y00 + y01
                for r,c,v in Y[0][0]: d[(r,c)] = d.get((r,c),0) + v
                for r,c,v in Y[0][1]: d[(r,c)] = d.get((r,c),0) + v
            elif k == 6: # y10 + y11
                for r,c,v in Y[1][0]: d[(r,c)] = d.get((r,c),0) + v
                for r,c,v in Y[1][1]: d[(r,c)] = d.get((r,c),0) + v
            return d

        # Contribution of inner term k to the 2x2 result block (t)
        # Z_t = X_t * Y_t
        # The result block Z_t is placed in the C matrix.
        # The outer term t contributes to specific C blocks with specific signs.
        
        # C00 gets: t=0(+), t=3(+), t=4(-), t=6(+)
        # C01 gets: t=2(+), t=4(+)
        # C10 gets: t=1(+), t=3(+)
        # C11 gets: t=0(+), t=1(-), t=2(+), t=5(+)
        
        # Let's define the target C blocks for each outer term t
        # c_targets[t] = list of (block_row, block_col, coeff)
        c_targets = {
            0: [(0,0,1), (1,1,1)],
            1: [(1,0,1), (1,1,-1)],
            2: [(0,1,1), (1,1,1)],
            3: [(0,0,1), (1,0,1)],
            4: [(0,0,-1), (0,1,1)],
            5: [(1,1,1)],
            6: [(0,0,1)]
        }
        
        # For each inner term k in the 7, we have a scalar multiplication.
        # This scalar multiplication produces a scalar value which is added to the
        # corresponding entries of the 2x2 block Z_t.
        # The 2x2 block Z_t has entries z00, z01, z10, z11.
        # Inner term k contributes to Z_t entries as per Strassen formulas:
        # z00: k=0(+), k=3(+), k=4(-), k=6(+)
        # z01: k=2(+), k=4(+)
        # z10: k=1(+), k=3(+)
        # z11: k=0(+), k=1(-), k=2(+), k=5(+)
        
        inner_targets = {
            0: [(0,0,1), (1,1,1)],
            1: [(1,0,1), (1,1,-1)],
            2: [(0,1,1), (1,1,1)],
            3: [(0,0,1), (1,0,1)],
            4: [(0,0,-1), (0,1,1)],
            5: [(1,1,1)],
            6: [(0,0,1)]
        }
        
        for k in range(7):
            # Get the linear combinations for U and V for this specific term
            u_dict = get_inner_X_term(k)
            v_dict = get_inner_Y_term(k)
            
            # Populate U
            for (ar, ac), val in u_dict.items():
                if val != 0:
                    U[ar * m + ac, col] += val
            
            # Populate V
            for (br, bc), val in v_dict.items():
                if val != 0:
                    V[br * p + bc, col] += val
            
            # Populate W
            # The result of the scalar multiplication (u dot v) is added to
            # the C entries determined by the outer target (t) and inner target (k).
            # Outer target gives the block (br, bc) and coeff c_coeff.
            # Inner target gives the local position (li, lj) in that block and coeff z_coeff.
            # Global C row = br * 2 + li
            # Global C col = bc * 2 + lj
            # W index = Global_C_col * n + Global_C_row (based on previous successful logic)
            
            for (obr, ocb, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    global_row = obr * 2 + li
                    global_col = ocb * 2 + lj
                    w_idx = global_col * n + global_row
                    W[w_idx, col] += c_coeff * z_coeff
            
            col += 1
            
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
