import numpy as np
import time

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
    
    def w_idx(i, k):
        # W[k*n+i, r]
        return k * n + i

    # Helper to add a term to U, V, W
    # term_u: list of (index_in_U, coeff)
    # term_v: list of (index_in_V, coeff)
    # term_w: list of (index_in_W, coeff)
    col = 0
    
    # Define Strassen steps for 2x2 matrices
    # Each step is a multiplication of two 2x2 matrices.
    # We need to express the inputs (linear combos of 4x4 entries) and outputs (linear combos of 4x4 C entries)
    
    # Outer Strassen steps
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each outer step t, we have 7 inner steps k.
    # We need to map the inner step's 2x2 inputs to the 4x4 A/B indices,
    # and the inner step's 2x2 outputs to the 4x4 C indices.
    
    # Let's define the 7 inner Strassen multiplications for a generic 2x2 matrix mult X*Y = Z
    # X, Y are 2x2 matrices of linear combinations of A/B elements.
    # Z is a 2x2 matrix of linear combinations of C elements.
    
    # Inner Strassen steps for X*Y:
    # m1 = (X00+X11)(Y00+Y11)
    # m2 = (X10+X11)Y00
    # m3 = X00(Y01-Y11)
    # m4 = X11(Y10-Y00)
    # m5 = (X00+X01)Y11
    # m6 = (X10-X00)(Y00+Y01)
    # m7 = (X01-X11)(Y10+Y11)
    
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    # We will iterate over the 7 outer steps and 7 inner steps.
    
    # Helper to get A indices for a 2x2 block
    # A is 4x4. Block (br, bc) is 2x2.
    # A[br*2+i, bc*2+j]
    def get_A_idx(br, bc, i, j):
        return (br*2 + i) * m + (bc*2 + j)
        
    def get_B_idx(br, bc, i, j):
        return (br*2 + i) * p + (bc*2 + j)
        
    def get_C_idx(br, bc, i, j):
        # C is p x n (4x4).
        # C[i, k] where i is row, k is col.
        # Block (br, bc) corresponds to C rows br*2..br*2+1 and cols bc*2..bc*2+1.
        # W index is k*n + i.
        return (bc*2 + j) * n + (br*2 + i)

    # Outer steps definition
    # Each outer step t defines X and Y as 2x2 matrices of A/B linear combos.
    # X[i][j] is a list of (A_idx, coeff)
    # Y[i][j] is a list of (B_idx, coeff)
    
    outer_steps = []
    
    # Helper to create a 2x2 matrix of single elements
    def make_single_A(br, bc):
        mat = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                mat[i][j] = [(get_A_idx(br, bc, i, j), 1)]
        return mat
        
    def make_single_B(br, bc):
        mat = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                mat[i][j] = [(get_B_idx(br, bc, i, j), 1)]
        return mat

    def add_mat(X, Y):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                d = {}
                for idx, v in X[i][j]:
                    d[idx] = d.get(idx, 0) + v
                for idx, v in Y[i][j]:
                    d[idx] = d.get(idx, 0) + v
                res[i][j] = [(idx, v) for idx, v in d.items() if v != 0]
        return res

    def sub_mat(X, Y):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                d = {}
                for idx, v in X[i][j]:
                    d[idx] = d.get(idx, 0) + v
                for idx, v in Y[i][j]:
                    d[idx] = d.get(idx, 0) - v
                res[i][j] = [(idx, v) for idx, v in d.items() if v != 0]
        return res

    A00 = make_single_A(0, 0)
    A01 = make_single_A(0, 1)
    A10 = make_single_A(1, 0)
    A11 = make_single_A(1, 1)
    
    B00 = make_single_B(0, 0)
    B01 = make_single_B(0, 1)
    B10 = make_single_B(1, 0)
    B11 = make_single_B(1, 1)

    # Outer M1
    X1 = add_mat(A00, A11)
    Y1 = add_mat(B00, B11)
    outer_steps.append((X1, Y1))
    
    # Outer M2
    X2 = add_mat(A10, A11)
    Y2 = B00
    outer_steps.append((X2, Y2))
    
    # Outer M3
    X3 = A00
    Y3 = sub_mat(B01, B11)
    outer_steps.append((X3, Y3))
    
    # Outer M4
    X4 = A11
    Y4 = sub_mat(B10, B00)
    outer_steps.append((X4, Y4))
    
    # Outer M5
    X5 = add_mat(A00, A01)
    Y5 = B11
    outer_steps.append((X5, Y5))
    
    # Outer M6
    X6 = sub_mat(A10, A00)
    Y6 = add_mat(B00, B01)
    outer_steps.append((X6, Y6))
    
    # Outer M7
    X7 = sub_mat(A01, A11)
    Y7 = add_mat(B10, B11)
    outer_steps.append((X7, Y7))

    # Inner Strassen steps for 2x2 matrices
    # We need to generate 7 multiplications for each outer step.
    # For a given outer step (X, Y), we compute 7 inner products.
    # Each inner product is a scalar multiplication of two linear combos.
    # We then add the contribution to W.
    
    # Inner step definitions (indices 0..6)
    # m1: (X00+X11)(Y00+Y11)
    # m2: (X10+X11)Y00
    # m3: X00(Y01-Y11)
    # m4: X11(Y10-Y00)
    # m5: (X00+X01)Y11
    # m6: (X10-X00)(Y00+Y01)
    # m7: (X01-X11)(Y10+Y11)
    
    # We need to combine X[i][j] and Y[i][j] which are already lists of (idx, coeff).
    # For inner step 1:
    # X_part = X[0][0] + X[1][1]
    # Y_part = Y[0][0] + Y[1][1]
    
    def combine_list(list1, list2, op='+'):
        d = {}
        for idx, v in list1:
            d[idx] = d.get(idx, 0) + v
        for idx, v in list2:
            if op == '+':
                d[idx] = d.get(idx, 0) + v
            else:
                d[idx] = d.get(idx, 0) - v
        return [(idx, v) for idx, v in d.items() if v != 0]

    inner_defs = []
    
    # m1
    inner_defs.append((
        lambda X: combine_list(X[0][0], X[1][1]),
        lambda Y: combine_list(Y[0][0], Y[1][1])
    ))
    # m2
    inner_defs.append((
        lambda X: combine_list(X[1][0], X[1][1]),
        lambda Y: Y[0][0]
    ))
    # m3
    inner_defs.append((
        lambda X: X[0][0],
        lambda Y: combine_list(Y[0][1], Y[1][1], '-')
    ))
    # m4
    inner_defs.append((
        lambda X: X[1][1],
        lambda Y: combine_list(Y[1][0], Y[0][0], '-')
    ))
    # m5
    inner_defs.append((
        lambda X: combine_list(X[0][0], X[0][1]),
        lambda Y: Y[1][1]
    ))
    # m6
    inner_defs.append((
        lambda X: combine_list(X[1][0], X[0][0], '-'),
        lambda Y: combine_list(Y[0][0], Y[0][1])
    ))
    # m7
    inner_defs.append((
        lambda X: combine_list(X[0][1], X[1][1], '-'),
        lambda Y: combine_list(Y[1][0], Y[1][1])
    ))

    # Contribution of inner products m1..m7 to Z (2x2 result)
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    # We need to map Z[i][j] to C indices.
    # Z is the result of multiplying block X (from A) and Y (from B).
    # The position of this block in C is determined by the outer step.
    # Wait, the outer steps M1..M7 are not directly blocks of C.
    # They are intermediate values.
    # The final C blocks are linear combinations of M1..M7.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So, for each outer step t (0..6), we have 7 inner products.
    # Each inner product contributes to a specific element of the 2x2 matrix M_t.
    # Let's say inner product k contributes to M_t[i][j] with coefficient c.
    # Then M_t[i][j] contributes to C[br*2+i, bc*2+j] with coefficient d,
    # where (br, bc) is the block position corresponding to M_t in the C formula.
    # This is tricky because M_t is a 2x2 matrix, and it contributes to multiple C blocks?
    # No. M_t is a 2x2 matrix.
    # In the formula C00 = M1 + ..., M1 is a 2x2 matrix.
    # So M1[0][0] contributes to C00[0][0], M1[0][1] to C00[0][1], etc.
    # And C00 is the top-left 2x2 block of C.
    
    # So, for outer step t, we have a 2x2 matrix M_t.
    # M_t[i][j] is computed by summing contributions from the 7 inner products.
    # Specifically:
    # M_t[0][0] = m1 + m4 - m5 + m7
    # M_t[0][1] = m3 + m5
    # M_t[1][0] = m2 + m4
    # M_t[1][1] = m1 - m2 + m3 + m6
    
    # And M_t itself is added to specific C blocks with coefficients.
    # Let's define the contribution of M_t to the 4 C blocks (C00, C01, C10, C11).
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So, if M_t[i][j] has value v, it contributes to:
    # If t=0 (M1): C00[i][j] += v, C11[i][j] += v
    # If t=1 (M2): C10[i][j] += v, C11[i][j] -= v
    # If t=2 (M3): C01[i][j] += v, C11[i][j] += v
    # If t=3 (M4): C00[i][j] += v, C10[i][j] += v
    # If t=4 (M5): C00[i][j] -= v, C01[i][j] += v
    # If t=5 (M6): C11[i][j] += v
    # If t=6 (M7): C00[i][j] += v
    
    # We can encode this as a list of (block_row, block_col, coeff) for each t.
    c_block_contrib = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1), (1,1,-1)],      # M2
        [(0,1,1), (1,1,1)],       # M3
        [(0,0,1), (1,0,1)],       # M4
        [(0,0,-1), (0,1,1)],      # M5
        [(1,1,1)],                # M6
        [(0,0,1)],                # M7
    ]
    
    # Inner contribution to M_t[i][j]
    # m1: (0,0,1), (1,1,1)
    # m2: (1,0,1), (1,1,-1)
    # m3: (0,1,1), (1,1,1)
    # m4: (0,0,1), (1,0,1)
    # m5: (0,0,-1), (0,1,1)
    # m6: (1,1,1)
    # m7: (0,0,1)
    
    inner_contrib = [
        [(0,0,1), (1,1,1)],       # m1
        [(1,0,1), (1,1,-1)],      # m2
        [(0,1,1), (1,1,1)],       # m3
        [(0,0,1), (1,0,1)],       # m4
        [(0,0,-1), (0,1,1)],      # m5
        [(1,1,1)],                # m6
        [(0,0,1)],                # m7
    ]

    for t in range(7):
        X, Y = outer_steps[t]
        for k in range(7):
            # Get the linear combos for this inner product
            X_part = inner_defs[k][0](X)
            Y_part = inner_defs[k][1](Y)
            
            # Fill U and V
            for idx, v in X_part:
                U[idx, col] += v
            for idx, v in Y_part:
                V[idx, col] += v
            
            # Fill W
            # This inner product contributes to M_t[i][j] with coeffs from inner_contrib[k]
            # And M_t contributes to C blocks with coeffs from c_block_contrib[t]
            for (li, lj, z_coeff) in inner_contrib[k]:
                for (br, bc, c_coeff) in c_block_contrib[t]:
                    # Global i = br*2 + li
                    # Global k = bc*2 + lj
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    w_idx_val = get_C_idx(br, bc, li, lj)
                    W[w_idx_val, col] += z_coeff * c_coeff
            
            col += 1
            
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
