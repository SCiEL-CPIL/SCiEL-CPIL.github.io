import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for (i, j), val in u_coeffs.items():
            U[i * m + j][col] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k][col] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i][col] += val
        col += 1

    # Define the 7 terms for 2x2 multiplication (Strassen)
    # Each term is a dict of coefficients for A, B, C
    # Indices for 2x2: 0,1
    # A: (i,j), B: (j,k), C: (k,i)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # Inner 2x2 terms:
    # 1. (A00+A11)(B00+B11) -> C00, C11
    # Wait, the 7 terms of Strassen are the scalar multiplications.
    # We need the decomposition of the 2x2 product into 7 rank-1 tensors.
    
    # Let's use the explicit 7 terms derived earlier.
    # Term 1: U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
    # Term 2: U: A10=1, A11=1; V: B00=1; W: C10=1, C11=-1
    # Term 3: U: A00=1; V: B01=1, B11=-1; W: C01=1, C11=1
    # Term 4: U: A11=1; V: B10=1, B00=-1; W: C00=1, C10=1
    # Term 5: U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=1
    # Term 6: U: A10=1, A00=-1; V: B00=1, B01=1; W: C11=1
    # Term 7: U: A01=1, A11=-1; V: B10=1, B11=1; W: C00=1
    
    inner_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Strassen Steps for 4x4
    # Each step defines X (2x2) and Y (2x2) in terms of A-blocks and B-blocks
    # A-blocks: A00, A01, A10, A11
    # B-blocks: B00, B01, B10, B11
    
    # We need to map the local 2x2 indices to global 4x4 indices.
    # Local index (li, lj) in a block corresponds to global (2*I+li, 2*J+lj).
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    # X = A00 + A11, Y = B00 + B11
    # X is a 2x2 matrix. X[0,0] = A00[0,0] + A11[0,0], etc.
    # Actually, X and Y are 2x2 matrices of linear forms.
    # We can represent the linear form for each element of X and Y.
    
    # Let's define the linear forms for X and Y for each step.
    # x_coeffs[li][lj] is a dict of global (i,j) -> coeff
    # y_coeffs[li][lj] is a dict of global (j,k) -> coeff
    
    # Step 1:
    # X = A00 + A11
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    # Y = B00 + B11
    # Y[0,0] = B[0,0] + B[2,2]
    # Y[0,1] = B[0,1] + B[2,3]
    # Y[1,0] = B[1,0] + B[3,2]
    # Y[1,1] = B[1,1] + B[3,3]
    
    # For each inner term, we combine these.
    
    # To simplify, we can just iterate through the 7 inner terms and for each,
    # construct the global U, V, W.
    
    # Inner term r has local U_l, V_l, W_l.
    # U_l is a 2x2 matrix of coeffs for the local A.
    # V_l is a 2x2 matrix of coeffs for the local B.
    # W_l is a 2x2 matrix of coeffs for the local C.
    
    # The local A is X. X[li, lj] is a linear form in global A.
    # So global U[i,j] += sum(li,lj) U_l[li,lj] * coeff_of_A[i,j] in X[li,lj].
    
    # Let's precompute the linear forms for X and Y for each step.
    
    steps = []
    
    # Helper to create linear form for a block
    def block_form(block_i, block_j, sign=1):
        # Returns dict of global (i,j) -> coeff
        d = {}
        for li in range(2):
            for lj in range(2):
                gi = 2 * block_i + li
                gj = 2 * block_j + lj
                d[(gi, gj)] = sign
        return d
    
    # Step 1: X = A00 + A11, Y = B00 + B11
    X1 = [[block_form(0,0) , block_form(0,0)], [block_form(0,0), block_form(0,0)]] # Placeholder
    # Actually, X is a 2x2 matrix where each element is a sum of block forms.
    # X[0,0] = A00[0,0] + A11[0,0]
    # Let's build X and Y as 2x2 lists of dicts.
    
    def add_blocks(b1, b2, sign1=1, sign2=1):
        # b1, b2 are tuples (bi, bj)
        d = {}
        for li in range(2):
            for lj in range(2):
                gi = 2 * b1[0] + li
                gj = 2 * b1[1] + lj
                d[(gi, gj)] = d.get((gi, gj), 0) + sign1
                gi = 2 * b2[0] + li
                gj = 2 * b2[1] + lj
                d[(gi, gj)] = d.get((gi, gj), 0) + sign2
        return d

    # Step 1
    X_s1 = [[add_blocks((0,0), (1,1)), add_blocks((0,0), (1,1))],
            [add_blocks((0,0), (1,1)), add_blocks((0,0), (1,1))]]
    # Wait, X is a 2x2 matrix.
    # X[0,0] = A00[0,0] + A11[0,0]
    # X[0,1] = A00[0,1] + A11[0,1]
    # So X[li, lj] = add_blocks((0,0), (1,1)) for all li,lj?
    # Yes, because the block addition is element-wise.
    
    Y_s1 = [[add_blocks((0,0), (1,1)), add_blocks((0,0), (1,1))],
            [add_blocks((0,0), (1,1)), add_blocks((0,0), (1,1))]]
    steps.append((X_s1, Y_s1))
    
    # Step 2: M2 = (A10+A11)B00
    # X = A10 + A11
    # X[0,0] = A10[0,0] + A11[0,0]
    X_s2 = [[add_blocks((1,0), (1,1)), add_blocks((1,0), (1,1))],
            [add_blocks((1,0), (1,1)), add_blocks((1,0), (1,1))]]
    # Y = B00
    Y_s2 = [[block_form(0,0), block_form(0,0)],
            [block_form(0,0), block_form(0,0)]]
    steps.append((X_s2, Y_s2))
    
    # Step 3: M3 = A00(B01-B11)
    X_s3 = [[block_form(0,0), block_form(0,0)],
            [block_form(0,0), block_form(0,0)]]
    Y_s3 = [[add_blocks((0,1), (1,1), 1, -1), add_blocks((0,1), (1,1), 1, -1)],
            [add_blocks((0,1), (1,1), 1, -1), add_blocks((0,1), (1,1), 1, -1)]]
    steps.append((X_s3, Y_s3))
    
    # Step 4: M4 = A11(B10-B00)
    X_s4 = [[block_form(1,1), block_form(1,1)],
            [block_form(1,1), block_form(1,1)]]
    Y_s4 = [[add_blocks((1,0), (0,0), 1, -1), add_blocks((1,0), (0,0), 1, -1)],
            [add_blocks((1,0), (0,0), 1, -1), add_blocks((1,0), (0,0), 1, -1)]]
    steps.append((X_s4, Y_s4))
    
    # Step 5: M5 = (A00+A01)B11
    X_s5 = [[add_blocks((0,0), (0,1)), add_blocks((0,0), (0,1))],
            [add_blocks((0,0), (0,1)), add_blocks((0,0), (0,1))]]
    Y_s5 = [[block_form(1,1), block_form(1,1)],
            [block_form(1,1), block_form(1,1)]]
    steps.append((X_s5, Y_s5))
    
    # Step 6: M6 = (A10-A00)(B00+B01)
    X_s6 = [[add_blocks((1,0), (0,0), 1, -1), add_blocks((1,0), (0,0), 1, -1)],
            [add_blocks((1,0), (0,0), 1, -1), add_blocks((1,0), (0,0), 1, -1)]]
    Y_s6 = [[add_blocks((0,0), (0,1)), add_blocks((0,0), (0,1))],
            [add_blocks((0,0), (0,1)), add_blocks((0,0), (0,1))]]
    steps.append((X_s6, Y_s6))
    
    # Step 7: M7 = (A01-A11)(B10+B11)
    X_s7 = [[add_blocks((0,1), (1,1), 1, -1), add_blocks((0,1), (1,1), 1, -1)],
            [add_blocks((0,1), (1,1), 1, -1), add_blocks((0,1), (1,1), 1, -1)]]
    Y_s7 = [[add_blocks((1,0), (1,1)), add_blocks((1,0), (1,1))],
            [add_blocks((1,0), (1,1)), add_blocks((1,0), (1,1))]]
    steps.append((X_s7, Y_s7))
    
    # Now, for each step and each inner term, add to U, V, W
    for s_idx, (X, Y) in enumerate(steps):
        for t_idx, (u_l, v_l, w_l) in enumerate(inner_terms):
            u_g = {}
            v_g = {}
            w_g = {}
            
            # Combine U
            for li in range(2):
                for lj in range(2):
                    coeff_l = u_l.get((li, lj), 0)
                    if coeff_l != 0:
                        # X[li, lj] is a dict of global (i,j) -> coeff
                        for (gi, gj), coeff_g in X[li][lj].items():
                            u_g[(gi, gj)] = u_g.get((gi, gj), 0) + coeff_l * coeff_g
            
            # Combine V
            for li in range(2):
                for lj in range(2):
                    coeff_l = v_l.get((li, lj), 0)
                    if coeff_l != 0:
                        for (gj, gk), coeff_g in Y[li][lj].items():
                            v_g[(gj, gk)] = v_g.get((gj, gk), 0) + coeff_l * coeff_g
            
            # Combine W
            # W_l is for C. C is 2x2.
            # The local C indices (ki, ki) map to global C indices.
            # C[ki, ki] corresponds to block C[ki, ki].
            # Global C index is (2*ki + li, 2*ki + lj)?
            # No, C is 4x4.
            # Local C[li, lj] corresponds to C[2*li + ..., 2*lj + ...]?
            # Wait, the output of the 2x2 multiplication is a 2x2 matrix Z.
            # Z[li, lj] is the element at local position (li, lj).
            # This maps to global C[2*li + ..., 2*lj + ...]?
            # No, the blocks of C are 2x2.
            # C[0,0] block is C[0:2, 0:2].
            # So local C[li, lj] is the element C[li, lj] within the block?
            # No, the Strassen algorithm computes the blocks of C.
            # So Z is the matrix of blocks?
            # No, Z is a 2x2 matrix of *values*?
            # No, X and Y are 2x2 matrices of *linear forms*.
            # The product Z is a 2x2 matrix of linear forms.
            # Z[li, lj] = sum_ab X[li, ab] Y[ab, lj].
            # This Z[li, lj] is the linear form for the block C[li, lj].
            # So the global C indices for Z[li, lj] are (2*li + u, 2*lj + v) for u,v in 0,1.
            
            # Wait, the inner term w_l is a 2x2 matrix of coefficients for the *local* C.
            # So w_l[ki, ki] is the coefficient for the local C element (ki, ki).
            # This local C element corresponds to the block C[ki, ki].
            # So we need to map the local C indices to global C indices.
            
            for ki in range(2):
                for kj in range(2):
                    coeff_l = w_l.get((ki, kj), 0)
                    if coeff_l != 0:
                        # The local C[ki, kj] corresponds to the block C[ki, kj].
                        # The elements of this block are C[2*ki + u, 2*kj + v].
                        # But w_l is a coefficient for the *entire* block?
                        # No, w_l is a 2x2 matrix.
                        # In the 2x2 case, C is 2x2.
                        # Here, the "2x2" multiplication is of blocks.
                        # So the result is a 2x2 matrix of blocks.
                        # Each block is 2x2.
                        # So the local index (ki, kj) refers to the block position.
                        # The coefficient coeff_l applies to *all* elements in that block?
                        # Yes, because the linear form for the block is uniform?
                        # No, the linear form for the block C[ki, kj] is a sum of C elements.
                        # But in the Strassen reconstruction, we add the *blocks*.
                        # So if the local term says "add 1 to C[0,0]", it means add 1 to the *block* C[0,0].
                        # So all 4 elements in that block get coefficient 1.
                        
                        for u in range(2):
                            for v in range(2):
                                gi = 2 * ki + u
                                gj = 2 * kj + v
                                w_g[(gj, gi)] = w_g.get((gj, gi), 0) + coeff_l
            
            add_term(u_g, v_g, w_g)
            
    # Add M_4,4,1 part
    for i in range(4):
        for j in range(4):
            u_g = {(i, j): 1}
            v_g = {(j, 4): 1}
            w_g = {(4, i): 1}
            add_term(u_g, v_g, w_g)
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
