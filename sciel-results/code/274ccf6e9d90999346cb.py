import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook for the rest (4x4 * 4x3)
    # Rank 48
    # Strassen for the first part (4x4 * 4x4)
    # Rank 49
    # Total Rank 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to U, V, W
    # u_indices: list of (index, coeff)
    # v_indices: list of (index, coeff)
    # w_indices: list of (index, coeff)
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        if col >= rank:
            return
        for idx, c in u_indices:
            U[idx][col] += c
        for idx, c in v_indices:
            V[idx][col] += c
        for idx, c in w_indices:
            W[idx][col] += c
        col += 1
    
    # Strassen 4x4x4 part
    # A indices: i*4 + j
    # B indices: j*4 + k (for k in 0..3)
    # C indices: k*4 + i
    
    # Define linear forms for A and B blocks
    # A00: i in {0,1}, j in {0,1} -> indices 0,1,4,5
    # A01: i in {0,1}, j in {2,3} -> indices 2,3,6,7
    # A10: i in {2,3}, j in {0,1} -> indices 8,9,12,13
    # A11: i in {2,3}, j in {2,3} -> indices 10,11,14,15
    
    # B00: j in {0,1}, k in {0,1} -> indices 0,1,4,5
    # B01: j in {0,1}, k in {2,3} -> indices 2,3,6,7
    # B10: j in {2,3}, k in {0,1} -> indices 8,9,12,13
    # B11: j in {2,3}, k in {2,3} -> indices 10,11,14,15
    
    # We define the 7 M_l products.
    # Each M_l = X_l * Y_l where X_l, Y_l are 2x2 matrices of linear forms.
    # The entries of X_l are linear combinations of A entries.
    # The entries of Y_l are linear combinations of B entries.
    
    # Let's define the linear forms as dictionaries {index: coeff}
    
    # A blocks
    A00 = {0:1, 1:1, 4:1, 5:1}
    A01 = {2:1, 3:1, 6:1, 7:1}
    A10 = {8:1, 9:1, 12:1, 13:1}
    A11 = {10:1, 11:1, 14:1, 15:1}
    
    # B blocks
    B00 = {0:1, 1:1, 4:1, 5:1}
    B01 = {2:1, 3:1, 6:1, 7:1}
    B10 = {8:1, 9:1, 12:1, 13:1}
    B11 = {10:1, 11:1, 14:1, 15:1}
    
    # Define X and Y for each M_l
    # X_l is a 2x2 matrix of dicts
    # Y_l is a 2x2 matrix of dicts
    
    # M1: X = A00+A11, Y = B00+B11
    # M2: X = A10+A11, Y = B00
    # M3: X = A00, Y = B01-B11
    # M4: X = A11, Y = B10-B00
    # M5: X = A00+A01, Y = B11
    # M6: X = A10-A00, Y = B00+B01
    # M7: X = A01-A11, Y = B10+B11
    
    # Function to add two dicts
    def add_dicts(d1, d2):
        d = dict(d1)
        for k, v in d2.items():
            d[k] = d.get(k, 0) + v
        return d
    
    # Function to subtract two dicts
    def sub_dicts(d1, d2):
        d = dict(d1)
        for k, v in d2.items():
            d[k] = d.get(k, 0) - v
        return d
    
    # Function to scale a dict
    def scale_dict(d, s):
        return {k: v*s for k, v in d.items()}
    
    # Define the 7 X_l and Y_l
    # Note: X_l and Y_l are 2x2 matrices.
    # X_l[0][0] corresponds to the (0,0) entry of X_l, which is a linear form.
    
    # M1
    X1 = [
        [add_dicts(A00, A11), add_dicts(A00, A11)], # Wait, X1 is A00+A11 which is a 2x2 matrix.
        # A00 is [[A00_00, A00_01], [A00_10, A00_11]]
        # A11 is [[A11_00, A11_01], [A11_10, A11_11]]
        # So X1[0][0] = A00[0][0] + A11[0][0]
    ]
    
    # Let's be precise with indices.
    # A00[0][0] is A[0,0] = index 0
    # A00[0][1] is A[0,1] = index 1
    # A00[1][0] is A[1,0] = index 4
    # A00[1][1] is A[1,1] = index 5
    
    # A11[0][0] is A[2,2] = index 10
    # A11[0][1] is A[2,3] = index 11
    # A11[1][0] is A[3,2] = index 14
    # A11[1][1] is A[3,3] = index 15
    
    # So X1 is:
    # [[ {0:1, 10:1}, {1:1, 11:1} ],
    #  [ {4:1, 14:1}, {5:1, 15:1} ]]
    
    # Y1 is:
    # [[ {0:1, 10:1}, {1:1, 11:1} ],
    #  [ {4:1, 14:1}, {5:1, 15:1} ]]
    # (Since B00+B11 has same index structure relative to B's blocks)
    # B00[0][0] is B[0,0] = index 0
    # B00[0][1] is B[0,1] = index 1
    # B00[1][0] is B[1,0] = index 4
    # B00[1][1] is B[1,1] = index 5
    # B11[0][0] is B[2,2] = index 10
    # B11[0][1] is B[2,3] = index 11
    # B11[1][0] is B[3,2] = index 14
    # B11[1][1] is B[3,3] = index 15
    
    # So Y1 is:
    # [[ {0:1, 10:1}, {1:1, 11:1} ],
    #  [ {4:1, 14:1}, {5:1, 15:1} ]]
    
    # We can define a generic function to create the X and Y matrices for each M_l
    # based on the block definitions.
    
    def create_block_matrix(block_dict):
        # block_dict maps (r,c) to {index: coeff}
        # We need to convert the flat block dicts like A00 to this format.
        # A00 indices: 0,1,4,5
        # (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
        pass
    
    # Let's just hardcode the 7 X and Y matrices as lists of lists of dicts.
    
    # M1
    X1 = [
        [{0:1, 10:1}, {1:1, 11:1}],
        [{4:1, 14:1}, {5:1, 15:1}]
    ]
    Y1 = [
        [{0:1, 10:1}, {1:1, 11:1}],
        [{4:1, 14:1}, {5:1, 15:1}]
    ]
    
    # M2: X = A10+A11, Y = B00
    # A10: 8,9,12,13. A11: 10,11,14,15.
    # X2 = [
    #   [{8:1, 10:1}, {9:1, 11:1}],
    #   [{12:1, 14:1}, {13:1, 15:1}]
    # ]
    # Y2 = B00
    # Y2 = [
    #   [{0:1}, {1:1}],
    #   [{4:1}, {5:1}]
    # ]
    X2 = [
        [{8:1, 10:1}, {9:1, 11:1}],
        [{12:1, 14:1}, {13:1, 15:1}]
    ]
    Y2 = [
        [{0:1}, {1:1}],
        [{4:1}, {5:1}]
    ]
    
    # M3: X = A00, Y = B01-B11
    # X3 = A00
    X3 = [
        [{0:1}, {1:1}],
        [{4:1}, {5:1}]
    ]
    # Y3 = B01 - B11
    # B01: 2,3,6,7. B11: 10,11,14,15.
    # B01[0][0]=2, B01[0][1]=3, B01[1][0]=6, B01[1][1]=7
    # B11[0][0]=10, B11[0][1]=11, B11[1][0]=14, B11[1][1]=15
    Y3 = [
        [{2:1, 10:-1}, {3:1, 11:-1}],
        [{6:1, 14:-1}, {7:1, 15:-1}]
    ]
    
    # M4: X = A11, Y = B10-B00
    # X4 = A11
    X4 = [
        [{10:1}, {11:1}],
        [{14:1}, {15:1}]
    ]
    # Y4 = B10 - B00
    # B10: 8,9,12,13. B00: 0,1,4,5.
    # B10[0][0]=8, B10[0][1]=9, B10[1][0]=12, B10[1][1]=13
    # B00[0][0]=0, B00[0][1]=1, B00[1][0]=4, B00[1][1]=5
    Y4 = [
        [{8:1, 0:-1}, {9:1, 1:-1}],
        [{12:1, 4:-1}, {13:1, 5:-1}]
    ]
    
    # M5: X = A00+A01, Y = B11
    # X5 = A00 + A01
    # A00: 0,1,4,5. A01: 2,3,6,7.
    X5 = [
        [{0:1, 2:1}, {1:1, 3:1}],
        [{4:1, 6:1}, {5:1, 7:1}]
    ]
    # Y5 = B11
    Y5 = [
        [{10:1}, {11:1}],
        [{14:1}, {15:1}]
    ]
    
    # M6: X = A10-A00, Y = B00+B01
    # X6 = A10 - A00
    # A10: 8,9,12,13. A00: 0,1,4,5.
    X6 = [
        [{8:1, 0:-1}, {9:1, 1:-1}],
        [{12:1, 4:-1}, {13:1, 5:-1}]
    ]
    # Y6 = B00 + B01
    # B00: 0,1,4,5. B01: 2,3,6,7.
    Y6 = [
        [{0:1, 2:1}, {1:1, 3:1}],
        [{4:1, 6:1}, {5:1, 7:1}]
    ]
    
    # M7: X = A01-A11, Y = B10+B11
    # X7 = A01 - A11
    # A01: 2,3,6,7. A11: 10,11,14,15.
    X7 = [
        [{2:1, 10:-1}, {3:1, 11:-1}],
        [{6:1, 14:-1}, {7:1, 15:-1}]
    ]
    # Y7 = B10 + B11
    # B10: 8,9,12,13. B11: 10,11,14,15.
    Y7 = [
        [{8:1, 10:1}, {9:1, 11:1}],
        [{12:1, 14:1}, {13:1, 15:1}]
    ]
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # Now generate the 49 terms for the 4x4x4 part
    for l in range(7):
        X = Xs[l]
        Y = Ys[l]
        
        # 7 Strassen products for M_l = X * Y
        # P1 = (X00+X11)(Y00+Y11)
        # P2 = (X10+X11)Y00
        # P3 = X00(Y01-Y11)
        # P4 = X11(Y10-Y00)
        # P5 = (X00+X01)Y11
        # P6 = (X10-X00)(Y00+Y01)
        # P7 = (X01-X11)(Y10+Y11)
        
        # Helper to add two 2x2 matrices of dicts
        def add_mat(d1, d2):
            return [
                [add_dicts(d1[0][0], d2[0][0]), add_dicts(d1[0][1], d2[0][1])],
                [add_dicts(d1[1][0], d2[1][0]), add_dicts(d1[1][1], d2[1][1])]
            ]
        
        def sub_mat(d1, d2):
            return [
                [sub_dicts(d1[0][0], d2[0][0]), sub_dicts(d1[0][1], d2[0][1])],
                [sub_dicts(d1[1][0], d2[1][0]), sub_dicts(d1[1][1], d2[1][1])]
            ]
        
        # P1
        X_p1 = add_mat(X, X) # Wait, X00+X11 is not X+X.
        # X00 is X[0][0], X11 is X[1][1].
        # The matrix for P1 factor is [[X00+X11, X00+X11], [X00+X11, X00+X11]]? No.
        # The Strassen algorithm computes M = X * Y.
        # P1 = (X00 + X11) * (Y00 + Y11).
        # Here X00, X11 are scalars (or linear forms).
        # So the factor for X in P1 is the scalar (X00+X11) multiplied by the identity matrix?
        # No, the "multiplication" is of the linear forms.
        # The term in the decomposition is u * v * w.
        # u corresponds to the linear form in A.
        # v corresponds to the linear form in B.
        # w corresponds to the linear form in C.
        
        # For P1 = (X00+X11)(Y00+Y11):
        # u is the linear form (X00+X11). This is a sum of entries of X.
        # But X entries are linear forms in A.
        # So u is a linear form in A.
        # v is a linear form in B.
        # w is a linear form in C.
        
        # Wait, the Strassen decomposition for 2x2 matrices Z = X Y is:
        # Z00 = P1 + P4 - P5 + P7
        # Z01 = P3 + P5
        # Z10 = P2 + P4
        # Z11 = P1 - P2 + P3 + P6
        
        # Each Pk is a product of two linear forms.
        # So we have 7 terms for the 2x2 multiplication.
        # Each term k has:
        # u_k: linear form in A (from X)
        # v_k: linear form in B (from Y)
        # w_k: linear form in C (from Z)
        
        # The linear form in C for Pk is determined by which entry of Z it contributes to and with what coefficient.
        # P1 contributes to Z00 (+1) and Z11 (+1).
        # So w_1 has coeff 1 at Z00 index and 1 at Z11 index.
        
        # Let's define the C indices for the 4x4x4 part.
        # C is 4x4.
        # C00: k in {0,1}, i in {0,1} -> indices 0,1,4,5
        # C01: k in {2,3}, i in {0,1} -> indices 2,3,6,7
        # C10: k in {0,1}, i in {2,3} -> indices 8,9,12,13
        # C11: k in {2,3}, i in {2,3} -> indices 10,11,14,15
        
        # For M_l, the result is a 2x2 matrix of linear forms in C.
        # Let's call the entries of M_l as M_l[0][0], M_l[0][1], M_l[1][0], M_l[1][1].
        # These correspond to the blocks of C.
        # M_l[0][0] contributes to C00.
        # M_l[0][1] contributes to C01.
        # M_l[1][0] contributes to C10.
        # M_l[1][1] contributes to C11.
        
        # So for each Pk, we need to determine its contribution to C00, C01, C10, C11.
        
        # P1: Z00 (+1), Z11 (+1)
        # P2: Z10 (+1)
        # P3: Z01 (+1)
        # P4: Z00 (+1), Z10 (+1)
        # P5: Z00 (-1), Z01 (+1)
        # P6: Z11 (+1)
        # P7: Z00 (+1), Z11 (+1)
        
        # Let's map these to C block indices.
        # C00 indices: 0,1,4,5
        # C01 indices: 2,3,6,7
        # C10 indices: 8,9,12,13
        # C11 indices: 10,11,14,15
        
        # The "scalar" Pk is actually a product of linear forms.
        # So u_k is a linear form in A.
        # v_k is a linear form in B.
        # w_k is a linear form in C.
        
        # For P1:
        # u_1 = X00 + X11 (linear form in A)
        # v_1 = Y00 + Y11 (linear form in B)
        # w_1 = C00 + C11 (linear form in C)
        
        # How to represent C00 as a linear form?
        # C00 is a 2x2 matrix of entries.
        # The term P1 is a scalar (product of linear forms).
        # It contributes to all entries of C00 and C11 with the same coefficient?
        # Yes, because M_l[0][0] is a scalar multiple of the identity? No.
        # M_l[0][0] is a linear form.
        # Wait, M_l = X * Y.
        # X and Y are 2x2 matrices of linear forms.
        # So M_l is a 2x2 matrix of linear forms.
        # Each entry M_l[r][c] is a linear form.
        # The Strassen algorithm computes these linear forms using 7 multiplications.
        # So each multiplication Pk produces a scalar (product of two linear forms).
        # This scalar is then added to the linear forms of the entries of M_l.
        
        # So for P1, u_1 and v_1 are linear forms.
        # The result of P1 is a scalar.
        # This scalar is added to M_l[0][0] and M_l[1][1].
        # So w_1 should have coefficients corresponding to the entries of C00 and C11.
        # Specifically, w_1[c_idx] = coeff for C00 entry + coeff for C11 entry.
        # Since P1 is a scalar, it contributes equally to all 4 entries of C00 and all 4 entries of C11?
        # No, that's not right.
        
        # Let's re-evaluate.
        # The tensor decomposition term is u \otimes v \otimes w.
        # u is a vector of size 16 (A).
        # v is a vector of size 28 (B).
        # w is a vector of size 28 (C).
        
        # The product u[i] v[j] w[k] contributes to T[i,j,k].
        # T[i,j,k] = 1 if i,j,k are valid.
        
        # In the Strassen algorithm, we compute C = AB.
        # C[i,k] = sum_j A[i,j] B[j,k].
        
        # The term P1 = (X00+X11)(Y00+Y11).
        # X00 is a linear form in A. Let's say X00 = sum_a a_idx * A[a_idx].
        # Y00 is a linear form in B. Let's say Y00 = sum_b b_idx * B[b_idx].
        # So P1 = (sum_a a_idx A[a_idx]) * (sum_b b_idx B[b_idx]).
        # This is a bilinear form.
        # It contributes to C.
        # Specifically, P1 contributes to C00 and C11.
        # How does it contribute to the entries of C00?
        # C00 is a 2x2 matrix.
        # The entry C00[r,c] is a linear form in C.
        # In the decomposition, w is a vector.
        # w[k_idx] is the coefficient for C[k_idx].
        
        # The contribution of P1 to C00[r,c] is P1 * 1.
        # So w[k_idx] should be 1 for all k_idx in C00 and C11?
        # No, that would mean P1 contributes 1 to every entry of C00 and C11.
        # Is that correct?
        
        # Let's check the 2x2 case.
        # Z00 = P1 + P4 - P5 + P7.
        # Z00 is a single entry.
        # So P1 contributes 1 to Z00.
        # In the 4x4 case, M_l[0][0] corresponds to the block C00.
        # But M_l[0][0] is a linear form.
        # Wait, M_l is a 2x2 matrix of linear forms.
        # So M_l[0][0] is a linear form.
        # The Strassen algorithm computes the 4 linear forms M_l[0][0], M_l[0][1], M_l[1][0], M_l[1][1].
        # Each of these is a sum of the 7 products Pk.
        
        # So for P1:
        # It contributes to M_l[0][0] with coeff 1.
        # It contributes to M_l[1][1] with coeff 1.
        
        # Now, how does M_l[0][0] relate to C?
        # M_l[0][0] is one of the linear forms that sum up to the entries of C00.
        # Specifically, C00 is a 2x2 matrix.
        # Let's say C00 = M1[0][0] + M4[0][0] - M5[0][0] + M7[0][0].
        # So each entry of C00 is the sum of the corresponding entries of the M_l[0][0] blocks.
        
        # This implies that the linear form M_l[0][0] is actually a 2x2 matrix of linear forms?
        # No, M_l is a 2x2 matrix of linear forms.
        # So M_l[0][0] is a scalar linear form.
        # But C00 is a 2x2 matrix.
        # So C00[0][0] = M1[0][0] + ...
        # C00[0][1] = M1[0][0] + ...?
        # No, that's not how Strassen works.
        
        # Strassen works on 2x2 matrices.
        # If we apply Strassen to 4x4 matrices, we split them into 2x2 blocks.
        # The blocks are 2x2 matrices.
        # So A00 is a 2x2 matrix.
        # The product M1 = (A00+A11)(B00+B11) is a 2x2 matrix product.
        # So M1 is a 2x2 matrix.
        # The entries of M1 are linear forms.
        # So M1[0][0] is a linear form.
        # M1[0][1] is a linear form.
        # etc.
        
        # Then C00 = M1 + M4 - M5 + M7.
        # So C00[0][0] = M1[0][0] + M4[0][0] - M5[0][0] + M7[0][0].
        # C00[0][1] = M1[0][1] + M4[0][1] - M5[0][1] + M7[0][1].
        
        # So each entry of C00 is a sum of the corresponding entries of the M_l matrices.
        
        # Now, each M_l is computed using 7 Strassen products.
        # So M_l[0][0] = P_l,1 + P_l,4 - P_l,5 + P_l,7.
        # M_l[0][1] = P_l,3 + P_l,5.
        # M_l[1][0] = P_l,2 + P_l,4.
        # M_l[1][1] = P_l,1 - P_l,2 + P_l,3 + P_l,6.
        
        # So P_l,1 contributes to M_l[0][0] and M_l[1][1].
        # And M_l[0][0] contributes to C00[0][0] and C00[0][1]? No.
        # M_l[0][0] is a scalar. It contributes to C00[0][0] with coeff 1.
        # Does it contribute to C00[0][1]?
        # C00[0][1] = M1[0][1] + ...
        # So P_l,1 contributes to M1[0][0] and M1[1][1].
        # M1[0][0] contributes to C00[0][0].
        # M1[1][1] contributes to C00[1][1].
        
        # So P_l,1 contributes to C00[0][0] and C00[1][1].
        # And also to C11[0][0] and C11[1][1]?
        # M1[1][1] contributes to C11[1][1].
        
        # So for P_l,1:
        # u_l,1 = X_l[0][0] + X_l[1][1]
        # v_l,1 = Y_l[0][0] + Y_l[1][1]
        # w_l,1 has coeff 1 at C00[0][0], C00[1][1], C11[0][0], C11[1][1].
        
        # C00[0][0] is C[0,0] = index 0.
        # C00[1][1] is C[1,1] = index 5.
        # C11[0][0] is C[2,2] = index 10.
        # C11[1][1] is C[3,3] = index 15.
        
        # So w_l,1 = {0:1, 5:1, 10:1, 15:1}.
        
        # This seems correct.
        
        # Let's define the w vectors for each Pk.
        
        # C00 indices: 0,1,4,5
        # C01 indices: 2,3,6,7
        # C10 indices: 8,9,12,13
        # C11 indices: 10,11,14,15
        
        # P1: Z00 (+1), Z11 (+1)
        # Z00 is M_l[0][0]. Contributes to C00[0][0] and C00[0][1]? No.
        # M_l[0][0] is a scalar.
        # C00[0][0] = M1[0][0] + ...
        # C00[0][1] = M1[0][1] + ...
        # So M_l[0][0] only contributes to C00[0][0].
        # M_l[0][1] only contributes to C00[0][1].
        # M_l[1][0] only contributes to C00[1][0].
        # M_l[1][1] only contributes to C00[1][1].
        
        # So P1 contributes to M_l[0][0] and M_l[1][1].
        # So P1 contributes to C00[0][0] and C00[1][1].
        # And to C11[0][0] and C11[1][1].
        
        # C00[0][0] -> C[0,0] -> 0
        # C00[1][1] -> C[1,1] -> 5
        # C11[0][0] -> C[2,2] -> 10
        # C11[1][1] -> C[3,3] -> 15
        
        w1 = {0:1, 5:1, 10:1, 15:1}
        
        # P2: Z10 (+1)
        # M_l[1][0] -> C00[1][0] -> C[1,0] -> 4
        # And C11[1][0] -> C[3,2] -> 14
        w2 = {4:1, 14:1}
        
        # P3: Z01 (+1)
        # M_l[0][1] -> C00[0][1] -> C[0,1] -> 1
        # And C11[0][1] -> C[2,3] -> 11
        w3 = {1:1, 11:1}
        
        # P4: Z00 (+1), Z10 (+1)
        # M_l[0][0] -> C[0,0] -> 0
        # M_l[1][0] -> C[1,0] -> 4
        # M_l[0][0] -> C[2,2] -> 10
        # M_l[1][0] -> C[3,2] -> 14
        w4 = {0:1, 4:1, 10:1, 14:1}
        
        # P5: Z00 (-1), Z01 (+1)
        # M_l[0][0] -> C[0,0] -> 0
        # M_l[0][1] -> C[0,1] -> 1
        # M_l[0][0] -> C[2,2] -> 10
        # M_l[0][1] -> C[2,3] -> 11
        w5 = {0:-1, 1:1, 10:-1, 11:1}
        
        # P6: Z11 (+1)
        # M_l[1][1] -> C[1,1] -> 5
        # M_l[1][1] -> C[3,3] -> 15
        w6 = {5:1, 15:1}
        
        # P7: Z00 (+1), Z11 (+1)
        # Same as P1
        w7 = {0:1, 5:1, 10:1, 15:1}
        
        ws = [w1, w2, w3, w4, w5, w6, w7]
        
        # Now generate the 49 terms
        for l in range(7):
            X = Xs[l]
            Y = Ys[l]
            
            # P1
            u1 = add_dicts(X[0][0], X[1][1])
            v1 = add_dicts(Y[0][0], Y[1][1])
            add_term(u1, v1, ws[0])
            
            # P2
            u2 = add_dicts(X[1][0], X[1][1])
            v2 = Y[0][0]
            add_term(u2, v2, ws[1])
            
            # P3
            u3 = X[0][0]
            v3 = sub_dicts(Y[0][1], Y[1][1])
            add_term(u3, v3, ws[2])
            
            # P4
            u4 = X[1][1]
            v4 = sub_dicts(Y[1][0], Y[0][0])
            add_term(u4, v4, ws[3])
            
            # P5
            u5 = add_dicts(X[0][0], X[0][1])
            v5 = Y[1][1]
            add_term(u5, v5, ws[4])
            
            # P6
            u6 = sub_dicts(X[1][0], X[0][0])
            v6 = add_dicts(Y[0][0], Y[0][1])
            add_term(u6, v6, ws[5])
            
            # P7
            u7 = sub_dicts(X[0][1], X[1][1])
            v7 = add_dicts(Y[1][0], Y[1][1])
            add_term(u7, v7, ws[6])
            
    # Schoolbook for the rest (4x4 * 4x3)
    # A is 4x4, B_rest is 4x3, C_rest is 4x3
    # B_rest indices: j in 0..3, k in 4..6
    # C_rest indices: k in 4..6, i in 0..3
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    if time.time() > deadline - 5:
        pass
        
    return {"U": U, "V": V, "W": W}
