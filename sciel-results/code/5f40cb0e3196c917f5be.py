import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # We will build U, V, W as lists of lists or numpy arrays
    # Rank 65
    rank = 65
    U = [[0.0] * rank for _ in range(n * m)] # 16 x 65
    V = [[0.0] * rank for _ in range(m * p)] # 20 x 65
    W = [[0.0] * rank for _ in range(p * n)] # 20 x 65
    
    # Part 1: A (4x4) * B_left (4x4) using Strassen (Rank 49)
    # B_left corresponds to B[:, 0:4]
    
    # Helper to add a term to U, V, W
    # term_u: dict { (i, j): coeff } for A
    # term_v: dict { (j, k): coeff } for B
    # term_w: dict { (k, i): coeff } for C
    # col: column index in U, V, W
    
    def add_term(term_u, term_v, term_w, col):
        for (i, j), coeff in term_u.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in term_v.items():
            V[j * p + k][col] += coeff
        for (k, i), coeff in term_w.items():
            W[k * n + i][col] += coeff

    # Strassen 2x2 decomposition
    # Inputs: P (2x2), Q (2x2)
    # P = [[P00, P01], [P10, P11]]
    # Q = [[Q00, Q01], [Q10, Q11]]
    # 7 products:
    # M1 = (P00+P11)(Q00+Q11)
    # M2 = (P10+P11)Q00
    # M3 = P00(Q01-Q11)
    # M4 = P11(Q10-Q00)
    # M5 = (P00+P01)Q11
    # M6 = (P10-P00)(Q00+Q01)
    # M7 = (P01-P11)(Q10+Q11)
    
    # Result C (2x2):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express P and Q entries in terms of A and B entries.
    # A is 4x4. Blocks A11, A12, A21, A22 are 2x2.
    # A11: A[0,0], A[0,1], A[1,0], A[1,1]
    # A12: A[0,2], A[0,3], A[1,2], A[1,3]
    # A21: A[2,0], A[2,1], A[3,0], A[3,1]
    # A22: A[2,2], A[2,3], A[3,2], A[3,3]
    
    # B_left is 4x4. Blocks B11, B12, B21, B22 are 2x2.
    # B11: B[0,0], B[0,1], B[1,0], B[1,1]
    # B12: B[0,2], B[0,3], B[1,2], B[1,3]
    # B21: B[2,0], B[2,1], B[3,0], B[3,1]
    # B22: B[2,2], B[2,3], B[3,2], B[3,3]
    
    # Define linear forms for the 7 outer Strassen products M_r
    # Each M_r is P_r * Q_r
    # P_r is a 2x2 matrix of linear forms of A
    # Q_r is a 2x2 matrix of linear forms of B
    
    # We will represent a linear form as a dict { (index): coeff }
    
    def block_form(block_name, local_idx, source_matrix):
        # source_matrix: 'A' or 'B'
        # block_name: 11, 12, 21, 22
        # local_idx: (r, c) in 2x2 block
        # Returns dict of (global_i, global_j) -> coeff
        form = {}
        # Map block to global offset
        if source_matrix == 'A':
            if block_name == '11':
                gi, gj = 0, 0
            elif block_name == '12':
                gi, gj = 0, 2
            elif block_name == '21':
                gi, gj = 2, 0
            else: # 22
                gi, gj = 2, 2
        else: # B
            if block_name == '11':
                gi, gj = 0, 0
            elif block_name == '12':
                gi, gj = 0, 2
            elif block_name == '21':
                gi, gj = 2, 0
            else: # 22
                gi, gj = 2, 2
        
        form[(gi + local_idx[0], gj + local_idx[1])] = 1
        return form

    def add_forms(f1, f2, sign=1):
        res = dict(f1)
        for key, val in f2.items():
            res[key] = res.get(key, 0) + sign * val
        return res

    def sub_forms(f1, f2):
        return add_forms(f1, f2, -1)

    # Define the 7 outer products
    # P_r and Q_r are 2x2 matrices of forms
    # P_r = [[P00, P01], [P10, P11]]
    
    outer_products = []
    
    # M1: (A11+A22)(B11+B22)
    P1 = [
        [add_forms(block_form('11', (0,0), 'A'), block_form('22', (0,0), 'A')), 
         add_forms(block_form('11', (0,1), 'A'), block_form('22', (0,1), 'A'))],
        [add_forms(block_form('11', (1,0), 'A'), block_form('22', (1,0), 'A')), 
         add_forms(block_form('11', (1,1), 'A'), block_form('22', (1,1), 'A'))]
    ]
    Q1 = [
        [add_forms(block_form('11', (0,0), 'B'), block_form('22', (0,0), 'B')), 
         add_forms(block_form('11', (0,1), 'B'), block_form('22', (0,1), 'B'))],
        [add_forms(block_form('11', (1,0), 'B'), block_form('22', (1,0), 'B')), 
         add_forms(block_form('11', (1,1), 'B'), block_form('22', (1,1), 'B'))]
    ]
    outer_products.append((P1, Q1))
    
    # M2: (A21+A22)B11
    P2 = [
        [add_forms(block_form('21', (0,0), 'A'), block_form('22', (0,0), 'A')), 
         add_forms(block_form('21', (0,1), 'A'), block_form('22', (0,1), 'A'))],
        [add_forms(block_form('21', (1,0), 'A'), block_form('22', (1,0), 'A')), 
         add_forms(block_form('21', (1,1), 'A'), block_form('22', (1,1), 'A'))]
    ]
    Q2 = [
        [block_form('11', (0,0), 'B'), block_form('11', (0,1), 'B')],
        [block_form('11', (1,0), 'B'), block_form('11', (1,1), 'B')]
    ]
    outer_products.append((P2, Q2))
    
    # M3: A11(B12-B22)
    P3 = [
        [block_form('11', (0,0), 'A'), block_form('11', (0,1), 'A')],
        [block_form('11', (1,0), 'A'), block_form('11', (1,1), 'A')]
    ]
    Q3 = [
        [sub_forms(block_form('12', (0,0), 'B'), block_form('22', (0,0), 'B')), 
         sub_forms(block_form('12', (0,1), 'B'), block_form('22', (0,1), 'B'))],
        [sub_forms(block_form('12', (1,0), 'B'), block_form('22', (1,0), 'B')), 
         sub_forms(block_form('12', (1,1), 'B'), block_form('22', (1,1), 'B'))]
    ]
    outer_products.append((P3, Q3))
    
    # M4: A22(B21-B11)
    P4 = [
        [block_form('22', (0,0), 'A'), block_form('22', (0,1), 'A')],
        [block_form('22', (1,0), 'A'), block_form('22', (1,1), 'A')]
    ]
    Q4 = [
        [sub_forms(block_form('21', (0,0), 'B'), block_form('11', (0,0), 'B')), 
         sub_forms(block_form('21', (0,1), 'B'), block_form('11', (0,1), 'B'))],
        [sub_forms(block_form('21', (1,0), 'B'), block_form('11', (1,0), 'B')), 
         sub_forms(block_form('21', (1,1), 'B'), block_form('11', (1,1), 'B'))]
    ]
    outer_products.append((P4, Q4))
    
    # M5: (A11+A12)B22
    P5 = [
        [add_forms(block_form('11', (0,0), 'A'), block_form('12', (0,0), 'A')), 
         add_forms(block_form('11', (0,1), 'A'), block_form('12', (0,1), 'A'))],
        [add_forms(block_form('11', (1,0), 'A'), block_form('12', (1,0), 'A')), 
         add_forms(block_form('11', (1,1), 'A'), block_form('12', (1,1), 'A'))]
    ]
    Q5 = [
        [block_form('22', (0,0), 'B'), block_form('22', (0,1), 'B')],
        [block_form('22', (1,0), 'B'), block_form('22', (1,1), 'B')]
    ]
    outer_products.append((P5, Q5))
    
    # M6: (A21-A11)(B11+B12)
    P6 = [
        [sub_forms(block_form('21', (0,0), 'A'), block_form('11', (0,0), 'A')), 
         sub_forms(block_form('21', (0,1), 'A'), block_form('11', (0,1), 'A'))],
        [sub_forms(block_form('21', (1,0), 'A'), block_form('11', (1,0), 'A')), 
         sub_forms(block_form('21', (1,1), 'A'), block_form('11', (1,1), 'A'))]
    ]
    Q6 = [
        [add_forms(block_form('11', (0,0), 'B'), block_form('12', (0,0), 'B')), 
         add_forms(block_form('11', (0,1), 'B'), block_form('12', (0,1), 'B'))],
        [add_forms(block_form('11', (1,0), 'B'), block_form('12', (1,0), 'B')), 
         add_forms(block_form('11', (1,1), 'B'), block_form('12', (1,1), 'B'))]
    ]
    outer_products.append((P6, Q6))
    
    # M7: (A12-A22)(B21+B22)
    P7 = [
        [sub_forms(block_form('12', (0,0), 'A'), block_form('22', (0,0), 'A')), 
         sub_forms(block_form('12', (0,1), 'A'), block_form('22', (0,1), 'A'))],
        [sub_forms(block_form('12', (1,0), 'A'), block_form('22', (1,0), 'A')), 
         sub_forms(block_form('12', (1,1), 'A'), block_form('22', (1,1), 'A'))]
    ]
    Q7 = [
        [add_forms(block_form('21', (0,0), 'B'), block_form('22', (0,0), 'B')), 
         add_forms(block_form('21', (0,1), 'B'), block_form('22', (0,1), 'B'))],
        [add_forms(block_form('21', (1,0), 'B'), block_form('22', (1,0), 'B')), 
         add_forms(block_form('21', (1,1), 'B'), block_form('22', (1,1), 'B'))]
    ]
    outer_products.append((P7, Q7))
    
    # Inner Strassen coefficients for 2x2 multiplication
    # m1 = (P00+P11)(Q00+Q11)
    # m2 = (P10+P11)Q00
    # m3 = P00(Q01-Q11)
    # m4 = P11(Q10-Q00)
    # m5 = (P00+P01)Q11
    # m6 = (P10-P00)(Q00+Q01)
    # m7 = (P01-P11)(Q10+Q11)
    
    inner_coeffs = [
        # m1
        ([ (0,0), (1,1) ], [ (0,0), (1,1) ]),
        # m2
        ([ (1,0), (1,1) ], [ (0,0) ]),
        # m3
        ([ (0,0) ], [ (0,1), (1,1) ]), # (0,1) - (1,1)
        # m4
        ([ (1,1) ], [ (1,0), (0,0) ]), # (1,0) - (0,0)
        # m5
        ([ (0,0), (0,1) ], [ (1,1) ]),
        # m6
        ([ (1,0), (0,0) ], [ (0,0), (0,1) ]), # (1,0)-(0,0), (0,0)+(0,1)
        # m7
        ([ (0,1), (1,1) ], [ (1,0), (1,1) ])  # (0,1)-(1,1), (1,0)+(1,1)
    ]
    
    # Note: The lists above indicate which P entries and Q entries are involved.
    # We need to handle the +/- signs.
    
    # Let's define the inner Strassen terms more explicitly with signs.
    # Term k: P_form, Q_form, W_contrib
    
    # For each outer product r (0..6), we have 7 inner terms s (0..6).
    col = 0
    for r in range(7):
        P_r, Q_r = outer_products[r]
        for s in range(7):
            # Determine P_form and Q_form for this inner term
            # s=0: m1 = (P00+P11)(Q00+Q11)
            if s == 0:
                P_form = add_forms(P_r[0][0], P_r[1][1])
                Q_form = add_forms(Q_r[0][0], Q_r[1][1])
            elif s == 1:
                P_form = add_forms(P_r[1][0], P_r[1][1])
                Q_form = Q_r[0][0]
            elif s == 2:
                P_form = P_r[0][0]
                Q_form = sub_forms(Q_r[0][1], Q_r[1][1])
            elif s == 3:
                P_form = P_r[1][1]
                Q_form = sub_forms(Q_r[1][0], Q_r[0][0])
            elif s == 4:
                P_form = add_forms(P_r[0][0], P_r[0][1])
                Q_form = Q_r[1][1]
            elif s == 5:
                P_form = sub_forms(P_r[1][0], P_r[0][0])
                Q_form = add_forms(Q_r[0][0], Q_r[0][1])
            else: # s == 6
                P_form = sub_forms(P_r[0][1], P_r[1][1])
                Q_form = add_forms(Q_r[1][0], Q_r[1][1])
            
            # Determine W contribution
            # Inner Strassen result C_local (2x2)
            # C00 = m1 + m4 - m5 + m7
            # C01 = m3 + m5
            # C10 = m2 + m4
            # C11 = m1 - m2 + m3 + m6
            
            # We need to map C_local (a,b) to C_global (I,J)
            # C_global is 4x4.
            # Block r contributes to C_global with weights from Outer Strassen.
            
            # Outer Strassen weights for M_r:
            # M1: C11(+), C22(+)
            # M2: C21(+), C22(-)
            # M3: C12(+), C22(+)
            # M4: C11(+), C21(+)
            # M5: C11(-), C12(+)
            # M6: C22(+)
            # M7: C11(+)
            
            # Let's define outer_weights[r] = { (block_row, block_col): weight }
            # block_row, block_col in {0,1}
            
            if r == 0: # M1
                outer_w = { (0,0): 1, (1,1): 1 }
            elif r == 1: # M2
                outer_w = { (1,0): 1, (1,1): -1 }
            elif r == 2: # M3
                outer_w = { (0,1): 1, (1,1): 1 }
            elif r == 3: # M4
                outer_w = { (0,0): 1, (1,0): 1 }
            elif r == 4: # M5
                outer_w = { (0,0): -1, (0,1): 1 }
            elif r == 5: # M6
                outer_w = { (1,1): 1 }
            else: # r == 6, M7
                outer_w = { (0,0): 1 }
                
            # Inner weights for m_s
            if s == 0: # m1
                inner_w = { (0,0): 1, (1,1): 1 }
            elif s == 1: # m2
                inner_w = { (1,0): 1, (1,1): -1 }
            elif s == 2: # m3
                inner_w = { (0,1): 1, (1,1): 1 }
            elif s == 3: # m4
                inner_w = { (0,0): 1, (1,0): 1 }
            elif s == 4: # m5
                inner_w = { (0,0): -1, (0,1): 1 }
            elif s == 5: # m6
                inner_w = { (1,1): 1 }
            else: # s == 6, m7
                inner_w = { (0,0): 1 }
                
            # Combine
            W_form = {}
            for (br, bc), w_out in outer_w.items():
                for (lr, lc), w_in in inner_w.items():
                    # Global row I = br * 2 + lr
                    # Global col J = bc * 2 + lc
                    I = br * 2 + lr
                    J = bc * 2 + lc
                    W_form[(J, I)] = W_form.get((J, I), 0) + w_out * w_in
            
            add_term(P_form, Q_form, W_form, col)
            col += 1
            
    # Part 2: A (4x4) * B_right (4x1) using Schoolbook (Rank 16)
    # B_right is B[:, 4]
    # C_right is C[:, 4]
    for i in range(4):
        for j in range(4):
            term_u = { (i, j): 1 }
            term_v = { (j, 4): 1 }
            term_w = { (4, i): 1 }
            add_term(term_u, term_v, term_w, col)
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
