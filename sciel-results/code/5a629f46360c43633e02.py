import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 97 solution:
    # 4x4x4 Strassen (rank 49) + 4x4x3 Schoolbook (rank 48)
    
    R = 97
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for 4x4x4 (cols 0-3 of B) ---
    
    # Define linear forms for X_k and Y_k
    # X_k and Y_k are 2x2 matrices of linear forms.
    # We represent a linear form as a dict {index: coeff}
    # A indices: 0..15 (i*m + j)
    # B indices: 0..27 (j*p + k) -- but for the first 4 cols, k is 0..3
    
    def make_a(i, j, coeff=1):
        return {i * m + j: coeff}
        
    def make_b(j, k, coeff=1):
        return {j * p + k: coeff}
        
    def add_form(d1, d2, sign=1):
        # d1 + sign * d2
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0) + sign * v
        return res
        
    # X_k and Y_k definitions
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A12: (0,2), (0,3), (1,2), (1,3)
    # A21: (2,0), (2,1), (3,0), (3,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    
    A11 = [[make_a(0,0), make_a(0,1)], [make_a(1,0), make_a(1,1)]]
    A12 = [[make_a(0,2), make_a(0,3)], [make_a(1,2), make_a(1,3)]]
    A21 = [[make_a(2,0), make_a(2,1)], [make_a(3,0), make_a(3,1)]]
    A22 = [[make_a(2,2), make_a(2,3)], [make_a(3,2), make_a(3,3)]]
    
    B11 = [[make_b(0,0), make_b(0,1)], [make_b(1,0), make_b(1,1)]]
    B12 = [[make_b(0,2), make_b(0,3)], [make_b(1,2), make_b(1,3)]]
    B21 = [[make_b(2,0), make_b(2,1)], [make_b(3,0), make_b(3,1)]]
    B22 = [[make_b(2,2), make_b(2,3)], [make_b(3,2), make_b(3,3)]]
    
    def add_matrix(M1, M2, sign=1):
        return [[add_form(M1[r][c], M2[r][c], sign) for c in range(2)] for r in range(2)]
        
    def sub_matrix(M1, M2):
        return add_matrix(M1, M2, -1)
        
    X1 = add_matrix(A11, A22)
    Y1 = add_matrix(B11, B22)
    
    X2 = add_matrix(A21, A22)
    Y2 = B11
    
    X3 = A11
    Y3 = sub_matrix(B12, B22)
    
    X4 = A22
    Y4 = sub_matrix(B21, B11)
    
    X5 = add_matrix(A11, A12)
    Y5 = B22
    
    X6 = sub_matrix(A21, A11)
    Y6 = add_matrix(B11, B12)
    
    X7 = sub_matrix(A12, A22)
    Y7 = add_matrix(B21, B22)
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # Mapping of block products to C blocks and signs
    # C11 = M1 - M4 + M5 - M6
    # C12 = M3 + M6
    # C21 = M2 + M7
    # C22 = M1 - M3 + M4 + M7
    
    # For each k (1..7), for each l (1..7 Strassen terms),
    # we determine the contribution to C.
    
    # Strassen term l contributes to Z positions:
    # l=1: Z00(+1), Z11(+1)
    # l=2: Z10(+1)
    # l=3: Z01(+1), Z11(-1)
    # l=4: Z00(-1), Z11(+1)
    # l=5: Z00(+1)
    # l=6: Z00(-1), Z01(+1)
    # l=7: Z10(+1), Z11(+1)
    
    z_contrib = {
        1: [(0, 0, 1), (1, 1, 1)],
        2: [(1, 0, 1)],
        3: [(0, 1, 1), (1, 1, -1)],
        4: [(0, 0, -1), (1, 1, 1)],
        5: [(0, 0, 1)],
        6: [(0, 0, -1), (0, 1, 1)],
        7: [(1, 0, 1), (1, 1, 1)]
    }
    
    # Map k (1..7) to list of (block_row, block_col, sign)
    # Block indices: 0:11, 1:12, 2:21, 3:22
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    k_to_c = {
        1: [(0, 0, 1), (1, 1, 1)], # M1 -> C11(+), C22(+)
        2: [(1, 0, 1)],            # M2 -> C21(+)
        3: [(0, 1, 1), (1, 1, -1)],# M3 -> C12(+), C22(-)
        4: [(0, 0, -1), (1, 1, 1)],# M4 -> C11(-), C22(+)
        5: [(0, 0, 1)],            # M5 -> C11(+)
        6: [(0, 0, -1), (0, 1, 1)],# M6 -> C11(-), C12(+)
        7: [(1, 0, 1), (1, 1, 1)]  # M7 -> C21(+), C22(+)
    }
    
    for k in range(1, 8):
        Xk = Xs[k-1]
        Yk = Ys[k-1]
        
        # Generate the 7 linear forms for Xk and Yk corresponding to Strassen terms
        # x00, x01, x10, x11
        # y00, y01, y10, y11
        
        x00, x01 = Xk[0][0], Xk[0][1]
        x10, x11 = Xk[1][0], Xk[1][1]
        
        y00, y01 = Yk[0][0], Yk[0][1]
        y10, y11 = Yk[1][0], Yk[1][1]
        
        # Term 1: (x00 + x11)(y00 + y11)
        a1 = add_form(x00, x11)
        b1 = add_form(y00, y11)
        
        # Term 2: (x10 + x11)y00
        a2 = add_form(x10, x11)
        b2 = y00
        
        # Term 3: x00(y01 - y11)
        a3 = x00
        b3 = sub_matrix([y01], [y11])[0] if False else add_form(y01, y11, -1) # y01 is dict, not matrix
        # Correction: y01, y11 are dicts.
        b3 = add_form(y01, y11, -1)
        
        # Term 4: x11(y10 - y00)
        a4 = x11
        b4 = add_form(y10, y00, -1)
        
        # Term 5: (x00 + x01)y11
        a5 = add_form(x00, x01)
        b5 = y11
        
        # Term 6: (x10 - x00)(y00 + y01)
        a6 = add_form(x10, x00, -1)
        b6 = add_form(y00, y01)
        
        # Term 7: (x01 - x11)(y10 + y11)
        a7 = add_form(x01, x11, -1)
        b7 = add_form(y10, y11)
        
        terms = [
            (a1, b1),
            (a2, b2),
            (a3, b3),
            (a4, b4),
            (a5, b5),
            (a6, b6),
            (a7, b7)
        ]
        
        for l in range(7):
            a, b = terms[l]
            
            # Initialize U, V, W columns
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for idx, coeff in a.items():
                u_col[idx] = coeff
                
            for idx, coeff in b.items():
                v_col[idx] = coeff
                
            # Determine W contributions
            # For each Z position (zi, zj) with sign sz from z_contrib[l]
            for zi, zj, sz in z_contrib[l+1]:
                # For each C block (br, bc) with sign sc from k_to_c[k]
                for br, bc, sc in k_to_c[k]:
                    total_sign = sz * sc
                    # Global row r = 2*br + zi
                    # Global col c = 2*bc + zj
                    r = 2 * br + zi
                    c = 2 * bc + zj
                    # W index: c * n + r
                    w_idx = c * n + r
                    w_col[w_idx] += total_sign
            
            # Add to matrices
            for i in range(n * m):
                U[i][col] = u_col[i]
            for j in range(m * p):
                V[j][col] = v_col[j]
            for k_idx in range(p * n):
                W[k_idx][col] = w_col[k_idx]
                
            col += 1
            
    # --- Part 2: Schoolbook for 4x4x3 (cols 4-6 of B) ---
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
