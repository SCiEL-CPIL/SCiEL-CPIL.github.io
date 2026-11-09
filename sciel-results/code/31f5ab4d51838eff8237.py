import numpy as np

def construct():
    """
    Returns a rank-49 decomposition for 4x4 matrix multiplication.
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    n = 4
    rank = 49
    
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))
    
    # Helper to get indices
    # A is 4x4, indexed by i, k. U index: i * 4 + k
    # B is 4x4, indexed by k, j. V index: k * 4 + j
    # C is 4x4, indexed by i, j. W index: j * 4 + i (based on prompt: W[k*n+i] where k=col, i=row)
    
    def u_idx(i, k):
        return i * n + k
        
    def v_idx(k, j):
        return k * n + j
        
    def w_idx(i, j):
        return j * n + i

    # Strassen decomposition for 4x4
    # We split A, B into 2x2 blocks of size 2x2.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A * B = [[C00, C01], [C10, C11]]
    
    # Strassen products:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Px is a 2x2 matrix product.
    # We compute each 2x2 product using Strassen's algorithm (7 scalar mults).
    
    # We will iterate through the 7 Strassen terms (P1..P7).
    # For each term, we define the linear combinations of A and B blocks.
    # Then we apply the 7 scalar Strassen steps to those 2x2 blocks.
    
    # The 7 scalar Strassen steps for X * Y = Z (where X, Y, Z are 2x2):
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We map these to the global 4x4 indices.
    # Block (bi, bj) of A corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # Element (r, c) in block (bi, bj) is at global (2*bi+r, 2*bj+c).
    
    def global_a(bi, bj, r, c):
        return 2*bi + r, 2*bj + c
        
    def global_b(bi, bj, r, c):
        return 2*bi + r, 2*bj + c
        
    def global_c(bi, bj, r, c):
        return 2*bi + r, 2*bj + c

    # Define the 7 block-level terms
    # Each term is defined by:
    # - U_coeffs: list of (global_row, global_col, coeff)
    # - V_coeffs: list of (global_row, global_col, coeff)
    # - W_contrib: dict mapping (block_row, block_col) to list of (global_row, global_col, coeff)
    #   Wait, W is indexed by global output indices.
    #   The term P_t contributes to specific blocks of C.
    #   For example, P1 contributes to C00 and C11.
    #   Inside C00 (which is 2x2), the result of the 2x2 Strassen is distributed.
    
    # Let's structure the data:
    # For each of the 7 block terms (t=0..6):
    #   U_lin: list of (u_idx, coeff)
    #   V_lin: list of (v_idx, coeff)
    #   W_map: dict { (c_block_row, c_block_col): [ (c_global_row, c_global_col, coeff), ... ] }
    #     Actually, the 2x2 result Z has elements Z00, Z01, Z10, Z11.
    #     These map to specific positions in the 2x2 block of C.
    
    # The 7 scalar steps (k=0..6) for a 2x2 product:
    #   U_scalar: list of (u_idx, coeff)
    #   V_scalar: list of (v_idx, coeff)
    #   W_scalar: list of (z_pos, coeff) where z_pos is (0,0), (0,1), etc.
    
    # We will combine these.
    
    # Block terms definitions
    # t=0: P1 = (A00+A11)(B00+B11)
    # t=1: P2 = (A10+A11)B00
    # t=2: P3 = A00(B01-B11)
    # t=3: P4 = A11(B10-B00)
    # t=4: P5 = (A00+A01)B11
    # t=5: P6 = (A10-A00)(B00+B01)
    # t=6: P7 = (A01-A11)(B10+B11)
    
    block_terms = []
    
    # Helper to add to a list of (idx, coeff)
    def add_to_list(lst, idx, coeff):
        lst.append((idx, coeff))
        
    # Helper to merge lists into a dict for accumulation
    def merge_dict(lst):
        d = {}
        for idx, c in lst:
            d[idx] = d.get(idx, 0) + c
        return d

    # Define U and V linear forms for each block term
    # We need to express the 2x2 matrices L_t and R_t in terms of global A and B indices.
    
    # For each term t, we define the 4 elements of L_t (2x2) and R_t (2x2).
    # L_t[r, c] is a linear combination of A elements.
    # R_t[r, c] is a linear combination of B elements.
    
    # Then we apply the 7 scalar Strassen steps to L_t and R_t.
    
    # Scalar Strassen steps for 2x2:
    # M1 = (L00+L11)(R00+R11)
    # M2 = (L10+L11)R00
    # M3 = L00(R01-R11)
    # M4 = L11(R10-R00)
    # M5 = (L00+L01)R11
    # M6 = (L10-L00)(R00+R01)
    # M7 = (L01-L11)(R10+R11)
    
    # Result Z:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We will generate the U, V, W contributions for each of the 49 rank terms.
    
    for t in range(7):
        # Define L and R matrices (2x2) as lists of dicts {global_idx: coeff}
        L = [[{} for _ in range(2)] for _ in range(2)]
        R = [[{} for _ in range(2)] for _ in range(2)]
        
        # Define which blocks of C this term contributes to
        # contrib: dict { (block_row, block_col): coeff }
        # The 2x2 result Z will be added to C at these blocks.
        contrib = {}
        
        if t == 0:
            # P1 = (A00+A11)(B00+B11)
            for r in range(2):
                for c in range(2):
                    # L[r,c] = A00[r,c] + A11[r,c]
                    add_to_dict(L[r][c], u_idx(*global_a(0,0,r,c)), 1)
                    add_to_dict(L[r][c], u_idx(*global_a(1,1,r,c)), 1)
                    # R[r,c] = B00[r,c] + B11[r,c]
                    add_to_dict(R[r][c], v_idx(*global_b(0,0,r,c)), 1)
                    add_to_dict(R[r][c], v_idx(*global_b(1,1,r,c)), 1)
            # Contributes to C00 (+1) and C11 (+1)
            contrib[(0,0)] = 1
            contrib[(1,1)] = 1
            
        elif t == 1:
            # P2 = (A10+A11)B00
            for r in range(2):
                for c in range(2):
                    add_to_dict(L[r][c], u_idx(*global_a(1,0,r,c)), 1)
                    add_to_dict(L[r][c], u_idx(*global_a(1,1,r,c)), 1)
                    add_to_dict(R[r][c], v_idx(*global_b(0,0,r,c)), 1)
            # Contributes to C10 (+1) and C11 (-1)
            contrib[(1,0)] = 1
            contrib[(1,1)] = -1
            
        elif t == 2:
            # P3 = A00(B01-B11)
            for r in range(2):
                for c in range(2):
                    add_to_dict(L[r][c], u_idx(*global_a(0,0,r,c)), 1)
                    add_to_dict(R[r][c], v_idx(*global_b(0,1,r,c)), 1)
                    add_to_dict(R[r][c], v_idx(*global_b(1,1,r,c)), -1)
            # Contributes to C01 (+1) and C11 (+1)
            contrib[(0,1)] = 1
            contrib[(1,1)] = 1
            
        elif t == 3:
            # P4 = A11(B10-B00)
            for r in range(2):
                for c in range(2):
                    add_to_dict(L[r][c], u_idx(*global_a(1,1,r,c)), 1)
                    add_to_dict(R[r][c], v_idx(*global_b(1,0,r,c)), 1)
                    add_to_dict(R[r][c], v_idx(*global_b(0,0,r,c)), -1)
            # Contributes to C00 (+1) and C10 (+1)
            contrib[(0,0)] = 1
            contrib[(1,0)] = 1
            
        elif t == 4:
            # P5 = (A00+A01)B11
            for r in range(2):
                for c in range(2):
                    add_to_dict(L[r][c], u_idx(*global_a(0,0,r,c)), 1)
                    add_to_dict(L[r][c], u_idx(*global_a(0,1,r,c)), 1)
                    add_to_dict(R[r][c], v_idx(*global_b(1,1,r,c)), 1)
            # Contributes to C00 (-1) and C01 (+1)
            contrib[(0,0)] = -1
            contrib[(0,1)] = 1
            
        elif t == 5:
            # P6 = (A10-A00)(B00+B01)
            for r in range(2):
                for c in range(2):
                    add_to_dict(L[r][c], u_idx(*global_a(1,0,r,c)), 1)
                    add_to_dict(L[r][c], u_idx(*global_a(0,0,r,c)), -1)
                    add_to_dict(R[r][c], v_idx(*global_b(0,0,r,c)), 1)
                    add_to_dict(R[r][c], v_idx(*global_b(0,1,r,c)), 1)
            # Contributes to C11 (+1)
            contrib[(1,1)] = 1
            
        elif t == 6:
            # P7 = (A01-A11)(B10+B11)
            for r in range(2):
                for c in range(2):
                    add_to_dict(L[r][c], u_idx(*global_a(0,1,r,c)), 1)
                    add_to_dict(L[r][c], u_idx(*global_a(1,1,r,c)), -1)
                    add_to_dict(R[r][c], v_idx(*global_b(1,0,r,c)), 1)
                    add_to_dict(R[r][c], v_idx(*global_b(1,1,r,c)), 1)
            # Contributes to C00 (+1)
            contrib[(0,0)] = 1
            
        # Now apply 7 scalar Strassen steps to L and R
        # Extract L00, L01, L10, L11 and R00, R01, R10, R11
        L00, L01, L10, L11 = L[0][0], L[0][1], L[1][0], L[1][1]
        R00, R01, R10, R11 = R[0][0], R[0][1], R[1][0], R[1][1]
        
        # Define the 7 scalar products
        # Each is a pair (U_dict, V_dict)
        # And a contribution to Z00, Z01, Z10, Z11
        
        # M1 = (L00+L11)(R00+R11)
        U1 = merge_dict(list(L00.items()) + list(L11.items()))
        V1 = merge_dict(list(R00.items()) + list(R11.items()))
        # Z00 += M1, Z11 += M1
        W1 = {(0,0): 1, (1,1): 1}
        
        # M2 = (L10+L11)R00
        U2 = merge_dict(list(L10.items()) + list(L11.items()))
        V2 = dict(R00)
        # Z10 += M2, Z11 -= M2
        W2 = {(1,0): 1, (1,1): -1}
        
        # M3 = L00(R01-R11)
        U3 = dict(L00)
        V3 = merge_dict(list(R01.items()) + [(k, -v) for k, v in R11.items()])
        # Z01 += M3, Z11 += M3
        W3 = {(0,1): 1, (1,1): 1}
        
        # M4 = L11(R10-R00)
        U4 = dict(L11)
        V4 = merge_dict(list(R10.items()) + [(k, -v) for k, v in R00.items()])
        # Z00 += M4, Z10 += M4
        W4 = {(0,0): 1, (1,0): 1}
        
        # M5 = (L00+L01)R11
        U5 = merge_dict(list(L00.items()) + list(L01.items()))
        V5 = dict(R11)
        # Z00 -= M5, Z01 += M5
        W5 = {(0,0): -1, (0,1): 1}
        
        # M6 = (L10-L00)(R00+R01)
        U6 = merge_dict(list(L10.items()) + [(k, -v) for k, v in L00.items()])
        V6 = merge_dict(list(R00.items()) + list(R01.items()))
        # Z11 += M6
        W6 = {(1,1): 1}
        
        # M7 = (L01-L11)(R10+R11)
        U7 = merge_dict(list(L01.items()) + [(k, -v) for k, v in L11.items()])
        V7 = merge_dict(list(R10.items()) + list(R11.items()))
        # Z00 += M7
        W7 = {(0,0): 1}
        
        scalar_products = [
            (U1, V1, W1),
            (U2, V2, W2),
            (U3, V3, W3),
            (U4, V4, W4),
            (U5, V5, W5),
            (U6, V6, W6),
            (U7, V7, W7)
        ]
        
        for k in range(7):
            rank_idx = t * 7 + k
            U_dict, V_dict, W_dict = scalar_products[k]
            
            for u_idx_val, coeff in U_dict.items():
                U[u_idx_val, rank_idx] += coeff
                
            for v_idx_val, coeff in V_dict.items():
                V[v_idx_val, rank_idx] += coeff
                
            for (z_r, z_c), z_coeff in W_dict.items():
                # This scalar product contributes to the 2x2 block of C.
                # The 2x2 block is at position (block_row, block_col) in C.
                # The element (z_r, z_c) within that block maps to global (2*block_row+z_r, 2*block_col+z_c).
                for (block_r, block_c), block_coeff in contrib.items():
                    global_i = 2 * block_r + z_r
                    global_j = 2 * block_c + z_c
                    w_idx_val = w_idx(global_i, global_j)
                    W[w_idx_val, rank_idx] += z_coeff * block_coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def add_to_dict(d, key, val):
    d[key] = d.get(key, 0) + val
