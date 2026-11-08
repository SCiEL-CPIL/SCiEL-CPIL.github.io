import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    This is a rewrite of the logic using direct NumPy array manipulation for clarity and speed.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W as NumPy arrays for easier manipulation
    # U shape: (16, 49)
    # V shape: (16, 49)
    # W shape: (16, 49)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Helper to get linear index for U (A matrix)
    # A[i, j] -> U[i*4 + j, :]
    def u_idx(i, j):
        return i * 4 + j
        
    # Helper to get linear index for V (B matrix)
    # B[i, j] -> V[i*4 + j, :]
    def v_idx(i, j):
        return i * 4 + j
        
    # Helper to get linear index for W (C matrix, trace convention)
    # C[i, j] -> W[j*4 + i, :]
    def w_idx(i, j):
        return j * 4 + i

    # We define the 7 block-level Strassen terms for 4x4 matrices.
    # Each term corresponds to a 2x2 matrix multiplication of linear combinations.
    # We will iterate through the 7 block terms, and for each, iterate through the 7 scalar Strassen terms.
    
    # Block indices: 00, 01, 10, 11
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # Define the 7 block-level (L, R) pairs.
    # L and R are 2x2 matrices of "element references".
    # An element reference is a list of (row, col, coeff) for the global A or B matrix.
    
    def make_ref(row, col, coeff=1):
        return [(row, col, coeff)]
        
    def add_ref(ref1, ref2):
        # Combine two lists of refs, summing coeffs for same (row, col)
        d = {}
        for r, c, v in ref1:
            d[(r, c)] = d.get((r, c), 0) + v
        for r, c, v in ref2:
            d[(r, c)] = d.get((r, c), 0) + v
        return [(k[0], k[1], v) for k, v in d.items() if v != 0]
        
    def sub_ref(ref1, ref2):
        d = {}
        for r, c, v in ref1:
            d[(r, c)] = d.get((r, c), 0) + v
        for r, c, v in ref2:
            d[(r, c)] = d.get((r, c), 0) - v
        return [(k[0], k[1], v) for k, v in d.items() if v != 0]

    # A block (bi, bj) element (r, c) is at global (2*bi+r, 2*bj+c)
    def a_ref(bi, bj, r, c, coeff=1):
        return [(2*bi+r, 2*bj+c, coeff)]
        
    def b_ref(bi, bj, r, c, coeff=1):
        return [(2*bi+r, 2*bj+c, coeff)]

    # The 7 Strassen block terms (L, R)
    # L and R are 2x2 lists of refs.
    
    # Term 0: P1 = (A00 + A11)(B00 + B11)
    L0 = [[add_ref(a_ref(0,0,r,c), a_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    R0 = [[add_ref(b_ref(0,0,r,c), b_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    # Contrib to C: C00 += P1, C11 += P1
    contrib0 = { (0,0): [(0,0,1), (1,1,1)] } # This mapping is for the 2x2 result P to global C
    # Wait, the contrib structure in the original code was:
    # contrib_t[(pr, pc)] = list of (out_row, out_col, coeff)
    # For P1, the result 2x2 matrix adds to C00 and C11.
    # So for each element (pr, pc) of the 2x2 result P1, it contributes to C[2*0+pr, 2*0+pc] and C[2*1+pr, 2*1+pc].
    
    # Let's define contrib more simply.
    # For each of the 7 block terms, the 2x2 result P is added to specific 2x2 blocks of C with specific signs.
    
    # Block Term 0: P1. C00 += P1, C11 += P1
    # Block Term 1: P2. C10 += P2, C11 -= P2
    # Block Term 2: P3. C01 += P3, C11 += P3
    # Block Term 3: P4. C00 += P4, C10 += P4
    # Block Term 4: P5. C00 -= P5, C01 += P5
    # Block Term 5: P6. C11 += P6
    # Block Term 6: P7. C00 += P7
    
    # We will store these as a list of (block_row, block_col, sign) for each term.
    # When the 2x2 result P (from scalar Strassen) is computed, element (pr, pc) of P
    # contributes to global C at (2*block_row + pr, 2*block_col + pc) with the sign.
    
    block_contribs = [
        [(0, 0, 1), (1, 1, 1)], # Term 0
        [(1, 0, 1), (1, 1, -1)], # Term 1
        [(0, 1, 1), (1, 1, 1)], # Term 2
        [(0, 0, 1), (1, 0, 1)], # Term 3
        [(0, 0, -1), (0, 1, 1)], # Term 4
        [(1, 1, 1)], # Term 5
        [(0, 0, 1)] # Term 6
    ]

    # Define L and R for all 7 terms
    # Term 0: (A00+A11)(B00+B11)
    L0 = [[add_ref(a_ref(0,0,r,c), a_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    R0 = [[add_ref(b_ref(0,0,r,c), b_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    
    # Term 1: (A10+A11)B00
    L1 = [[add_ref(a_ref(1,0,r,c), a_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    R1 = [[b_ref(0,0,r,c) for c in range(2)] for r in range(2)]
    
    # Term 2: A00(B01-B11)
    L2 = [[a_ref(0,0,r,c) for c in range(2)] for r in range(2)]
    R2 = [[sub_ref(b_ref(0,1,r,c), b_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    
    # Term 3: A11(B10-B00)
    L3 = [[a_ref(1,1,r,c) for c in range(2)] for r in range(2)]
    R3 = [[sub_ref(b_ref(1,0,r,c), b_ref(0,0,r,c)) for c in range(2)] for r in range(2)]
    
    # Term 4: (A00+A01)B11
    L4 = [[add_ref(a_ref(0,0,r,c), a_ref(0,1,r,c)) for c in range(2)] for r in range(2)]
    R4 = [[b_ref(1,1,r,c) for c in range(2)] for r in range(2)]
    
    # Term 5: (A10-A00)(B00+B01)
    L5 = [[sub_ref(a_ref(1,0,r,c), a_ref(0,0,r,c)) for c in range(2)] for r in range(2)]
    R5 = [[add_ref(b_ref(0,0,r,c), b_ref(0,1,r,c)) for c in range(2)] for r in range(2)]
    
    # Term 6: (A01-A11)(B10+B11)
    L6 = [[sub_ref(a_ref(0,1,r,c), a_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    R6 = [[add_ref(b_ref(1,0,r,c), b_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    
    L_terms = [L0, L1, L2, L3, L4, L5, L6]
    R_terms = [R0, R1, R2, R3, R4, R5, R6]
    
    # Strassen scalar coefficients for the 2x2 result
    # M1 = (a+d)(e+h) -> P11: +1, P22: +1
    # M2 = (c+d)e   -> P21: +1, P22: -1
    # M3 = a(f-h)    -> P12: +1, P22: +1
    # M4 = d(g-e)    -> P11: +1, P21: +1
    # M5 = (a+b)h    -> P11: -1, P12: +1
    # M6 = (c-a)(e+f)-> P22: +1
    # M7 = (b-d)(g+h)-> P11: +1
    
    sub_contrib = [
        [(0,0,1), (1,1,1)],    # M1
        [(1,0,1), (1,1,-1)],   # M2
        [(0,1,1), (1,1,1)],    # M3
        [(0,0,1), (1,0,1)],    # M4
        [(0,0,-1), (0,1,1)],   # M5
        [(1,1,1)],             # M6
        [(0,0,1)]              # M7
    ]
    
    for t in range(7):
        L_t = L_terms[t]
        R_t = R_terms[t]
        contrib_t = block_contribs[t]
        
        # Extract a,b,c,d from L_t and e,f,g,h from R_t
        # L_t[r][c] is a list of (row, col, coeff)
        
        a = L_t[0][0]
        b = L_t[0][1]
        c = L_t[1][0]
        d = L_t[1][1]
        e = R_t[0][0]
        f = R_t[0][1]
        g = R_t[1][0]
        h = R_t[1][1]
        
        # Define the 7 sub-products X_k * Y_k
        # X_k is a list of (row, col, coeff) for A
        # Y_k is a list of (row, col, coeff) for B
        
        sub_products = []
        
        # M1: X = a+d, Y = e+h
        x1 = add_ref(a, d)
        y1 = add_ref(e, h)
        sub_products.append((x1, y1))
        
        # M2: X = c+d, Y = e
        x2 = add_ref(c, d)
        y2 = e
        sub_products.append((x2, y2))
        
        # M3: X = a, Y = f-h
        x3 = a
        y3 = sub_ref(f, h)
        sub_products.append((x3, y3))
        
        # M4: X = d, Y = g-e
        x4 = d
        y4 = sub_ref(g, e)
        sub_products.append((x4, y4))
        
        # M5: X = a+b, Y = h
        x5 = add_ref(a, b)
        y5 = h
        sub_products.append((x5, y5))
        
        # M6: X = c-a, Y = e+f
        x6 = sub_ref(c, a)
        y6 = add_ref(e, f)
        sub_products.append((x6, y6))
        
        # M7: X = b-d, Y = g+h
        x7 = sub_ref(b, d)
        y7 = add_ref(g, h)
        sub_products.append((x7, y7))
        
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k
            
            # Update U
            for (row, col, coeff) in x_k:
                U[u_idx(row, col), rank_idx] += coeff
                
            # Update V
            for (row, col, coeff) in y_k:
                V[v_idx(row, col), rank_idx] += coeff
                
            # Update W
            # sub_contrib[k] gives the contribution of M_k to the 2x2 block result P
            # contrib_t gives the contribution of the 2x2 block result P to the global C
            for (pr, pc, sub_coeff) in sub_contrib[k]:
                for (br, bc, block_coeff) in contrib_t:
                    # Global C index: (2*br + pr, 2*bc + pc)
                    out_row = 2 * br + pr
                    out_col = 2 * bc + pc
                    total_coeff = sub_coeff * block_coeff
                    W[w_idx(out_row, out_col), rank_idx] += total_coeff

    # Convert to lists of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
