import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 4, 4, 7
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)

    # Strassen 2x2x2 base terms (7 terms)
    # Each term: (u_indices, v_indices, w_indices) with coefficients
    # For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h) -> C00 + C11
    # M2 = (c+d)e -> C10 - C11
    # M3 = a(f-h) -> C01 + C11
    # M4 = d(g-e) -> C00 + C10
    # M5 = (a+b)h -> -C00 + C01
    # M6 = (c-a)(e+f) -> C11
    # M7 = (b-d)(g+h) -> C00

    def strassen_2x2(i_off, j_off, k_off):
        """Add 7 Strassen terms for 2x2 block at offset (i_off, j_off, k_off)"""
        # Local indices within 2x2 block
        # A local: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B local: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C local: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

        # M1: (a+d)(e+h) -> C00 + C11
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + j_off] = 1; u[(i_off+1)*m + (j_off+1)] = 1
        v[j_off*p + k_off] = 1; v[(j_off+1)*p + (k_off+1)] = 1
        w[k_off*n + i_off] = 1; w[(k_off+1)*n + (i_off+1)] = 1
        add_term(u, v, w)

        # M2: (c+d)e -> C10 - C11
        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off+1)*m + j_off] = 1; u[(i_off+1)*m + (j_off+1)] = 1
        v[j_off*p + k_off] = 1
        w[k_off*n + (i_off+1)] = 1; w[(k_off+1)*n + (i_off+1)] = -1
        add_term(u, v, w)

        # M3: a(f-h) -> C01 + C11
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + j_off] = 1
        v[j_off*p + (k_off+1)] = 1; v[(j_off+1)*p + (k_off+1)] = -1
        w[(k_off+1)*n + i_off] = 1; w[(k_off+1)*n + (i_off+1)] = 1
        add_term(u, v, w)

        # M4: d(g-e) -> C00 + C10
        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off+1)*m + (j_off+1)] = 1
        v[(j_off+1)*p + k_off] = 1; v[j_off*p + k_off] = -1
        w[k_off*n + i_off] = 1; w[k_off*n + (i_off+1)] = 1
        add_term(u, v, w)

        # M5: (a+b)h -> -C00 + C01
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + j_off] = 1; u[i_off*m + (j_off+1)] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[(k_off+1)*n + i_off] = 1; w[k_off*n + i_off] = -1
        add_term(u, v, w)

        # M6: (c-a)(e+f) -> C11
        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off+1)*m + j_off] = 1; u[i_off*m + j_off] = -1
        v[j_off*p + k_off] = 1; v[j_off*p + (k_off+1)] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        add_term(u, v, w)

        # M7: (b-d)(g+h) -> C00
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + (j_off+1)] = 1; u[(i_off+1)*m + (j_off+1)] = -1
        v[(j_off+1)*p + k_off] = 1; v[(j_off+1)*p + (k_off+1)] = 1
        w[k_off*n + i_off] = 1
        add_term(u, v, w)

    # Part 1: 4x4x4 using recursive Strassen (49 terms)
    # Outer Strassen on 2x2 blocks, each block multiplied via inner Strassen
    # A 4x4 has 4 2x2 blocks: A00=(rows 0-1, cols 0-1), A01=(rows 0-1, cols 2-3),
    #                          A10=(rows 2-3, cols 0-1), A11=(rows 2-3, cols 2-3)
    # Outer Strassen terms combine these blocks, then each product is a 2x2*2x2
    
    # For each outer term, we have linear combinations of A-blocks, B-blocks, C-blocks
    # Each such combination requires multiplying the resulting 2x2 matrices
    # We expand each outer term into 7 inner Strassen terms
    
    # Outer term definitions: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    outer_strassen = [
        # M1: (A00+A11)(B00+B11) -> C00+C11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2: (A10+A11)B00 -> C10-C11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3: A00(B01-B11) -> C01+C11
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4: A11(B10-B00) -> C00+C10
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5: (A00+A01)B11 -> -C00+C01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6: (A10-A00)(B00+B01) -> C11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7: (A01-A11)(B10+B11) -> C00
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    for oa, ob, oc in outer_strassen:
        for ia, ib, ic in outer_strassen:
            # For this pair of outer terms, we need to compute the 2x2*2x2 products
            # The U part is a linear combination of A-blocks from ia
            # The V part is a linear combination of B-blocks from ib
            # The W part is a linear combination of C-blocks from ic
            # But we need to expand each block product into scalar multiplications
            
            # Actually, the correct approach: each outer term defines linear combos of blocks
            # The product of two such linear combos is a bilinear form
            # We need to decompose this into rank-1 terms over scalars
            
            # For term (oa, ob, oc), the contribution is:
            # sum_{a_blocks in oa} sum_{b_blocks in ob} coeff_a * coeff_b * (A_block * B_block)
            # Each A_block * B_block is a 2x2*2x2 multiplication = 7 Strassen terms
            # And the result goes into C_blocks per oc with appropriate coefficients
            
            u = zero_u(); v = zero_v(); w = zero_w()
            
            # For each pair of A-block and B-block in the outer terms,
            # add the 7 Strassen terms for their 2x2*2x2 product
            # The W coefficients come from the C-block mapping
            
            for (ba, ca) in oa:
                for (bb, cb) in ob:
                    # A_block ba is at rows (ba//2)*2, cols (ba%2)*2
                    # B_block bb is at rows (bb//2)*2, cols (bb%2)*2
                    # Their product goes to C-blocks per oc
                    # But we need to map the 2x2 result to the correct C positions
                    
                    # The product of A[ba] and B[bb] is a 2x2 matrix
                    # This 2x2 matrix should be placed at C positions determined by oc
                    # For each C-block in oc with coefficient cc, we add cc * (the 2x2 product)
                    
                    # To do this properly, we need to expand the 2x2*2x2 into scalar terms
                    # and map each scalar to the correct global position
                    
                    a_row_off = (ba // 2) * 2
                    a_col_off = (ba % 2) * 2
                    b_row_off = (bb // 2) * 2
                    b_col_off = (bb % 2) * 2
                    
                    # For each C-block in oc, we need to add the 2x2 product scaled by cc
                    # The C-block at position (cr, cc) in the 2x2 block grid
                    # corresponds to global rows cr*2, cols cc*2 in C
                    # But C is p x n = 7x4, and we're doing 4x4x4 so C is 4x4
                    # C block (cr, cc) is at rows cr*2 to cr*2+1, cols cc*2 to cc*2+1
                    
                    for (bc, cc_coeff) in oc:
                        c_row_off = (bc // 2) * 2
                        c_col_off = (bc % 2) * 2
                        
                        # Add 7 Strassen terms for A_block(ba) * B_block(bb) -> C_block(bc)
                        # with overall coefficient ca * cb * cc_coeff
                        
                        # Use the 7 Strassen terms for 2x2*2x2
                        # M1: (a00+a11)(b00+b11) -> c00+c11
                        u2 = zero_u(); v2 = zero_v(); w2 = zero_w()
                        u2[a_row_off*m + a_col_off] = ca * cb * cc_coeff
                        u2[(a_row_off+1)*m + (a_col_off+1)] = ca * cb * cc_coeff
                        v2[b_row_off*p + b_col_off] = 1
                        v2[(b_row_off+1)*p + (b_col_off+1)] = 1
                        w2[c_col_off*n + c_row_off] = 1
                        w2[(c_col_off+1)*n + (c_row_off+1)] = 1
                        # Add to main terms
                        for i in range(len(u)):
                            u[i] += u2[i]
                            v[i] += v2[i]
                            w[i] += w2[i]
                        
                        # M2: (a10+a11)b00 -> c10-c11
                        u2 = zero_u(); v2 = zero_v(); w2 = zero_w()
                        u2[(a_row_off+1)*m + a_col_off] = ca * cb * cc_coeff
                        u2[(a_row_off+1)*m + (a_col_off+1)] = ca * cb * cc_coeff
                        v2[b_row_off*p + b_col_off] = 1
                        w2[c_col_off*n + (c_row_off+1)] = 1
                        w2[(c_col_off+1)*n + (c_row_off+1)] = -1
                        for i in range(len(u)):
                            u[i] += u2[i]
                            v[i] += v2[i]
                            w[i] += w2[i]
                        
                        # M3: a00(b01-b11) -> c01+c11
                        u2 = zero_u(); v2 = zero_v(); w2 = zero_w()
                        u2[a_row_off*m + a_col_off] = ca * cb * cc_coeff
                        v2[b_row_off*p + (b_col_off+1)] = 1
                        v2[(b_row_off+1)*p + (b_col_off+1)] = -1
                        w2[(c_col_off+1)*n + c_row_off] = 1
                        w2[(c_col_off+1)*n + (c_row_off+1)] = 1
                        for i in range(len(u)):
                            u[i] += u2[i]
                            v[i] += v2[i]
                            w[i] += w2[i]
                        
                        # M4: a11(b10-b00) -> c00+c10
                        u2 = zero_u(); v2 = zero_v(); w2 = zero_w()
                        u2[(a_row_off+1)*m + (a_col_off+1)] = ca * cb * cc_coeff
                        v2[(b_row_off+1)*p + b_col_off] = 1
                        v2[b_row_off*p + b_col_off] = -1
                        w2[c_col_off*n + c_row_off] = 1
                        w2[c_col_off*n + (c_row_off+1)] = 1
                        for i in range(len(u)):
                            u[i] += u2[i]
                            v[i] += v2[i]
                            w[i] += w2[i]
                        
                        # M5: (a00+a01)b11 -> -c00+c01
                        u2 = zero_u(); v2 = zero_v(); w2 = zero_w()
                        u2[a_row_off*m + a_col_off] = ca * cb * cc_coeff
                        u2[a_row_off*m + (a_col_off+1)] = ca * cb * cc_coeff
                        v2[(b_row_off+1)*p + (b_col_off+1)] = 1
                        w2[(c_col_off+1)*n + c_row_off] = 1
                        w2[c_col_off*n + c_row_off] = -1
                        for i in range(len(u)):
                            u[i] += u2[i]
                            v[i] += v2[i]
                            w[i] += w2[i]
                        
                        # M6: (a10-a00)(b00+b01) -> c11
                        u2 = zero_u(); v2 = zero_v(); w2 = zero_w()
                        u2[(a_row_off+1)*m + a_col_off] = ca * cb * cc_coeff
                        u2[a_row_off*m + a_col_off] = -ca * cb * cc_coeff
                        v2[b_row_off*p + b_col_off] = 1
                        v2[b_row_off*p + (b_col_off+1)] = 1
                        w2[(c_col_off+1)*n + (c_row_off+1)] = 1
                        for i in range(len(u)):
                            u[i] += u2[i]
                            v[i] += v2[i]
                            w[i] += w2[i]
                        
                        # M7: (a01-a11)(b10+b11) -> c00
                        u2 = zero_u(); v2 = zero_v(); w2 = zero_w()
                        u2[a_row_off*m + (a_col_off+1)] = ca * cb * cc_coeff
                        u2[(a_row_off+1)*m + (a_col_off+1)] = -ca * cb * cc_coeff
                        v2[(b_row_off+1)*p + b_col_off] = 1
                        v2[(b_row_off+1)*p + (b_col_off+1)] = 1
                        w2[c_col_off*n + c_row_off] = 1
                        for i in range(len(u)):
                            u[i] += u2[i]
                            v[i] += v2[i]
                            w[i] += w2[i]
            
            add_term(u, v, w)

    # Part 2: 4x4x2 for columns 4-5 (28 terms)
    # 4 blocks of 2x2 * 2x2, each using 7 Strassen terms
    for i_block in range(2):
        for j_block in range(2):
            strassen_2x2(i_block*2, j_block*2, 4)

    # Part 3: 4x4x1 for column 6 (16 terms, schoolbook)
    for i in range(n):
        for j in range(m):
            u = zero_u(); v = zero_v(); w = zero_w()
            u[i*m + j] = 1
            v[j*p + 6] = 1
            w[6*n + i] = 1
            add_term(u, v, w)

    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
