import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen (7x7).
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i # Trace convention W[k*n+i, r]
    
    # Outer Strassen combinations
    # L_k, R_k are 2x2 matrices of linear forms over A and B blocks.
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # We represent linear forms as lists of (index, coeff)
    # A indices: (0,0), (0,1), (1,0), (1,1) corresponding to u_idx
    # B indices: (0,0), (0,1), (1,0), (1,1) corresponding to v_idx
    
    # Helper to create a linear form for a single A element
    def a_form(i, j, coeff=1):
        return [(u_idx(i, j), coeff)]
        
    def b_form(i, j, coeff=1):
        return [(v_idx(i, j), coeff)]
        
    def add_forms(f1, f2):
        d = {}
        for idx, c in f1:
            d[idx] = d.get(idx, 0) + c
        for idx, c in f2:
            d[idx] = d.get(idx, 0) + c
        return [(k, v) for k, v in d.items() if v != 0]
        
    def sub_forms(f1, f2):
        d = {}
        for idx, c in f1:
            d[idx] = d.get(idx, 0) + c
        for idx, c in f2:
            d[idx] = d.get(idx, 0) - c
        return [(k, v) for k, v in d.items() if v != 0]
            
    # Define A block forms (each is a 2x2 matrix of linear forms)
    # A00: elements (0,0), (0,1), (1,0), (1,1)
    A00 = [[a_form(0,0), a_form(0,1)], [a_form(1,0), a_form(1,1)]]
    A01 = [[a_form(0,2), a_form(0,3)], [a_form(1,2), a_form(1,3)]]
    A10 = [[a_form(2,0), a_form(2,1)], [a_form(3,0), a_form(3,1)]]
    A11 = [[a_form(2,2), a_form(2,3)], [a_form(3,2), a_form(3,3)]]
    
    # Define B block forms
    B00 = [[b_form(0,0), b_form(0,1)], [b_form(1,0), b_form(1,1)]]
    B01 = [[b_form(0,2), b_form(0,3)], [b_form(1,2), b_form(1,3)]]
    B10 = [[b_form(2,0), b_form(2,1)], [b_form(3,0), b_form(3,1)]]
    B11 = [[b_form(2,2), b_form(2,3)], [b_form(3,2), b_form(3,3)]]
    
    def add_blocks(B1, B2):
        return [[add_forms(B1[r][c], B2[r][c]) for c in range(2)] for r in range(2)]
        
    def sub_blocks(B1, B2):
        return [[sub_forms(B1[r][c], B2[r][c]) for c in range(2)] for r in range(2)]
        
    # Outer Combinations
    # M0: (A00+A11)(B00+B11)
    L0 = add_blocks(A00, A11)
    R0 = add_blocks(B00, B11)
    
    # M1: (A10+A11)B00
    L1 = add_blocks(A10, A11)
    R1 = B00
    
    # M2: A00(B01-B11)
    L2 = A00
    R2 = sub_blocks(B01, B11)
    
    # M3: A11(B10-B00)
    L3 = A11
    R3 = sub_blocks(B10, B00)
    
    # M4: (A00+A01)B11
    L4 = add_blocks(A00, A01)
    R4 = B11
    
    # M5: (A10-A00)(B00+B01)
    L5 = sub_blocks(A10, A00)
    R5 = add_blocks(B00, B01)
    
    # M6: (A01-A11)(B10+B11)
    L6 = sub_blocks(A01, A11)
    R6 = add_blocks(B10, B11)
    
    outer_combos = [
        (L0, R0),
        (L1, R1),
        (L2, R2),
        (L3, R3),
        (L4, R4),
        (L5, R5),
        (L6, R6)
    ]
    
    # Inner Strassen for 2x2 product
    # Input: L (2x2 forms), R (2x2 forms)
    # Output: 7 pairs of (left_form, right_form)
    # and their contributions to C (2x2 matrix of forms)
    
    def inner_strassen(L, R):
        l00, l01 = L[0][0], L[0][1]
        l10, l11 = L[1][0], L[1][1]
        
        r00, r01 = R[0][0], R[0][1]
        r10, r11 = R[1][0], R[1][1]
        
        # Inner combinations
        # m0: (l00+l11)(r00+r11)
        ml0 = add_forms(l00, l11)
        mr0 = add_forms(r00, r11)
        
        # m1: (l10+l11)r00
        ml1 = add_forms(l10, l11)
        mr1 = r00
        
        # m2: l00(r01-r11)
        ml2 = l00
        mr2 = sub_forms(r01, r11)
        
        # m3: l11(r10-r00)
        ml3 = l11
        mr3 = sub_forms(r10, r00)
        
        # m4: (l00+l01)r11
        ml4 = add_forms(l00, l01)
        mr4 = r11
        
        # m5: (l10-l00)(r00+r01)
        ml5 = sub_forms(l10, l00)
        mr5 = add_forms(r00, r01)
        
        # m6: (l01-l11)(r10+r11)
        ml6 = sub_forms(l01, l11)
        mr6 = add_forms(r10, r11)
        
        products = [
            (ml0, mr0),
            (ml1, mr1),
            (ml2, mr2),
            (ml3, mr3),
            (ml4, mr4),
            (ml5, mr5),
            (ml6, mr6)
        ]
        
        # Contributions to C (2x2)
        # C00 = m0 + m3 - m5 + m6
        # C01 = m2 + m4
        # C10 = m1 + m3
        # C11 = m0 - m1 + m2 + m4
        
        c00 = [(0, 1), (3, 1), (5, -1), (6, 1)]
        c01 = [(2, 1), (4, 1)]
        c10 = [(1, 1), (3, 1)]
        c11 = [(0, 1), (1, -1), (2, 1), (4, 1)]
        
        contribs = [
            c00, c01, c10, c11
        ]
        
        return products, contribs

    # Accumulate into U, V, W
    for t in range(7):
        L, R = outer_combos[t]
        products, contribs = inner_strassen(L, R)
        
        # We need to map the inner 2x2 result back to the outer 4x4 C.
        # The outer Strassen result M_t is a 2x2 matrix of 2x2 blocks.
        # M_t[0,0] contributes to C[0:2, 0:2]
        # M_t[0,1] contributes to C[0:2, 2:4]
        # M_t[1,0] contributes to C[2:4, 0:2]
        # M_t[1,1] contributes to C[2:4, 2:4]
        
        # The inner_strassen returns products for the 2x2 matrix M_t.
        # The 'contribs' list corresponds to the 4 positions of the 2x2 M_t.
        # contribs[0] is for M_t[0,0]
        # contribs[1] is for M_t[0,1]
        # contribs[2] is for M_t[1,0]
        # contribs[3] is for M_t[1,1]
        
        for k in range(7):
            rank_idx = t * 7 + k
            l_form, r_form = products[k]
            
            for idx, coeff in l_form:
                U[idx, rank_idx] += coeff
            for idx, coeff in r_form:
                V[idx, rank_idx] += coeff
                
            # Determine which blocks of C this inner product contributes to
            # and with what coefficient.
            # The inner product k contributes to the 4 positions of M_t according to 'contribs'.
            # Wait, 'contribs' is a list of 4 lists.
            # contribs[pos] is a list of (inner_product_idx, coeff).
            # We are iterating over inner_product_idx = k.
            # So we need to check if k is in contribs[pos].
            
            # Actually, it's easier to iterate over the 4 positions of M_t and see which inner products contribute.
            # But we are already in the loop for k.
            # Let's pre-compute a map: for each inner product k, what are its contributions to M_t's 4 blocks?
            
            # Let's restructure:
            # For each t:
            #   Get products, contribs
            #   For each k in 0..6:
            #       Get l_form, r_form
            #       Add to U, V
            #       For each pos in 0..3 (blocks of M_t):
            #           Check if k is in contribs[pos]
            #           If so, get coeff c
            #           Map pos to global C block (row_block, col_block)
            #           Map inner product's contribution to C block to specific W entries
            #           Wait, the inner product is a scalar. It contributes to the *entire* 2x2 block?
            #           No.
            #           M_t is a 2x2 matrix. Each entry M_t[r,c] is a 2x2 matrix.
            #           The inner Strassen computes the product L*R which is a 2x2 matrix.
            #           The result of the inner Strassen is the 2x2 matrix M_t.
            #           So, for a fixed t, the 7 inner products form the 2x2 matrix M_t.
            #           The 'contribs' from inner_strassen tell us how the 7 inner products combine to form the 4 entries of the 2x2 matrix M_t.
            #           Entry (r,c) of M_t is a 2x2 matrix.
            #           The value at (r,c) is a linear combination of the 7 inner products.
            #           Each inner product is a scalar (product of two linear forms).
            #           So, the entry M_t[r,c] is a 2x2 matrix where each element is a linear combination of the 7 inner products.
            #           This means each of the 4 elements of M_t[r,c] gets contributions from the 7 inner products.
            
            # Let's look at inner_strassen again.
            # It returns products (7 pairs of forms) and contribs (4 lists of (prod_idx, coeff)).
            # contribs[r*2+c] is the list of (prod_idx, coeff) for the element at (r,c) of the 2x2 result matrix.
            # So, for a specific element at global C position (R, C), which is inside block (r,c) of M_t,
            # and inside position (u,v) of the 2x2 block,
            # the contribution from inner product k is:
            #   if k is in contribs[r*2+c], with coeff c_k, then
            #   W[idx, rank_idx] += c_k
            #   where idx is the W index for C[R,C].
            
            # Wait, the inner product k is a scalar.
            # The element M_t[r,c] is a 2x2 matrix.
            # The inner Strassen computes the 2x2 matrix product.
            # The result is a 2x2 matrix of scalars (linear forms).
            # So, M_t[r,c] is a 2x2 matrix.
            # Each element of M_t[r,c] is a linear combination of the 7 inner products.
            # So, for each element (u,v) of the 2x2 block M_t[r,c],
            # we need to add the contribution of inner product k to the corresponding W entry.
            
            # The W entry corresponds to C[R,C] where R = 2*r + u, C = 2*c + v.
            
            # So, for each k:
            #   For each pos (0..3) in contribs:
            #       If k is in contribs[pos]:
            #           Let r = pos // 2, c = pos % 2
            #           Let coeff = c_k
            #           For u in 0..1:
            #               For v in 0..1:
            #                   R = 2*r + u
            #                   C = 2*c + v
            #                   w_idx_val = w_idx(R, C)
            #                   W[w_idx_val, rank_idx] += coeff
            #                   
            # This logic seems correct.
            
            for pos in range(4):
                # Check if k contributes to this position of the 2x2 block M_t
                contrib_list = contribs[pos]
                found = False
                c_k = 0
                for idx_prod, coeff in contrib_list:
                    if idx_prod == k:
                        found = True
                        c_k = coeff
                        break
                if found:
                    r = pos // 2
                    c = pos % 2
                    for u in range(2):
                        for v in range(2):
                            R = 2 * r + u
                            C = 2 * c + v
                            w_idx_val = w_idx(R, C)
                            W[w_idx_val, rank_idx] += c_k

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
