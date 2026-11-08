import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i # W[k*n+i, r] corresponds to C[i, k]
    
    # Helper to add/subtract linear forms (dicts)
    def add_forms(f1, f2):
        d = {}
        for idx, c in f1.items():
            d[idx] = d.get(idx, 0) + c
        for idx, c in f2.items():
            d[idx] = d.get(idx, 0) + c
        return {k: v for k, v in d.items() if v != 0}
        
    def sub_forms(f1, f2):
        d = {}
        for idx, c in f1.items():
            d[idx] = d.get(idx, 0) + c
        for idx, c in f2.items():
            d[idx] = d.get(idx, 0) - c
        return {k: v for k, v in d.items() if v != 0}

    # Initialize base forms for A and B
    # A indices: 0..15
    # B indices: 0..15
    A_forms = {i: {i: 1} for i in range(16)}
    B_forms = {i: {i: 1} for i in range(16)}
    
    # Outer Strassen Combinations
    # A blocks:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B blocks:
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # We need to form the 7 pairs of 2x2 matrices L_t, R_t.
    # Each entry is a linear form (dict).
    
    def get_A_block(r, c):
        # r, c in 0..1
        base = r * 8 + c * 2
        return [
            [A_forms[base + 0], A_forms[base + 1]],
            [A_forms[base + 4], A_forms[base + 5]]
        ]
        
    def get_B_block(r, c):
        base = r * 8 + c * 2
        return [
            [B_forms[base + 0], B_forms[base + 1]],
            [B_forms[base + 4], B_forms[base + 5]]
        ]

    A00 = get_A_block(0, 0)
    A01 = get_A_block(0, 1)
    A10 = get_A_block(1, 0)
    A11 = get_A_block(1, 1)
    
    B00 = get_B_block(0, 0)
    B01 = get_B_block(0, 1)
    B10 = get_B_block(1, 0)
    B11 = get_B_block(1, 1)

    def add_blocks(B1, B2):
        return [[add_forms(B1[r][c], B2[r][c]) for c in range(2)] for r in range(2)]
        
    def sub_blocks(B1, B2):
        return [[sub_forms(B1[r][c], B2[r][c]) for c in range(2)] for r in range(2)]

    # Outer Combinations
    L0 = add_blocks(A00, A11)
    R0 = add_blocks(B00, B11)
    
    L1 = add_blocks(A10, A11)
    R1 = B00
    
    L2 = A00
    R2 = sub_blocks(B01, B11)
    
    L3 = A11
    R3 = sub_blocks(B10, B00)
    
    L4 = add_blocks(A00, A01)
    R4 = B11
    
    L5 = sub_blocks(A10, A00)
    R5 = add_blocks(B00, B01)
    
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
    
    # Inner Strassen Coefficients (for 2x2 matrix multiplication)
    # W_in[elem_idx][prod_idx]
    # elem_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # prod_idx: 0..6
    W_in = [
        [1, 0, 0, 1, 0, -1, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 1, 0, 0]  # C11
    ]
    
    # Outer Strassen Coefficients
    # W_out[block_idx][prod_idx]
    # block_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W_out = [
        [1, 0, 0, 1, 0, -1, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 1, 0, 0]  # C11
    ]
    
    # Function to apply inner Strassen to a pair of 2x2 form-matrices
    def inner_strassen_forms(L, R):
        l00, l01 = L[0][0], L[0][1]
        l10, l11 = L[1][0], L[1][1]
        
        r00, r01 = R[0][0], R[0][1]
        r10, r11 = R[1][0], R[1][1]
        
        # Inner combinations
        ml0 = add_forms(l00, l11)
        mr0 = add_forms(r00, r11)
        
        ml1 = add_forms(l10, l11)
        mr1 = r00
        
        ml2 = l00
        mr2 = sub_forms(r01, r11)
        
        ml3 = l11
        mr3 = sub_forms(r10, r00)
        
        ml4 = add_forms(l00, l01)
        mr4 = r11
        
        ml5 = sub_forms(l10, l00)
        mr5 = add_forms(r00, r01)
        
        ml6 = sub_forms(l01, l11)
        mr6 = add_forms(r10, r11)
        
        return [
            (ml0, mr0),
            (ml1, mr1),
            (ml2, mr2),
            (ml3, mr3),
            (ml4, mr4),
            (ml5, mr5),
            (ml6, mr6)
        ]
        
    # Main Loop
    for t in range(7):
        L_t, R_t = outer_combos[t]
        inner_products = inner_strassen_forms(L_t, R_t)
        
        for k in range(7):
            rank_idx = t * 7 + k
            l_form, r_form = inner_products[k]
            
            # Fill U and V
            for idx, coeff in l_form.items():
                U[idx, rank_idx] = coeff
            for idx, coeff in r_form.items():
                V[idx, rank_idx] = coeff
                
            # Fill W
            # W[k_global * n + i_global, rank_idx]
            for i_global in range(4):
                for k_global in range(4):
                    i_block = i_global // 2
                    u = i_global % 2
                    k_block = k_global // 2
                    v = k_global % 2
                    
                    block_idx = i_block * 2 + k_block
                    elem_idx = u * 2 + v
                    
                    coeff = W_out[block_idx][t] * W_in[elem_idx][k]
                    if coeff != 0:
                        W[k_global * n + i_global, rank_idx] = coeff
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
