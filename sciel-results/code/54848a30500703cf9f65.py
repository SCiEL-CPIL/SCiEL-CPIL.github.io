import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Strassen rank-7 terms for 2x2x2 multiplication
    # Each term is (U_dict, V_dict, W_dict) where keys are (row, col)
    strassen_terms_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Top level terms for 4x4x4 using 2x2 blocks
    # The 4x4 matrix is divided into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # This is not a direct Strassen application on 4x4.
    # Standard recursive Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We need to construct the 4x4x4 tensor using these 7 products.
    # Each product is a 2x2x2 multiplication (or sum thereof).
    # Actually, each Mi is a 2x2x2 multiplication of specific linear combinations.
    # The rank of 4x4x4 is 7 * 7 = 49 if we use Strassen recursively.
    
    # Let's define the 7 terms for 4x4x4.
    # Each term corresponds to one of the 7 Strassen products.
    # Each Strassen product is a 2x2x2 multiplication.
    # The 2x2x2 multiplication has rank 7.
    # So we have 7 groups of 7 terms, total 49 terms.
    
    # Define the 2x2 blocks indices
    # Block (b_r, b_c) covers rows 2*b_r..2*b_r+1 and cols 2*b_c..2*b_c+1
    
    def add_to_dict(d, key, val):
        if key in d:
            d[key] += val
        else:
            d[key] = val

    # Strassen combinations for 4x4
    # M1: (A00+A11) (B00+B11)
    # M2: (A10+A11) B00
    # M3: A00 (B01-B11)
    # M4: A11 (B10-B00)
    # M5: (A00+A01) B11
    # M6: (A10-A00) (B00+B01)
    # M7: (A01-A11) (B10+B11)
    
    # For each M, we need the U, V, W factors for the 2x2 multiplication,
    # mapped to the 4x4 grid.
    
    # U_factor for M: dict of (i,j) -> coeff
    # V_factor for M: dict of (j,k) -> coeff
    # W_factor for M: dict of (i,k) -> coeff (this is the C block update)
    
    # Let's build the 49 terms.
    all_terms = []
    
    for s_idx in range(7):
        s_u, s_v, s_w = strassen_terms_2x2[s_idx]
        
        # Determine which blocks are involved in this Strassen term M_s
        # This is hardcoded based on the 7 formulas above.
        
        if s_idx == 0: # M1
            # A: A00, A11 with coeff 1
            # B: B00, B11 with coeff 1
            # C: C00 (+), C11 (+) ... wait, C update is M1 + M4 - M5 + M7 for C00, etc.
            # The term M1 contributes to C00, C01, C10, C11?
            # No, M1 is a 2x2 matrix product.
            # (A00+A11) (B00+B11) = P1 (2x2 matrix)
            # C00 += P1? No.
            # C00 = M1 + M4 - M5 + M7. So C00 gets M1.
            # C01 = M3 + M5. C01 does NOT get M1.
            # C10 = M2 + M4. C10 does NOT get M1.
            # C11 = M1 - M2 + M3 + M6. C11 gets M1.
            
            # So M1 (which is a 2x2 product) contributes its 4 entries to C00 and C11?
            # No. M1 is a 2x2 matrix $P_{ab}$.
            # $C_{00}$ is a 2x2 block. $C_{00} = P_1 + P_4 - P_5 + P_7$.
            # So the top-left 2x2 block of C gets $P_1$.
            # The bottom-right 2x2 block of C gets $P_1$.
            
            # The "product" $P_1$ is computed by $7$ scalar multiplications.
            # Each scalar multiplication corresponds to a rank-1 term in the 2x2x2 tensor.
            # So for each of the 7 Strassen terms in the 2x2 calculation of M1,
            # we create a term in the 4x4x4 tensor.
            
            # U factors: A00 and A11.
            # V factors: B00 and B11.
            # W factors: C00 and C11.
            
            u_blocks = [((0,0), 1), ((1,1), 1)] # (block_row, block_col), coeff
            v_blocks = [((0,0), 1), ((1,1), 1)]
            w_blocks = [((0,0), 1), ((1,1), 1)]
            
        elif s_idx == 1: # M2: (A10+A11) B00
            u_blocks = [((1,0), 1), ((1,1), 1)]
            v_blocks = [((0,0), 1)]
            w_blocks = [((1,0), 1)] # C10 gets M2
            
        elif s_idx == 2: # M3: A00 (B01-B11)
            u_blocks = [((0,0), 1)]
            v_blocks = [((0,1), 1), ((1,1), -1)]
            w_blocks = [((0,1), 1)] # C01 gets M3
            
        elif s_idx == 3: # M4: A11 (B10-B00)
            u_blocks = [((1,1), 1)]
            v_blocks = [((1,0), 1), ((0,0), -1)]
            w_blocks = [((0,0), 1), ((1,0), 1)] # C00 gets M4, C10 gets M4
            
        elif s_idx == 4: # M5: (A00+A01) B11
            u_blocks = [((0,0), 1), ((0,1), 1)]
            v_blocks = [((1,1), 1)]
            w_blocks = [((0,0), -1), ((0,1), 1)] # C00 gets -M5, C01 gets M5
            
        elif s_idx == 5: # M6: (A10-A00) (B00+B01)
            u_blocks = [((1,0), 1), ((0,0), -1)]
            v_blocks = [((0,0), 1), ((0,1), 1)]
            w_blocks = [((1,1), 1)] # C11 gets M6
            
        else: # s_idx == 6, M7: (A01-A11) (B10+B11)
            u_blocks = [((0,1), 1), ((1,1), -1)]
            v_blocks = [((1,0), 1), ((1,1), 1)]
            w_blocks = [((0,0), 1)] # C00 gets M7

        # Now, for this Strassen term M_s, we expand the 2x2x2 multiplication.
        # The 2x2x2 multiplication has 7 rank-1 terms (strassen_terms_2x2).
        # For each of these 7 terms, we create a global term.
        
        for t_idx in range(7):
            t_u, t_v, t_w = strassen_terms_2x2[t_idx]
            
            u_global = {}
            for (r, c), coeff_t in t_u.items():
                for (b_r, b_c), coeff_b in u_blocks:
                    i = 2*b_r + r
                    j = 2*b_c + c
                    add_to_dict(u_global, (i,j), coeff_t * coeff_b)
                    
            v_global = {}
            for (r, c), coeff_t in t_v.items():
                for (b_r, b_c), coeff_b in v_blocks:
                    j = 2*b_r + r
                    k = 2*b_c + c
                    add_to_dict(v_global, (j,k), coeff_t * coeff_b)
                    
            w_global = {}
            for (r, c), coeff_t in t_w.items():
                for (b_r, b_c), coeff_b in w_blocks:
                    i = 2*b_r + r
                    k = 2*b_c + c
                    add_to_dict(w_global, (i,k), coeff_t * coeff_b)
                    
            # Clean up zeros
            u_global = {k: v for k, v in u_global.items() if v != 0}
            v_global = {k: v for k, v in v_global.items() if v != 0}
            w_global = {k: v for k, v in w_global.items() if v != 0}
            
            if u_global and v_global and w_global:
                all_terms.append((u_global, v_global, w_global))
                
    # This gives 49 terms for the left half (k=0..3).
    # Now duplicate for the right half (k=4..7).
    terms_right = []
    for u_dict, v_dict, w_dict in all_terms:
        u_new = u_dict
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_right.append((u_new, v_new, w_new))
        
    total_terms = all_terms + terms_right
    rank = len(total_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(total_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col_idx] = coeff
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
