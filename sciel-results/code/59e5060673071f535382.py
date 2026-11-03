import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 98
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # a_coeffs: dict (i, j) -> coeff for A
    # b_coeffs: dict (j, k) -> coeff for B
    # c_coeffs: dict (i, k) -> coeff for C
    # offset: column index in U, V, W
    def add_term(a_coeffs, b_coeffs, c_coeffs, offset):
        for (i, j), val in a_coeffs.items():
            U[i * m + j][offset] += val
        for (j, k), val in b_coeffs.items():
            V[j * p + k][offset] += val
        for (i, k), val in c_coeffs.items():
            W[k * n + i][offset] += val

    # Base Strassen terms for 2x2
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    base_terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    base_terms.append({
        'a': {(0,0): 1, (1,1): 1},
        'b': {(0,0): 1, (1,1): 1},
        'c': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0} # Placeholder, need to map to final C
    })
    # Actually, the W coefficients depend on how this M1 is used in the final C.
    # In 2x2 Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each base term r, we have:
    # U_r, V_r are fixed.
    # W_r is a 2x2 matrix.
    
    # Let's define the 7 base terms with their U, V, and W (2x2) matrices.
    
    def get_base_strassen_terms():
        terms = []
        # M1
        terms.append({
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0} # This is wrong. W is the contribution to C.
        })
        # It's better to store the linear forms and the output coefficients separately.
        # Term r:
        # L_A = sum u_ij A_ij
        # L_B = sum v_jk B_jk
        # C_ik += w_ik * L_A * L_B
        
        # M1: L_A = A00 + A11, L_B = B00 + B11.
        # Contributes to C00 (+1), C11 (+1) ? No.
        # C00 = M1 ...
        # C11 = M1 ...
        # So w = {(0,0): 1, (1,1): 1}? No.
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to C00 and +1 to C11.
        
        t1 = {
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): 1}
        }
        t2 = {
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(1,0): 1, (1,1): -1} # C10 = M2 + M4, C11 = M1 - M2 ...
            # C10 gets +1 from M2. C11 gets -1 from M2.
        }
        t3 = {
            'u': {(0,0): 1},
            'v': {(0,1): 1, (1,1): -1},
            'w': {(0,1): 1, (1,1): 1} # C01 = M3 + M5, C11 = ... + M3
        }
        t4 = {
            'u': {(1,1): 1},
            'v': {(1,0): 1, (0,0): -1},
            'w': {(0,0): 1, (1,0): 1} # C00 = M1 + M4 ..., C10 = M2 + M4
        }
        t5 = {
            'u': {(0,0): 1, (0,1): 1},
            'v': {(1,1): 1},
            'w': {(0,0): -1, (0,1): 1} # C00 = ... - M5, C01 = M3 + M5
        }
        t6 = {
            'u': {(1,0): 1, (0,0): -1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(1,1): 1} # C11 = ... + M6
        }
        t7 = {
            'u': {(0,1): 1, (1,1): -1},
            'v': {(1,0): 1, (1,1): 1},
            'w': {(0,0): 1} # C00 = ... + M7
        }
        
        return [t1, t2, t3, t4, t5, t6, t7]

    base_terms = get_base_strassen_terms()
    
    # Now for 4x4.
    # We have 7 block terms.
    # Each block term involves linear combinations of 2x2 blocks.
    # We need to expand each block term into 7 scalar terms.
    
    # Block Term 1: (A00 + A11)(B00 + B11)
    # A_block = A00 + A11
    # B_block = B00 + B11
    # This block term contributes to C00, C01, C10, C11 blocks with coeffs:
    # C00 += 1 * M1_block
    # C01 += 0
    # C10 += 0
    # C11 += 1 * M1_block
    # Wait, in 4x4 Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So Block M1 contributes +1 to C00 and +1 to C11.
    
    # We need a function to compose a block term with base terms.
    
    def compose_block_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, base_terms, offset_start):
        # a_block_coeffs: dict (block_i, block_j) -> coeff
        # b_block_coeffs: dict (block_j, block_k) -> coeff
        # c_block_coeffs: dict (block_i, block_k) -> coeff
        
        # For each base term, we expand the block multiplication.
        # The block multiplication is:
        # L_A_block = sum a_ij A_ij (where A_ij is a 2x2 block)
        # L_B_block = sum b_jk B_jk (where B_jk is a 2x2 block)
        # Product is a 2x2 matrix of scalar products.
        
        # We iterate over the 7 base terms.
        # Each base term t has u_t, v_t, w_t for 2x2.
        # The scalar product is:
        # L_A_scalar = sum_{i,j} (sum_{bi,bj} a_{bi,bj} * u_t[bi,bj]) A_{i,j}
        # Wait, indices are tricky.
        
        # Let's map block indices to local indices.
        # Block (bi, bj) contains A[2*bi + i, 2*bj + j] for i,j in 0,1.
        
        # For a given base term t:
        # u_t[li, lj] is the coeff for A_local[li, lj] in L_A.
        # L_A = sum_{li,lj} u_t[li,lj] A_local[li,lj]
        # A_local[li,lj] = sum_{bi,bj} a_block[bi,bj] A[2*bi+li, 2*bj+lj]
        # So L_A = sum_{bi,bj,li,lj} a_block[bi,bj] u_t[li,lj] A[2*bi+li, 2*bj+lj]
        
        # Similarly for L_B.
        
        # The contribution to C_block is:
        # C_block[li, lj] += w_t[li, lj] * L_A * L_B
        # C_block[li, lj] is the 2x2 block of C.
        # C[2*ci+li, 2*ck+lj] += c_block[ci,ck] * w_t[li,lj] * L_A * L_B
        
        # So for each base term t, we generate one rank-1 term.
        # U_coeffs: for each (i, j), sum over bi, bj, li, lj such that i=2*bi+li, j=2*bj+lj
        # U[i, j] = sum a_block[bi,bj] * u_t[li,lj]
        
        # V_coeffs: for each (j, k), sum over bj, bk, lj, lk such that j=2*bj+lj, k=2*bk+lk
        # V[j, k] = sum b_block[bj,bk] * v_t[lj,lk]
        
        # W_coeffs: for each (i, k), sum over ci, ck, li, lk such that i=2*ci+li, k=2*ck+lk
        # W[i, k] = sum c_block[ci,ck] * w_t[li,lk]
        
        # We return the list of (U, V, W) dicts for the 7 base terms.
        
        generated_terms = []
        for t in base_terms:
            u_new = {}
            v_new = {}
            w_new = {}
            
            # Compute U
            for (bi, bj), a_val in a_block_coeffs.items():
                for li in range(2):
                    for lj in range(2):
                        u_val = t['u'].get((li, lj), 0)
                        if u_val != 0 and a_val != 0:
                            i = 2 * bi + li
                            j = 2 * bj + lj
                            u_new[(i, j)] = u_new.get((i, j), 0) + a_val * u_val
            
            # Compute V
            for (bj, bk), b_val in b_block_coeffs.items():
                for lj in range(2):
                    for lk in range(2):
                        v_val = t['v'].get((lj, lk), 0)
                        if v_val != 0 and b_val != 0:
                            j = 2 * bj + lj
                            k = 2 * bk + lk
                            v_new[(j, k)] = v_new.get((j, k), 0) + b_val * v_val
            
            # Compute W
            for (ci, ck), c_val in c_block_coeffs.items():
                for li in range(2):
                    for lk in range(2):
                        w_val = t['w'].get((li, lk), 0)
                        if w_val != 0 and c_val != 0:
                            i = 2 * ci + li
                            k = 2 * ck + lk
                            w_new[(i, k)] = w_new.get((i, k), 0) + c_val * w_val
            
            generated_terms.append({
                'u': u_new,
                'v': v_new,
                'w': w_new
            })
            
        return generated_terms

    # Define the 7 block terms for 4x4
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    block_terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    block_terms.append({
        'a': {(0,0): 1, (1,1): 1},
        'b': {(0,0): 1, (1,1): 1},
        'c': {(0,0): 1, (1,1): 1}
    })
    # M2 = (A10 + A11)B00
    block_terms.append({
        'a': {(1,0): 1, (1,1): 1},
        'b': {(0,0): 1},
        'c': {(1,0): 1, (1,1): -1}
    })
    # M3 = A00(B01 - B11)
    block_terms.append({
        'a': {(0,0): 1},
        'b': {(0,1): 1, (1,1): -1},
        'c': {(0,1): 1, (1,1): 1}
    })
    # M4 = A11(B10 - B00)
    block_terms.append({
        'a': {(1,1): 1},
        'b': {(1,0): 1, (0,0): -1},
        'c': {(0,0): 1, (1,0): 1}
    })
    # M5 = (A00 + A01)B11
    block_terms.append({
        'a': {(0,0): 1, (0,1): 1},
        'b': {(1,1): 1},
        'c': {(0,0): -1, (0,1): 1}
    })
    # M6 = (A10 - A00)(B00 + B01)
    block_terms.append({
        'a': {(1,0): 1, (0,0): -1},
        'b': {(0,0): 1, (0,1): 1},
        'c': {(1,1): 1}
    })
    # M7 = (A01 - A11)(B10 + B11)
    block_terms.append({
        'a': {(0,1): 1, (1,1): -1},
        'b': {(1,0): 1, (1,1): 1},
        'c': {(0,0): 1}
    })
    
    # Generate 49 terms for first block (cols 0-3 of B)
    offset = 0
    for bt in block_terms:
        terms = compose_block_term(bt['a'], bt['b'], bt['c'], base_terms, offset)
        for t in terms:
            add_term(t['u'], t['v'], t['w'], offset)
            offset += 1
            
    # Generate 49 terms for second block (cols 4-7 of B)
    # A is the same.
    # B is shifted: B_j, k+4
    # C is shifted: C_i, k+4
    
    # We can reuse the same U coefficients.
    # V coefficients: shift k by 4.
    # W coefficients: shift k by 4.
    
    offset2 = 49
    for bt in block_terms:
        terms = compose_block_term(bt['a'], bt['b'], bt['c'], base_terms, offset2)
        for t in terms:
            # Shift V and W
            v_shifted = {}
            for (j, k), val in t['v'].items():
                v_shifted[(j, k+4)] = val
            w_shifted = {}
            for (i, k), val in t['w'].items():
                w_shifted[(i, k+4)] = val
            add_term(t['u'], v_shifted, w_shifted, offset2)
            offset2 += 1
            
    return {"U": U, "V": V, "W": W}
