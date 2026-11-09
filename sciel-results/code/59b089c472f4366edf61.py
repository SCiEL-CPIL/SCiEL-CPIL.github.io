"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    # rank will be 6 blocks * 7 (Strassen for 2x2) + 3 blocks * 4 (simple for 1x2) = 42 + 12 = 54
    # Wait: 3 column pairs (k pairs), each has 2 full 2x2 blocks and 1 half block (width 1)
    # k pairs: (0,1), (2,3), (4,5)
    # For each k pair:
    #   - A columns 0,1 x B rows 0,1 -> Strassen 7 terms
    #   - A columns 2,3 x B rows 2,3 -> Strassen 7 terms  
    #   - A column 4 x B row 4 -> 4 simple terms
    # Total per k pair: 7 + 7 + 4 = 18
    # Total rank: 3 * 18 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]  # shape (10, 54)
    V = [[0] * rank for _ in range(m * p)]  # shape (30, 54)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 54)
    
    col = 0
    
    for kp in range(3):
        k1 = kp * 2
        k2 = kp * 2 + 1
        
        # Strassen for A[:, 0:2] * B[0:2, k1:k2]
        # A rows: i in {0,1}, j in {0,1}
        # B rows: j in {0,1}, k in {k1, k2}
        # C: k in {k1,k2}, i in {0,1}
        #
        # Strassen for 2x2 * 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # But we need half-integer coefficients, so let's use the standard Strassen with integer coeffs.
        # Actually the standard Strassen uses ±1 only, which is fine.
        #
        # Let me re-index: A subblock is A[i,j] for i in {0,1}, j in {0,1} (local)
        # Global A row: i*5 + j
        # B subblock is B[j,k] for j in {0,1}, k in {k1,k2}
        # Global B row: j*6 + k
        # C subblock is C[k,i] for k in {k1,k2}, i in {0,1}
        # Global C row: k*2 + i
        
        def add_strassen(k_start):
            """Add 7 Strassen terms for 2x2 A-block (cols 0-1) * 2x2 B-block (rows 0-1, cols k_start..k_start+1)"""
            nonlocal col
            k_s = k_start
            k_e = k_start + 1
            
            # M1 = (A[0,0]+A[1,1]) * (B[0,k_s]+B[1,k_e])
            # We need to express this as sum of rank-1 terms with coefficients.
            # M1 contributes to: C[k_s,0], C[k_e,1], and -C[k_e,0], +C[k_s,1] etc.
            # This is getting complex. Let me think differently.
            
            # Actually, let me just build each rank-1 term explicitly.
            # For Strassen, each M term is a product of two linear combinations.
            # We can expand each M into rank-1 terms with coefficients ±1.
            #
            # M1 = (A11 + A22)(B11 + B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
            # That's 4 rank-1 terms, not 1. So Strassen doesn't directly give 7 rank-1 terms
            # unless we allow the factors to be linear combinations in U and V.
            #
            # In tensor decomposition, each rank-1 term is U_c ⊗ V_c ⊗ W_c where
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            #
            # So for M1 = (A[0,0] + A[1,1]) * (B[0,k_s] + B[1,k_e]), we can have:
            # A single rank-1 term where:
            #   U has +1 at (0,0) and +1 at (1,1)
            #   V has +1 at (0,k_s) and +1 at (1,k_e)
            #   W is determined by how M1 contributes to C entries.
            #
            # Yes! Each M term IS a single rank-1 term with U and V having multiple non-zeros.
            #
            # Let me define the 7 M terms and their contributions to C:
            #
            # M1 = (A11+A22)*(B11+B22): U={A11:+1, A22:+1}, V={B11:+1, B22:+1}
            #   Contribution to C: +C11, -C12... wait let me redo.
            
            # Let me use the formula more carefully.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5  
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # where:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B11)
            # M4 = A22(B12-B22)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # Let's define local indices:
            # A local: (a_i, a_j) where a_i, a_j in {0,1}, global A row = a_i*5 + a_j
            # B local: (b_j, b_k) where b_j in {0,1}, b_k in {0,1} (b_k=0->k_s, b_k=1->k_e), global B row = b_j*6 + (k_s+b_k)
            # C local: (c_k, c_i) where c_k in {0,1}, c_i in {0,1}, global C row = (k_s+c_k)*2 + c_i
            
            def get_au(ai, aj):
                return ai * 5 + aj
            def get_vv(bj, bk):
                return bj * 6 + (k_s + bk)
            def get_ww(ck, ci):
                return (k_s + ck) * 2 + ci
            
            def add_term(u_dict, v_dict, w_dict):
                nonlocal col
                c = col
                for (idx, val) in u_dict.items():
                    U[idx][c] += val
                for (idx, val) in v_dict.items():
                    V[idx][c] += val
                for (idx, val) in w_dict.items():
                    W[idx][c] += val
                col += 1
            
            # M1 = (A11+A22)(B11+B22)
            u1 = {get_au(0,0): 1, get_au(1,1): 1}
            v1 = {get_vv(0,0): 1, get_vv(1,1): 1}
            # M1 contributes: +C11, +C22
            w1 = {get_ww(0,0): 1, get_ww(1,1): 1}
            add_term(u1, v1, w1)
            
            # M2 = (A21+A22)B11
            u2 = {get_au(1,0): 1, get_au(1,1): 1}
            v2 = {get_vv(0,0): 1}
            # M2 contributes: +C21, -C22
            w2 = {get_ww(1,0): 1, get_ww(1,1): -1}
            add_term(u2, v2, w2)
            
            # M3 = A11(B21-B11)
            u3 = {get_au(0,0): 1}
            v3 = {get_vv(1,0): 1, get_vv(0,0): -1}
            # M3 contributes: +C12, +C22
            w3 = {get_ww(0,1): 1, get_ww(1,1): 1}
            add_term(u3, v3, w3)
            
            # M4 = A22(B12-B22)
            u4 = {get_au(1,1): 1}
            v4 = {get_vv(0,1): 1, get_vv(1,1): -1}
            # M4 contributes: +C11, +C21
            w4 = {get_ww(0,0): 1, get_ww(1,0): 1}
            add_term(u4, v4, w4)
            
            # M5 = (A11+A12)B22
            u5 = {get_au(0,0): 1, get_au(0,1): 1}
            v5 = {get_vv(1,1): 1}
            # M5 contributes: -C11, +C12
            w5 = {get_ww(0,0): -1, get_ww(0,1): 1}
            add_term(u5, v5, w5)
            
            # M6 = (A21-A11)(B11+B12)
            u6 = {get_au(1,0): 1, get_au(0,0): -1}
            v6 = {get_vv(0,0): 1, get_vv(0,1): 1}
            # M6 contributes: +C22
            w6 = {get_ww(1,1): 1}
            add_term(u6, v6, w6)
            
            # M7 = (A12-A22)(B21+B22)
            u7 = {get_au(0,1): 1, get_au(1,1): -1}
            v7 = {get_vv(1,0): 1, get_vv(1,1): 1}
            # M7 contributes: +C11, -C22
            w7 = {get_ww(0,0): 1, get_ww(1,1): -1}
            add_term(u7, v7, w7)
        
        # Strassen for cols 0,1
        add_strassen(k1)
        # That used 7 terms, col advanced by 7
        
        # Strassen for cols 2,3
        def add_strassen2(k_start):
            nonlocal col
            k_s = k_start
            k_e = k_start + 1
            
            def get_au(ai, aj):
                return ai * 5 + (aj + 2)  # offset by 2 for cols 2,3
            def get_vv(bj, bk):
                return (bj + 2) * 6 + (k_s + bk)  # offset by 2 for rows 2,3
            def get_ww(ck, ci):
                return (k_s + ck) * 2 + ci
            
            def add_term(u_dict, v_dict, w_dict):
                nonlocal col
                c = col
                for (idx, val) in u_dict.items():
                    U[idx][c] += val
                for (idx, val) in v_dict.items():
                    V[idx][c] += val
                for (idx, val) in w_dict.items():
                    W[idx][c] += val
                col += 1
            
            u1 = {get_au(0,0): 1, get_au(1,1): 1}
            v1 = {get_vv(0,0): 1, get_vv(1,1): 1}
            w1 = {get_ww(0,0): 1, get_ww(1,1): 1}
            add_term(u1, v1, w1)
            
            u2 = {get_au(1,0): 1, get_au(1,1): 1}
            v2 = {get_vv(0,0): 1}
            w2 = {get_ww(1,0): 1, get_ww(1,1): -1}
            add_term(u2, v2, w2)
            
            u3 = {get_au(0,0): 1}
            v3 = {get_vv(1,0): 1, get_vv(0,0): -1}
            w3 = {get_ww(0,1): 1, get_ww(1,1): 1}
            add_term(u3, v3, w3)
            
            u4 = {get_au(1,1): 1}
            v4 = {get_vv(0,1): 1, get_vv(1,1): -1}
            w4 = {get_ww(0,0): 1, get_ww(1,0): 1}
            add_term(u4, v4, w4)
            
            u5 = {get_au(0,0): 1, get_au(0,1): 1}
            v5 = {get_vv(1,1): 1}
            w5 = {get_ww(0,0): -1, get_ww(0,1): 1}
            add_term(u5, v5, w5)
            
            u6 = {get_au(1,0): 1, get_au(0,0): -1}
            v6 = {get_vv(0,0): 1, get_vv(0,1): 1}
            w6 = {get_ww(1,1): 1}
            add_term(u6, v6, w6)
            
            u7 = {get_au(0,1): 1, get_au(1,1): -1}
            v7 = {get_vv(1,0): 1, get_vv(1,1): 1}
            w7 = {get_ww(0,0): 1, get_ww(1,1): -1}
            add_term(u7, v7, w7)
        
        add_strassen2(k1)
        
        # Simple terms for col 4, row 4
        # A[:,4] * B[4, k1:k2] gives C[k1:k2, :]
        # 4 simple rank-1 terms
        for ai in range(2):
            for bk in range(2):
                c = col
                U[ai * 5 + 4][c] = 1
                V[4 * 6 + (k1 + bk)][c] = 1
                W[(k1 + bk) * 2 + ai][c] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
