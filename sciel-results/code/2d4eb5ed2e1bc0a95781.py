"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 65: 49 for 4x4x4 (Strassen) + 16 for 4x4x1 (Schoolbook)
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_dict: { (i,j): coeff }
    # v_dict: { (j,k): coeff }
    # w_dict: { (k,i): coeff }
    def add_term(r, u_dict, v_dict, w_dict):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][r] += coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][r] += coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i][r] += coeff

    # Part 1: 4x4x4 using Strassen
    # Strassen 2x2 formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Inner Strassen for 2x2 matrices P, Q
    # u1 = P11+P22, v1 = Q11+Q22
    # u2 = P21+P22, v2 = Q11
    # u3 = P11, v3 = Q12-Q22
    # u4 = P22, v4 = Q21-Q11
    # u5 = P11+P12, v5 = Q22
    # u6 = P21-P11, v6 = Q11+Q12
    # u7 = P12-P22, v7 = Q21+Q22
    
    # Result R = P Q
    # R11 = M1 - M4 + M5 - M7
    # R12 = M3 + M7
    # R21 = M2 + M4
    # R22 = M1 - M3 + M6 - M7
    
    # Outer Strassen for 4x4 (blocks 00, 01, 10, 11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # We iterate over the 7 outer terms
    # For each, we define P and Q as linear combos of A and B blocks
    # Then we iterate over the 7 inner terms
    # For each inner term, we define u and v as linear combos of P and Q entries
    # Then we expand u and v into A and B entries
    # And determine the contribution to C
    
    # Indices for blocks: 0, 1
    # Block (r,c) covers rows 2r..2r+1, cols 2c..2c+1
    
    # Outer terms definition
    # k, P_coeffs (block -> coeff), Q_coeffs (block -> coeff)
    outer_terms = [
        # M1
        (0, { (0,0): 1, (1,1): 1 }, { (0,0): 1, (1,1): 1 }),
        # M2
        (1, { (1,0): 1, (1,1): 1 }, { (0,0): 1 }),
        # M3
        (2, { (0,0): 1 }, { (0,1): 1, (1,1): -1 }),
        # M4
        (3, { (1,1): 1 }, { (1,0): 1, (0,0): -1 }),
        # M5
        (4, { (0,0): 1, (0,1): 1 }, { (1,1): 1 }),
        # M6
        (5, { (1,0): 1, (0,0): -1 }, { (0,0): 1, (0,1): 1 }),
        # M7
        (6, { (0,1): 1, (1,1): -1 }, { (1,0): 1, (1,1): 1 }),
    ]
    
    # Inner terms definition
    # l, u_coeffs (entry in P -> coeff), v_coeffs (entry in Q -> coeff)
    # P entries: (0,0), (0,1), (1,0), (1,1)
    inner_terms = [
        # u1
        (0, { (0,0): 1, (1,1): 1 }, { (0,0): 1, (1,1): 1 }),
        # u2
        (1, { (1,0): 1, (1,1): 1 }, { (0,0): 1 }),
        # u3
        (2, { (0,0): 1 }, { (0,1): 1, (1,1): -1 }),
        # u4
        (3, { (1,1): 1 }, { (1,0): 1, (0,0): -1 }),
        # u5
        (4, { (0,0): 1, (0,1): 1 }, { (1,1): 1 }),
        # u6
        (5, { (1,0): 1, (0,0): -1 }, { (0,0): 1, (0,1): 1 }),
        # u7
        (6, { (0,1): 1, (1,1): -1 }, { (1,0): 1, (1,1): 1 }),
    ]
    
    # Outer Strassen result coefficients
    # C_block -> { M_k -> coeff }
    outer_result_coeffs = {
        (0,0): { 0: 1, 3: -1, 4: 1, 6: -1 },
        (0,1): { 2: 1, 6: 1 },
        (1,0): { 1: 1, 3: 1 },
        (1,1): { 0: 1, 2: -1, 5: 1, 6: -1 },
    }
    
    # Inner Strassen result coefficients
    # R_entry -> { u_l -> coeff }
    inner_result_coeffs = {
        (0,0): { 0: 1, 3: -1, 4: 1, 6: -1 },
        (0,1): { 2: 1, 6: 1 },
        (1,0): { 1: 1, 3: 1 },
        (1,1): { 0: 1, 2: -1, 5: 1, 6: -1 },
    }
    
    term_idx = 0
    for k, p_coeffs, q_coeffs in outer_terms:
        for l, u_coeffs, v_coeffs in inner_terms:
            # Expand u and v into A and B
            u_dict = {}
            v_dict = {}
            
            # Expand u
            for (pr, pc), u_c in u_coeffs.items():
                # P is sum of A blocks
                for (br, bc), p_c in p_coeffs.items():
                    # A block (br, bc)
                    # P entry (pr, pc) comes from A block (br, bc) entry (pr, pc)
                    # Global A index: i = 2*br + pr, j = 2*bc + pc
                    i = 2 * br + pr
                    j = 2 * bc + pc
                    coeff = u_c * p_c
                    u_dict[(i, j)] = u_dict.get((i, j), 0) + coeff
            
            # Expand v
            for (qr, qc), v_c in v_coeffs.items():
                # Q is sum of B blocks
                for (br, bc), q_c in q_coeffs.items():
                    # B block (br, bc)
                    # Q entry (qr, qc) comes from B block (br, bc) entry (qr, qc)
                    # Global B index: j = 2*br + qr, k = 2*bc + qc
                    j = 2 * br + qr
                    k = 2 * bc + qc
                    coeff = v_c * q_c
                    v_dict[(j, k)] = v_dict.get((j, k), 0) + coeff
            
            # Determine W contributions
            w_dict = {}
            # This term (k, l) contributes to M_k
            # M_k contributes to C blocks
            for (cb_r, cb_c), outer_c in outer_result_coeffs.items():
                if k in outer_c:
                    # M_k contributes to C block (cb_r, cb_c)
                    # The inner result R_entry (ir, ic) contributes to C entry
                    # C global index: i = 2*cb_r + ir, k = 2*cb_c + ic
                    for (ir, ic), inner_c in inner_result_coeffs.items():
                        if l in inner_c:
                            i = 2 * cb_r + ir
                            k_global = 2 * cb_c + ic
                            coeff = outer_c[k] * inner_c[l]
                            w_dict[(k_global, i)] = w_dict.get((k_global, i), 0) + coeff
            
            add_term(term_idx, u_dict, v_dict, w_dict)
            term_idx += 1
    
    # Part 2: 4x4x1 using Schoolbook
    # C[:, 4] = A * B[:, 4]
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            u_dict = { (i, j): 1 }
            v_dict = { (j, 4): 1 }
            w_dict = { (4, i): 1 }
            add_term(term_idx, u_dict, v_dict, w_dict)
            term_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
