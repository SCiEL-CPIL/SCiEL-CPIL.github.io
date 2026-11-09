"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 63 decomposition: 9 blocks of 2x2, each using Strassen (rank 7)
    rank = 63
    
    # Initialize matrices with zeros
    # Using lists for speed and simplicity, or numpy? 
    # The seed uses lists. I will use lists to be safe with the interface, 
    # but numpy is faster for generation. 
    # Given 120s limit, list generation is trivial.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # We need to define the 7 rank-1 terms for 2x2.
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = AB.
    # The task convention for W is W[k*n+i], which is C[k, i].
    # Standard C[i, k] = sum_j A[i, j] B[j, k].
    # So C[k, i] = sum_j B[k, j] A[j, i].
    # This is (B A)^T? No.
    # C^T[k, i] = C[i, k] = sum_j A[i, j] B[j, k].
    # So the output is the transpose of the standard product?
    # Or the inputs are transposed?
    # Seed: U[i, j], V[j, k], W[k, i].
    # This sums to A[i, j] B[j, k] -> W[k, i].
    # So W[k, i] accumulates A[i, j] B[j, k].
    # This means W is C^T if C = AB.
    # Let's assume the tensor is defined by this contraction.
    
    # Strassen for C = AB (standard).
    # We need to adapt to W[k, i] = C[i, k].
    # So we compute standard C, then place values at W[k, i].
    
    # Strassen steps for C = AB:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For each step s, we have:
    # U_part: linear form in A
    # V_part: linear form in B
    # W_part: linear form in C (which maps to W indices)
    
    # Let's define the 7 terms as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: [u00, u01, u10, u11] for A[0,0], A[0,1], A[1,0], A[1,1]
    # v_coeffs: [v00, v01, v10, v11] for B[0,0], B[0,1], B[1,0], B[1,1]
    # w_coeffs: [w00, w01, w10, w11] for C[0,0], C[0,1], C[1,0], C[1,1]
    # Note: W index is k*n+i. For 2x2 block, local k, i.
    # W_local[k, i] corresponds to C[i, k].
    # So w_coeffs should be aligned with C[i, k].
    # w_coeffs[0] is for C[0,0] -> W[0,0]
    # w_coeffs[1] is for C[0,1] -> W[1,0]
    # w_coeffs[2] is for C[1,0] -> W[0,1]
    # w_coeffs[3] is for C[1,1] -> W[1,1]
    
    strassen_terms = []
    
    # Term 1: m1
    # A: a00 + a11 -> [1, 0, 0, 1]
    # B: b00 + b11 -> [1, 0, 0, 1]
    # C: C00 += 1, C11 += 1
    # w: C00(1), C01(0), C10(0), C11(1)
    strassen_terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # Term 2: m2
    # A: a10 + a11 -> [0, 0, 1, 1]
    # B: b00 -> [1, 0, 0, 0]
    # C: C10 += 1
    # w: C00(0), C01(0), C10(1), C11(0)
    strassen_terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    
    # Term 3: m3
    # A: a00 -> [1, 0, 0, 0]
    # B: b01 - b11 -> [0, 1, 0, -1]
    # C: C01 += 1
    # w: C00(0), C01(1), C10(0), C11(0)
    strassen_terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    
    # Term 4: m4
    # A: a11 -> [0, 0, 0, 1]
    # B: b10 - b00 -> [-1, 0, 1, 0]
    # C: C00 += 1, C10 += 1
    # w: C00(1), C01(0), C10(1), C11(0)
    strassen_terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    
    # Term 5: m5
    # A: a00 + a01 -> [1, 1, 0, 0]
    # B: b11 -> [0, 0, 0, 1]
    # C: C00 -= 1, C01 += 1
    # w: C00(-1), C01(1), C10(0), C11(0)
    strassen_terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    # Term 6: m6
    # A: a10 - a00 -> [-1, 0, 1, 0]
    # B: b00 + b01 -> [1, 1, 0, 0]
    # C: C11 += 1
    # w: C00(0), C01(0), C10(0), C11(1)
    strassen_terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # Term 7: m7
    # A: a01 - a11 -> [0, 1, 0, -1]
    # B: b10 + b11 -> [0, 0, 1, 1]
    # C: C00 += 1
    # w: C00(1), C01(0), C10(0), C11(0)
    strassen_terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    # Now construct the full 6x6 decomposition
    # Block indices: alpha (row of A), gamma (col of A / row of B), beta (col of B)
    # alpha, gamma, beta in {0, 1, 2}
    # Local indices: i_loc, j_loc, k_loc in {0, 1}
    
    col_idx = 0
    for alpha in range(3):
        for gamma in range(3):
            for beta in range(3):
                # This block multiplication contributes to C_block[alpha, beta]
                # Using Strassen for this specific A_block[alpha, gamma] * B_block[gamma, beta]
                
                # For each of the 7 Strassen terms
                for s in range(7):
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
                    
                    # Assign to U
                    # U index: i * 6 + j
                    # i = 2 * alpha + i_loc
                    # j = 2 * gamma + j_loc
                    for i_loc in range(2):
                        for j_loc in range(2):
                            coeff = u_coeffs[i_loc * 2 + j_loc]
                            if coeff != 0:
                                i = 2 * alpha + i_loc
                                j = 2 * gamma + j_loc
                                U[i * 6 + j][col_idx] += coeff
                                
                    # Assign to V
                    # V index: j * 6 + k
                    # j = 2 * gamma + j_loc
                    # k = 2 * beta + k_loc
                    for j_loc in range(2):
                        for k_loc in range(2):
                            coeff = v_coeffs[j_loc * 2 + k_loc]
                            if coeff != 0:
                                j = 2 * gamma + j_loc
                                k = 2 * beta + k_loc
                                V[j * 6 + k][col_idx] += coeff
                                
                    # Assign to W
                    # W index: k * 6 + i
                    # k = 2 * beta + k_loc
                    # i = 2 * alpha + i_loc
                    # w_coeffs are for C[i_loc, k_loc]
                    # So we need to map C[i_loc, k_loc] to W[k, i]
                    # W[k, i] corresponds to C[i, k].
                    # So W_local[k_loc, i_loc] gets w_coeffs[i_loc * 2 + k_loc]
                    for i_loc in range(2):
                        for k_loc in range(2):
                            coeff = w_coeffs[i_loc * 2 + k_loc]
                            if coeff != 0:
                                k = 2 * beta + k_loc
                                i = 2 * alpha + i_loc
                                W[k * 6 + i][col_idx] += coeff
                                
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
