"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28  # 4 blocks * 7 Strassen terms

    # We will build the tensor as a list of (u_idx, v_idx, w_idx, coeff) tuples
    # u_idx is the flat index into A (n*m = 8)
    # v_idx is the flat index into B (m*p = 16)
    # w_idx is the flat index into C (p*n = 8)
    # The trace convention is C[r, i] = sum_j A[i, j] B[j, r]
    # Flat index for A: i * m + j
    # Flat index for B: j * p + r
    # Flat index for C: r * n + i  (as per task description W[k*n+i,r] -> index is r*n + i? No, W is p*n. Let's re-read.)
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # Usually W is a matrix of size (p*n) x rank? Or just the tensor coefficients?
    # The return interface is: 'W': factor matrices of shapes (p*n, rank)
    # So W is a 2D array where row index is the flat index of C, and col index is the term index.
    # The flat index of C needs to correspond to the (r, i) pair.
    # If W has shape (p*n, rank), then W[flat_c, k] is the coefficient for C element flat_c in term k.
    # The task says W[k*n+i, r]... wait, W is a matrix (p*n, rank). The indices in the text "W[k*n+i,r]" likely refer to the logical index in the tensor, or perhaps the row index of W.
    # Let's assume the row index of W is the flat index of the C element.
    # How is the flat index of C defined?
    # Standard C order: C[i, r] where i is row (0..n-1), r is col (0..p-1).
    # Flat index: i * p + r.
    # BUT the task says "W[k*n+i,r]". This is confusing.
    # Let's look at the standard trace convention: C = A @ B.
    # C[i, r] = sum_j A[i, j] B[j, r].
    # The tensor T is T[i, j, r] = A[i, j] B[j, r].
    # The decomposition is T = sum_k u_k[i] v_k[j] w_k[r].
    # U has shape (n*m, rank). Row index is flat index of (i, j).
    # V has shape (m*p, rank). Row index is flat index of (j, r).
    # W has shape (p*n, rank). Row index is flat index of (r, i)? Or (i, r)?
    # If W is (p*n, rank), the row index is 0..p*n-1.
    # If the convention is C[i, r], the flat index is i*p + r. The size is n*p.
    # Here n=2, p=4, so size is 8.
    # The text says "W[k*n+i,r]". If k is the row index of W, and the value is associated with r and i...
    # Maybe the flat index of C is r * n + i?
    # Let's check the previous program. It used:
    # get_global_C returns r * 2 + i.
    # So the row index of W is r * n + i.
    # This means the flat index for W is not the standard C row-major, but rather a "column-major" of C if we view C as having r as the major index?
    # Actually, if C is n x p, and we flatten it by column (r first), the index is r * n + i.
    # Let's stick to the previous program's convention which presumably worked (score 1.1429 implies it was valid).
    # So:
    # U index: i * m + j
    # V index: j * p + r
    # W index: r * n + i

    terms = []

    # Strassen terms for 2x2 matrices
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A @ B
    #
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    #
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6

    # We define each term as a list of (a_idx, b_idx, c_idx, coeff)
    # where a_idx is local index in 2x2 A (0,1,2,3), etc.
    # Local index mapping:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)

    def local_to_ij(idx):
        return (idx // 2, idx % 2)

    # Term 0: m1
    # U: a00(1), a11(1)
    # V: b00(1), b11(1)
    # W: c00(1), c11(1)
    terms.append({
        'U': {0: 1, 3: 1},
        'V': {0: 1, 3: 1},
        'W': {0: 1, 3: 1}
    })

    # Term 1: m2
    # U: a10(1), a11(1)
    # V: b00(1)
    # W: c10(1), c11(-1)
    terms.append({
        'U': {2: 1, 3: 1},
        'V': {0: 1},
        'W': {2: 1, 3: -1}
    })

    # Term 2: m3
    # U: a00(1)
    # V: b01(1), b11(-1)
    # W: c01(1), c11(1)
    terms.append({
        'U': {0: 1},
        'V': {1: 1, 3: -1},
        'W': {1: 1, 3: 1}
    })

    # Term 3: m4
    # U: a11(1)
    # V: b10(1), b00(-1)
    # W: c00(1), c10(1)
    terms.append({
        'U': {3: 1},
        'V': {2: 1, 0: -1},
        'W': {0: 1, 2: 1}
    })

    # Term 4: m5
    # U: a00(1), a01(1)
    # V: b11(1)
    # W: c00(-1), c01(1)
    terms.append({
        'U': {0: 1, 1: 1},
        'V': {3: 1},
        'W': {0: -1, 1: 1}
    })

    # Term 5: m6
    # U: a10(1), a00(-1)
    # V: b00(1), b01(1)
    # W: c11(1)
    terms.append({
        'U': {2: 1, 0: -1},
        'V': {0: 1, 1: 1},
        'W': {3: 1}
    })

    # Term 6: m7
    # U: a01(1), a11(-1)
    # V: b10(1), b11(1)
    # W: c00(1)
    terms.append({
        'U': {1: 1, 3: -1},
        'V': {2: 1, 3: 1},
        'W': {0: 1}
    })

    # Blocks:
    # A is 2x4. Split columns into L (0,1) and R (2,3).
    # B is 4x4. Split rows into L (0,1) and R (2,3). Split cols into L (0,1) and R (2,3).
    # C is 2x4. Split cols into L (0,1) and R (2,3).

    # Block 0: A_L (cols 0,1) * B_LL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # Block 1: A_R (cols 2,3) * B_RL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # Block 2: A_L (cols 0,1) * B_LR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # Block 3: A_R (cols 2,3) * B_RR (rows 2,3, cols 2,3) -> C_R (cols 2,3)

    # Global Index Calculations:
    # U index (A): i * 4 + j
    # V index (B): j * 4 + r
    # W index (C): r * 2 + i  (Note: r is col of C, i is row of C. n=2, p=4)

    # We will iterate over blocks and terms.
    # For each block, we map local indices to global indices.

    def get_global_indices(block, local_a, local_b, local_c):
        # local_a, local_b, local_c are 0..3
        i_a, j_a = local_to_ij(local_a)
        j_b, r_b = local_to_ij(local_b)
        i_c, r_c = local_to_ij(local_c)

        # Adjust for block
        if block == 0:
            # A: i, j in 0,1
            g_a = i_a * 4 + j_a
            # B: j in 0,1, r in 0,1
            g_b = j_b * 4 + r_b
            # C: i in 0,1, r in 0,1
            g_c = r_c * 2 + i_c
        elif block == 1:
            # A: i, j in 2,3 (j offset 2)
            g_a = i_a * 4 + (j_a + 2)
            # B: j in 2,3 (j offset 2), r in 0,1
            g_b = (j_b + 2) * 4 + r_b
            # C: i in 0,1, r in 0,1
            g_c = r_c * 2 + i_c
        elif block == 2:
            # A: i, j in 0,1
            g_a = i_a * 4 + j_a
            # B: j in 0,1, r in 2,3 (r offset 2)
            g_b = j_b * 4 + (r_b + 2)
            # C: i in 0,1, r in 2,3 (r offset 2)
            g_c = (r_c + 2) * 2 + i_c
        elif block == 3:
            # A: i, j in 2,3
            g_a = i_a * 4 + (j_a + 2)
            # B: j in 2,3, r in 2,3
            g_b = (j_b + 2) * 4 + (r_b + 2)
            # C: i in 0,1, r in 2,3
            g_c = (r_c + 2) * 2 + i_c
        else:
            raise ValueError("Invalid block")

        return g_a, g_b, g_c

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            term = terms[s]
            for la, ca in term['U'].items():
                for lb, cb in term['V'].items():
                    for lc, cc in term['W'].items():
                        ga, gb, gc = get_global_indices(b, la, lb, lc)
                        # The coefficient for this term in the tensor is ca * cb * cc
                        coeff = ca * cb * cc
                        # We are building U, V, W such that sum_k U[i,k] V[j,k] W[r,k] = T[i,j,r]
                        # This requires that the term contributes to U, V, W consistently.
                        # In the previous program, it added to U, V, W separately.
                        # But here, a term is defined by a product of sums.
                        # We need to expand the product.
                        # The term s is a scalar multiplication of (sum ca * A[la]) * (sum cb * B[lb]) * (sum cc * C[lc])?
                        # No, the Strassen decomposition is:
                        # C = sum_s (A_term_s * B_term_s) * C_coeff_s
                        # Where A_term_s is a linear combination of A elements.
                        # So, for a fixed block b and term s:
                        # Contribution to T is sum_la sum_lb sum_lc ca*cb*cc * A[ga] * B[gb] * W[gc]?
                        # No, the tensor is T[ga, gb, gc] = A[ga] * B[gb] (if it's the identity)
                        # The decomposition is T = sum_k U_k A * V_k B * W_k C
                        # So for a specific k (block b, term s):
                        # U[ga, k] should be ca
                        # V[gb, k] should be cb
                        # W[gc, k] should be cc
                        # And the sum over k of U[ga,k] V[gb,k] W[gc,k] should equal T[ga,gb,gc].
                        #
                        # Let's check if this works for Strassen.
                        # For a 2x2 block:
                        # T[i,j,r] = A[i,j] B[j,r]
                        # Sum_s U_s[i,j] V_s[j,r] W_s[i,r]
                        #
                        # In the previous program, it did:
                        # U[ga, col] += ca
                        # V[gb, col] += cb
                        # W[gc, col] += cc
                        #
                        # This is correct.
                        
                        # However, we must be careful. The term 'U' dict contains {local_idx: coeff}.
                        # This means U[ga, col] += ca is only correct if there is a 1:1 mapping.
                        # Yes, for a fixed term s, the contribution to U is just the coefficient of the local element.
                        
                        # So we simply add the coefficients.
                        
                        # Wait, in the loop above, I am iterating over la, lb, lc.
                        # This is wrong. The term s defines:
                        # U_part = sum_la ca_la * A[la]
                        # V_part = sum_lb cb_lb * B[lb]
                        # W_part = sum_lc cc_lc * C[lc]
                        # The term value is U_part * V_part.
                        # The contribution to C is W_part * (U_part * V_part)?
                        # No, C = sum_s (U_part_s * V_part_s) * W_coeff_s?
                        # No, the Strassen formulas are:
                        # c00 = m1 + m4 - m5 + m7
                        # So for term m1, it contributes +1 to c00 and +1 to c11.
                        # This means W[local_c, col] is the coefficient.
                        
                        # So, for each block b and term s:
                        # For each local_a, local_b, local_c:
                        # We add ca to U[ga, col]
                        # We add cb to V[gb, col]
                        # We add cc to W[gc, col]
                        
                        # My nested loop is doing this, but it is inefficient and logically slightly off if interpreted as "for every combination".
                        # Actually, the addition is independent.
                        # U[ga, col] += ca (for each la in term['U'])
                        # V[gb, col] += cb (for each lb in term['V'])
                        # W[gc, col] += cc (for each lc in term['W'])
                        
                        # The nested loop is unnecessary and wrong because it would multiply the contributions?
                        # No, it's just adding.
                        # But it's cleaner to do it separately.

                        pass # We will rewrite the loop structure

    # Rewriting the loop structure for clarity and correctness
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            term = terms[s]
            
            for la, ca in term['U'].items():
                # Get global index for A
                i_a, j_a = local_to_ij(la)
                if b == 0:
                    ga = i_a * 4 + j_a
                elif b == 1:
                    ga = i_a * 4 + (j_a + 2)
                elif b == 2:
                    ga = i_a * 4 + j_a
                elif b == 3:
                    ga = i_a * 4 + (j_a + 2)
                U[ga][col] += ca

            for lb, cb in term['V'].items():
                # Get global index for B
                j_b, r_b = local_to_ij(lb)
                if b == 0:
                    gb = j_b * 4 + r_b
                elif b == 1:
                    gb = (j_b + 2) * 4 + r_b
                elif b == 2:
                    gb = j_b * 4 + (r_b + 2)
                elif b == 3:
                    gb = (j_b + 2) * 4 + (r_b + 2)
                V[gb][col] += cb

            for lc, cc in term['W'].items():
                # Get global index for C
                i_c, r_c = local_to_ij(lc)
                if b == 0:
                    gc = r_c * 2 + i_c
                elif b == 1:
                    gc = r_c * 2 + i_c
                elif b == 2:
                    gc = (r_c + 2) * 2 + i_c
                elif b == 3:
                    gc = (r_c + 2) * 2 + i_c
                W[gc][col] += cc

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
