import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    deadline = time.time() + 140
    
    # Block size
    bs = 3
    
    # Rank is 7 (Strassen steps) * 27 (schoolbook rank of 3x3)
    rank = 7 * 27
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen definitions
    # A_s = sum coeff * A_block
    # B_s = sum coeff * B_block
    # Blocks are indexed 0:00, 1:01, 2:10, 3:11
    # Block (br, bc) has global row br*bs, col bc*bs
    
    strassen_defs = [
        # M0: (A00 + A11)(B00 + B11)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M1: (A10 + A11)B00
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M2: A00(B01 - B11)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M3: A11(B10 - B00)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M4: (A00 + A01)B11
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M5: (A10 - A00)(B00 + B01)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M6: (A01 - A11)(B10 + B11)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    # Reconstruction coefficients: C_block = sum gamma_s * M_s
    # C_block indices: 0:00, 1:01, 2:10, 3:11
    # M_s indices: 0..6
    recon_coeffs = [
        # C00 = M0 + M3 - M4 + M6
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        # C01 = M2 + M4
        [(2, 1.0), (4, 1.0)],
        # C10 = M1 + M3
        [(1, 1.0), (3, 1.0)],
        # C11 = M0 - M1 + M2 + M5
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        # Iterate over local indices for 3x3 multiplication
        # u: row of A_s, v: col of A_s / row of B_s, j: col of B_s
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # This corresponds to the term A_s[u, v] * B_s[v, j]
                    # which contributes to M_s[u, j]
                    
                    # 1. Update U
                    # A_s[u, v] = sum_{(block, coeff)} coeff * A_block[u, v]
                    # A_block (br, bc) has global indices:
                    # row i = br*bs + u
                    # col k = bc*bs + v
                    # U index a = i*m + k
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row][r_idx] += coeff_a
                    
                    # 2. Update V
                    # B_s[v, j] = sum_{(block, coeff)} coeff * B_block[v, j]
                    # B_block (br, bc) has global indices:
                    # row k = br*bs + v
                    # col j = bc*bs + j
                    # V index b = k*p + j
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row][r_idx] += coeff_b
                    
                    # 3. Update W
                    # M_s[u, j] contributes to C_block (br, bc) with coefficient gamma
                    # C_block (br, bc) has global indices:
                    # row i = br*bs + u
                    # col j = bc*bs + j
                    # W index c = j*n + i (based on W[k*n+i] with k=j, i=i)
                    for c_block_idx, gamma in recon_coeffs:
                        if c_block_idx == s:
                            # Wait, recon_coeffs is indexed by C_block (0..3)
                            # We need to check which C_blocks receive contribution from M_s
                            pass
                    
                    # Correct W update logic:
                    # Iterate over C_blocks (0..3)
                    for c_block_idx in range(4):
                        coeffs = recon_coeffs[c_block_idx]
                        # Find if M_s is in coeffs
                        gamma = 0.0
                        for (s_idx, g) in coeffs:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            # W index c = j_glob * n + i_glob
                            w_row = j_glob * n + i_glob
                            W[w_row][r_idx] += gamma
                    
                    r_idx += 1
    
    assert r_idx == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
