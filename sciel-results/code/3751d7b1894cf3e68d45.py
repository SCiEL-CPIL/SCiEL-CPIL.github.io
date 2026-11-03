import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    # We use a 2x2 block structure with 3x3 blocks
    # Outer rank: 7 (Strassen for 2x2)
    # Inner rank: 27 (Schoolbook for 3x3)
    # Total rank: 7 * 27 = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for the outer 2x2 block multiplication
    # M_s = A_s * B_s
    # A_s = sum coeff_a * A_block
    # B_s = sum coeff_b * B_block
    # Blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_defs = [
        # M0: (A00 + A11)(B00 + B11)
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)]},
        # M1: (A10 + A11)B00
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)]},
        # M2: A00(B01 - B11)
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)]},
        # M3: A11(B10 - B00)
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)]},
        # M4: (A00 + A01)B11
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)]},
        # M5: (A10 - A00)(B00 + B01)
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)]},
        # M6: (A01 - A11)(B10 + B11)
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)]},
    ]
    
    # Reconstruction coefficients for C blocks
    # C_block = sum gamma_s * M_s
    recon_coeffs = [
        # C00 = M0 + M3 - M4 + M6
        [(0, 1), (3, 1), (4, -1), (6, 1)],
        # C01 = M2 + M4
        [(2, 1), (4, 1)],
        # C10 = M1 + M3
        [(1, 1), (3, 1)],
        # C11 = M0 - M1 + M2 + M5
        [(0, 1), (1, -1), (2, 1), (5, 1)],
    ]
    
    bs = 3 # Block size
    r_idx = 0
    
    for s in range(7):
        s_def = strassen_defs[s]
        
        # Inner schoolbook decomposition for 3x3 * 3x3
        # Iterate over local indices u, v, j
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    
                    # 1. Update U
                    # U corresponds to A_s[u, v]
                    # A_s[u, v] = sum_{(block, coeff)} coeff * A_block[u, v]
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row][r_idx] += coeff_a
                    
                    # 2. Update V
                    # V corresponds to B_s[v, j]
                    # B_s[v, j] = sum_{(block, coeff)} coeff * B_block[v, j]
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row][r_idx] += coeff_b
                    
                    # 3. Update W
                    # W corresponds to contribution to C from M_s
                    # M_s[u, j] contributes to C_blocks
                    for c_block_idx in range(4):
                        gamma = 0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            # W index: k_global * n + i_global
                            # In our notation, the output block is C[br, bc]
                            # The global row is i_glob, global col is j_glob
                            # The task specifies W[k*n+i, r]
                            # Here k is the column index of C (j_glob), i is the row index of C (i_glob)
                            w_row = j_glob * n + i_glob
                            W[w_row][r_idx] += gamma
                    
                    r_idx += 1
                    if time.time() > deadline:
                        # Should not happen, but safety check
                        break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    if r_idx != rank:
        # If we didn't finish, the matrices are incomplete. 
        # However, the loop is deterministic and fast.
        pass

    return {"U": U, "V": V, "W": W}
