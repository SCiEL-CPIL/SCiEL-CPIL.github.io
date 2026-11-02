import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6

    # Block-Strassen decomposition: split into 2x2 blocks of size 3x3.
    # Outer Strassen gives 7 multiplications of 3x3 blocks.
    # Inner schoolbook for 3x3 gives rank 27.
    # Total rank = 7 * 27 = 189.
    bs = 3
    rank = 7 * 27

    # Initialize matrices with zeros
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]

    # Strassen definitions for 2x2 blocks
    # Blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M_s = A_s * B_s
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

    # Reconstruction coefficients for C blocks
    # C_block = sum_s gamma_s * M_s
    recon_coeffs = [
        # C00
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        # C01
        [(2, 1.0), (4, 1.0)],
        # C10
        [(1, 1.0), (3, 1.0)],
        # C11
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        # Inner schoolbook rank-27 decomposition for 3x3 * 3x3
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # Update U
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row][r_idx] += coeff_a

                    # Update V
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row][r_idx] += coeff_b

                    # Update W
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            # W[k*n + i, r]
                            w_row = j_glob * n + i_glob
                            W[w_row][r_idx] += gamma

                    r_idx += 1

    assert r_idx == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
