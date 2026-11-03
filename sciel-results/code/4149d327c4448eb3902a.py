import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    
    # Block-Strassen: 2x2 blocks of size 3x3
    # Outer Strassen: 7 multiplications of 3x3 blocks
    # Inner schoolbook 3x3: rank 27
    # Total rank = 7 * 27 = 189
    
    bs = 3
    rank = 7 * 27
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen definitions: M_s = A_s * B_s
    # Blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
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
    
    # Correct Strassen reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M5 + M6
    recon_coeffs = [
        # C00: M0(+1), M3(+1), M4(-1), M6(+1)
        {0: 1.0, 3: 1.0, 4: -1.0, 6: 1.0},
        # C01: M2(+1), M4(+1)
        {2: 1.0, 4: 1.0},
        # C10: M1(+1), M5(+1)
        {1: 1.0, 5: 1.0},
        # C11: M0(+1), M1(-1), M5(+1), M6(+1)
        {0: 1.0, 1: -1.0, 5: 1.0, 6: 1.0},
    ]
    
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        # Inner schoolbook: for each (u,v,j) in 3x3x3
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # U: A_s[u,v] = sum_{(block, coeff)} coeff * A_block[u,v]
                    # A_block(br,bc): row i = br*bs + u, col k = bc*bs + v
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        U[i_glob * m + k_glob, r_idx] += coeff_a
                    
                    # V: B_s[v,j] = sum_{(block, coeff)} coeff * B_block[v,j]
                    # B_block(br,bc): row k = br*bs + v, col j = bc*bs + j
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        V[k_glob * p + j_glob, r_idx] += coeff_b
                    
                    # W: M_s[u,j] contributes to C blocks via recon_coeffs
                    # C_block(br,bc): row i = br*bs + u, col j = bc*bs + j
                    # W index: j*n + i
                    for c_block_idx in range(4):
                        gamma = recon_coeffs[c_block_idx].get(s, 0.0)
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            W[j_glob * n + i_glob, r_idx] += gamma
                    
                    r_idx += 1
    
    assert r_idx == rank, f"rank mismatch: {r_idx} != {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
