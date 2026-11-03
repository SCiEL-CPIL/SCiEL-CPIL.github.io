import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 110.0

    # Target tensor T[u, v, w] where u=i*m+k, v=k*p+j, w=j*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_triv = np.zeros((n * m, rank_trivial), dtype=float)
    V_triv = np.zeros((m * p, rank_trivial), dtype=float)
    W_triv = np.zeros((p * n, rank_trivial), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_triv[i * m + k, idx] = 1.0
                V_triv[k * p + j, idx] = 1.0
                W_triv[j * n + i, idx] = 1.0
                idx += 1

    # Block-Strassen approach with block size 2 (3x3 blocks of 2x2)
    # 27 block positions, each using Strassen (rank 7) = 189 total
    bs = 2  # block size 2, so 3x3 blocks
    nb = n // bs  # 3 blocks

    # Strassen 2x2 definitions (7 multiplications)
    # A is 2x2, B is 2x2
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B00)
    # M3 = A11(B01-B10)
    # M4 = (A00+A10)(B00+B01)
    # M5 = (A10+A11)(B10+B11)
    # M6 = (A00-A10)(B01-B11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For each Strassen term s (0..6), define the coefficients for A and B entries
    # A entries: 00, 01, 10, 11 -> indices 0, 1, 2, 3 in our 2x2 indexing
    # B entries: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    
    # strassen_defs[s]['A'] is list of (block_idx, coeff)
    # strassen_defs[s]['B'] is list of (block_idx, coeff)
    # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
    
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},  # M0
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},            # M1
        {'A': [(0, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},           # M2
        {'A': [(3, 1.0)], 'B': [(1, 1.0), (2, -1.0)]},           # M3
        {'A': [(0, 1.0), (2, 1.0)], 'B': [(0, 1.0), (1, 1.0)]},  # M4
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(2, 1.0), (3, 1.0)]},  # M5
        {'A': [(0, 1.0), (2, -1.0)], 'B': [(1, 1.0), (3, -1.0)]} # M6
    ]
    
    # Reconstruction coefficients for C blocks
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],  # C00
        [(2, 1.0), (4, 1.0)],                        # C01
        [(1, 1.0), (3, 1.0)],                        # C10
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)]    # C11
    ]

    rank_bs = 189
    U2 = np.zeros((n * m, rank_bs), dtype=float)
    V2 = np.zeros((m * p, rank_bs), dtype=float)
    W2 = np.zeros((p * n, rank_bs), dtype=float)

    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_defs[s]
                    # U
                    for block_idx, coeff_a in s_def['A']:
                        ar = block_idx // 2
                        ac = block_idx % 2
                        i_glob = i_b * 2 + ar
                        k_glob = k_b * 2 + ac
                        U2[i_glob * m + k_glob, r_idx] += coeff_a
                    # V
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = k_b * 2 + br
                        j_glob = j_b * 2 + bc
                        V2[k_glob * p + j_glob, r_idx] += coeff_b
                    # W: reconstruction
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            cr = c_block_idx // 2
                            cc = c_block_idx % 2
                            i_glob = i_b * 2 + cr
                            j_glob = j_b * 2 + cc
                            W2[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1

    # Verify
    T_recon2 = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
    if np.allclose(T_recon2, T, atol=1e-9):
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    else:
        # Fallback to trivial
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
