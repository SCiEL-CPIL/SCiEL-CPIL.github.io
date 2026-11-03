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

    # Strategy: 3x3 blocks of 2x2 matrices
    # 9 block multiplications, each of rank 7 (Strassen 2x2)
    # Total rank = 9 * 7 = 63
    
    bs = 2
    nb = n // bs  # 3 blocks
    
    rank_new = nb * nb * nb * 7  # 9 * 7 = 63
    
    U_new = np.zeros((n * m, rank_new), dtype=float)
    V_new = np.zeros((m * p, rank_new), dtype=float)
    W_new = np.zeros((p * n, rank_new), dtype=float)
    
    # Strassen 2x2 definitions
    # Blocks indexed: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
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
    
    # Reconstruction for 2x2 product C = AB
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_defs[s]
                    
                    # U: A block (i_b, k_b)
                    for block_idx, coeff_a in s_def['A']:
                        ar = block_idx // 2
                        ac = block_idx % 2
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U_new[i_glob * m + k_glob, r_idx] += coeff_a
                        
                    # V: B block (k_b, j_b)
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V_new[k_glob * p + j_glob, r_idx] += coeff_b
                        
                    # W: C block (i_b, j_b)
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            cr = c_block_idx // 2
                            cc = c_block_idx % 2
                            i_glob = i_b * bs + cr
                            j_glob = j_b * bs + cc
                            W_new[j_glob * n + i_glob, r_idx] += gamma
                            
                    r_idx += 1
                    
    assert r_idx == rank_new, f"r_idx {r_idx} != rank_new {rank_new}"
    
    # Verify
    T_recon_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
    if np.allclose(T_recon_new, T, atol=1e-9):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    else:
        # Fallback to trivial
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
