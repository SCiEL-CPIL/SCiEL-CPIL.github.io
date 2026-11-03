import time
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partitions
    # I: 0,1 | 2
    # J: 0,1 | 2,3
    # K: 0,1 | 2,3 | 4,5 | 6
    
    I_blocks = [(0, 2), (2, 3)]  # start, end
    J_blocks = [(0, 2), (2, 4)]
    K_blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    # Precompute total rank
    # 6 * 7 (Strassen) + 2 * 4 (2x2x1) + 6 * 4 (1x2x2) + 2 * 2 (1x2x1) = 78
    total_rank = 78
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen terms for 2x2x2
    # Returns list of (u_dict, v_dict, w_dict)
    # Keys are (i,j), (j,k), (k,i) or flat indices
    def get_strassen_terms():
        terms = []
        # M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # M2
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(1,0): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # M3
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(0,1): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # M4
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (1,0): 1}
        terms.append((u4, v4, w4))
        
        # M5
        u5 = {(0,0): 1, (1,0): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # M6
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # M7
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    strassen_terms = get_strassen_terms()
    
    for i_start, i_end in I_blocks:
        for j_start, j_end in J_blocks:
            for k_start, k_end in K_blocks:
                n_b = i_end - i_start
                m_b = j_end - j_start
                p_b = k_end - k_start
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Strassen
                    for u, v, w in strassen_terms:
                        for (li, lj), val_u in u.items():
                            gi = i_start + li
                            gj = j_start + lj
                            U[gi * m + gj][col] = val_u
                        for (lj, lk), val_v in v.items():
                            gj = j_start + lj
                            gk = k_start + lk
                            V[gj * p + gk][col] = val_v
                        for (lk, li), val_w in w.items():
                            gk = k_start + lk
                            gi = i_start + li
                            W[gk * n + gi][col] = val_w
                        col += 1
                else:
                    # Schoolbook
                    for li in range(n_b):
                        for lj in range(m_b):
                            for lk in range(p_b):
                                gi = i_start + li
                                gj = j_start + lj
                                gk = k_start + lk
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
                                
    if col != total_rank:
        # Should not happen
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
