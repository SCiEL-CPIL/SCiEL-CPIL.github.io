import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 105  # safety margin before 120s limit
    
    # Precompute Strassen decomposition for 2x2x2 (rank 7)
    # Returns U, V, W as lists of (row, col, val) where val is Fraction
    def strassen_222():
        U = []
        V = []
        W = []
        # 7 rank-1 terms
        # Term 0: (a11+a22)(b21-b11) -> contributes to c21, c22
        # Term 1: (a11+a22)(b11+b22) -> contributes to c11, c22
        # Term 2: (a12-a22)b21 -> contributes to c21
        # Term 3: a11(b22-b12) -> contributes to c12
        # Term 4: (a11+a12)b12 -> contributes to c12
        # Term 5: (a21-a11)(b11+b12) -> contributes to c21
        # Term 6: (a12+a22)(b21+b22) -> contributes to c22
        
        # Let's use the standard Strassen formulas:
        # m1 = (A11+A22)(B11+B22)
        # m2 = (A21+A22)B11
        # m3 = A11(B21-B22)
        # m4 = A22(B12-B11)
        # m5 = (A11+A12)B22
        # m6 = (A21-A11)(B11+B12)
        # m7 = (A12-A22)(B21+B22)
        #
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # U factors (A side):
        # m1: A11+A22 -> (0,0,1), (1,1,1)
        # m2: A21+A22 -> (1,0,1), (1,1,1)
        # m3: A11 -> (0,0,1)
        # m4: A22 -> (1,1,1)
        # m5: A11+A12 -> (0,0,1), (0,1,1)
        # m6: A21-A11 -> (1,0,1), (0,0,-1)
        # m7: A12-A22 -> (0,1,1), (1,1,-1)
        
        U_entries = [
            [(0,0,1), (1,1,1)],  # m1
            [(1,0,1), (1,1,1)],  # m2
            [(0,0,1)],           # m3
            [(1,1,1)],           # m4
            [(0,0,1), (0,1,1)],  # m5
            [(1,0,1), (0,0,-1)], # m6
            [(0,1,1), (1,1,-1)]  # m7
        ]
        
        # V factors (B side):
        # m1: B11+B22 -> (0,0,1), (1,1,1)
        # m2: B11 -> (0,0,1)
        # m3: B21-B22 -> (1,0,1), (1,1,-1)
        # m4: B12-B11 -> (0,1,1), (0,0,-1)
        # m5: B22 -> (1,1,1)
        # m6: B11+B12 -> (0,0,1), (0,1,1)
        # m7: B21+B22 -> (1,0,1), (1,1,1)
        
        V_entries = [
            [(0,0,1), (1,1,1)],  # m1
            [(0,0,1)],           # m2
            [(1,0,1), (1,1,-1)], # m3
            [(0,1,1), (0,0,-1)], # m4
            [(1,1,1)],           # m5
            [(0,0,1), (0,1,1)],  # m6
            [(1,0,1), (1,1,1)]   # m7
        ]
        
        # W factors (C side): C[i,j] gets contribution from m_k with coefficient w
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        W_entries = [
            [(0,0,1), (1,1,1)],  # m1: C11 += m1, C22 += m1
            [(1,0,1)],           # m2: C21 += m2
            [(0,1,1)],           # m3: C12 += m3
            [(0,0,1), (1,0,1)],  # m4: C11 += m4, C21 += m4
            [(0,0,-1), (0,1,1)], # m5: C11 -= m5, C12 += m5
            [(1,1,1)],           # m6: C22 += m6
            [(0,0,1), (1,1,-1)]  # m7: C11 += m7, C22 -= m7
        ]
        
        return U_entries, V_entries, W_entries
    
    # Build full decomposition given splits
    def build_decomposition(n_splits, m_splits, p_splits):
        n_off = [0]
        for s in n_splits[:-1]:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in m_splits[:-1]:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in p_splits[:-1]:
            p_off.append(p_off[-1] + s)
        
        U_rows = n * m
        V_rows = m * p
        W_rows = p * n
        
        U_list = []
        V_list = []
        W_list = []
        rank = 0
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]
                    
                    if nb == 2 and mb == 2 and pb == 2:
                        U_entries, V_entries, W_entries = strassen_222()
                        for term_idx in range(7):
                            col = rank
                            for li, lj, val in U_entries[term_idx]:
                                gi = i_off + li
                                gj = j_off + lj
                                U_list.append((gi * m + gj, col, Fraction(val)))
                            for lj, lk, val in V_entries[term_idx]:
                                gj = j_off + lj
                                gk = k_off + lk
                                V_list.append((gj * p + gk, col, Fraction(val)))
                            for lk, li, val in W_entries[term_idx]:
                                gk = k_off + lk
                                gi = i_off + li
                                W_list.append((gk * n + gi, Fraction(val)))
                            rank += 1
                    else:
                        for li in range(nb):
                            for lj in range(mb):
                                for lk in range(pb):
                                    col = rank
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U_list.append((gi * m + gj, col, Fraction(1)))
                                    V_list.append((gj * p + gk, col, Fraction(1)))
                                    W_list.append((gk * n + gi, Fraction(1)))
                                    rank += 1
        
        # Convert to matrices
        U = [[Fraction(0) for _ in range(rank)] for _ in range(U_rows)]
        V = [[Fraction(0) for _ in range(rank)] for _ in range(V_rows)]
        W = [[Fraction(0) for _ in range(rank)] for _ in range(W_rows)]
        
        for idx, col, val in U_list:
            U[idx][col] += val
        for idx, col, val in V_list:
            V[idx][col] += val
        for idx, col, val in W_list:
            W[idx][col] += val
        
        return {"U": U, "V": V, "W": W}
    
    def compute_rank(n_splits, m_splits, p_splits):
        rank = 0
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    if nb == 2 and mb == 2 and pb == 2:
                        rank += 7
                    else:
                        rank += nb * mb * pb
        return rank
    
    def get_compositions(d, max_part=2):
        if d == 0:
            yield []
            return
        for first in range(1, min(max_part, d) + 1):
            for rest in get_compositions(d - first, max_part):
                yield [first] + list(rest)
    
    # Generate all compositions
    n_comps = list(get_compositions(n))
    m_comps = list(get_compositions(m))
    p_comps = list(get_compositions(p))
    
    best_rank = n * m * p
    best_splits = (n_comps[0], m_comps[0], p_comps[0])
    
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                r = compute_rank(nc, mc, pc)
                if r < best_rank:
                    best_rank = r
                    best_splits = (nc, mc, pc)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    n_splits, m_splits, p_splits = best_splits
    return build_decomposition(n_splits, m_splits, p_splits)
# EVOLVE-BLOCK-END
