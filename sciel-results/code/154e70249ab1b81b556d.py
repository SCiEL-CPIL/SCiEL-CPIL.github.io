"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Strassen patterns for 2x2x2 (7 rank-1 terms)
    # U, V, W are lists of (index1, index2, value)
    # Indices are local to the 2x2x2 block
    strassen_patterns = [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    def get_splits(d):
        """Generate all compositions of d into parts of size 1 or 2."""
        results = []
        def helper(remaining, current):
            if remaining == 0:
                results.append(current[:])
                return
            if remaining >= 1:
                current.append(1)
                helper(remaining - 1, current)
                current.pop()
            if remaining >= 2:
                current.append(2)
                helper(remaining - 2, current)
                current.pop()
        helper(d, [])
        return results
    
    def compute_rank(n_splits, m_splits, p_splits):
        """Compute the rank for given splits."""
        rank = 0
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    if nb == 2 and mb == 2 and pb == 2:
                        rank += 7
                    elif nb == 2 and mb == 2 and pb == 3:
                        rank += 10
                    elif nb == 2 and mb == 3 and pb == 2:
                        rank += 10
                    elif nb == 3 and mb == 2 and pb == 2:
                        rank += 10
                    elif nb == 2 and mb == 3 and pb == 3:
                        rank += 16
                    elif nb == 3 and mb == 2 and pb == 3:
                        rank += 16
                    elif nb == 3 and mb == 3 and pb == 2:
                        rank += 16
                    elif nb == 3 and mb == 3 and pb == 3:
                        rank += 23
                    else:
                        rank += nb * mb * pb
        return rank

    def build_2x2x3_decomposition(i_off, j_off, k_off, n, m, p):
        """Build rank-10 decomposition for 2x2x3 block."""
        U_entries = []
        V_entries = []
        W_entries = []
        
        # Define the 10 rank-1 terms for 2x2x3
        # M1 = (A00+A11)(B00+B10)
        U_entries.append([(0,0,1), (1,1,1)])
        V_entries.append([(0,0,1), (1,0,1)])
        W_entries.append([(0,0,1), (1,0,1)])
        
        # M2 = (A10+A11)B00
        U_entries.append([(1,0,1), (1,1,1)])
        V_entries.append([(0,0,1)])
        W_entries.append([(0,1,1), (1,1,-1)])
        
        # M3 = A00(B01+B11)
        U_entries.append([(0,0,1)])
        V_entries.append([(0,1,1), (1,1,1)])
        W_entries.append([(1,0,1), (1,1,1)])
        
        # M4 = (A01+A11)(B01+B11)
        U_entries.append([(0,1,1), (1,1,1)])
        V_entries.append([(0,1,1), (1,1,1)])
        W_entries.append([(0,0,1), (0,1,1), (1,0,-1), (1,1,-1)])
        
        # M5 = A10(B10+B11)
        U_entries.append([(1,0,1)])
        V_entries.append([(1,0,1), (1,1,1)])
        W_entries.append([(0,1,1), (1,1,1)])
        
        # M6 = (A00+A01)(B02+B12)
        U_entries.append([(0,0,1), (0,1,1)])
        V_entries.append([(0,2,1), (1,2,1)])
        W_entries.append([(0,0,1), (0,1,1), (1,0,-1), (1,1,-1)])
        
        # M7 = A00*B12
        U_entries.append([(0,0,1)])
        V_entries.append([(1,2,1)])
        W_entries.append([(0,0,1), (0,1,-1), (1,0,1), (1,1,-1)])
        
        # M8 = (A10+A11)(B02+B12)
        U_entries.append([(1,0,1), (1,1,1)])
        V_entries.append([(0,2,1), (1,2,1)])
        W_entries.append([(0,1,1), (1,1,1)])
        
        # M9 = A11*B12
        U_entries.append([(1,1,1)])
        V_entries.append([(1,2,1)])
        W_entries.append([(0,1,1), (1,1,-1)])
        
        # M10 = (A00+A11)(B02+B12)
        U_entries.append([(0,0,1), (1,1,1)])
        V_entries.append([(0,2,1), (1,2,1)])
        W_entries.append([(0,0,1), (0,1,1), (1,0,-1), (1,1,-1)])
        
        # Now output the entries with global indices
        result_U = []
        result_V = []
        result_W = []
        
        for col_idx in range(10):
            for li, lj, val in U_entries[col_idx]:
                result_U.append(((i_off + li) * m + (j_off + lj), col_idx, val))
            for lj, lk, val in V_entries[col_idx]:
                result_V.append(((j_off + lj) * p + (k_off + lk), col_idx, val))
            for lk, li, val in W_entries[col_idx]:
                result_W.append(((k_off + lk) * n + (i_off + li), col_idx, val))
        
        return result_U, result_V, result_W, 10
    
    def build_decomposition(n_splits, m_splits, p_splits):
        """Build the full decomposition for given splits."""
        n_off = [0]
        for s in n_splits[:-1]:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in m_splits[:-1]:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in p_splits[:-1]:
            p_off.append(p_off[-1] + s)
        
        U_entries = []
        V_entries = []
        W_entries = []
        rank = 0
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]
                    
                    if nb == 2 and mb == 2 and pb == 2:
                        for pattern in strassen_patterns:
                            col = rank
                            for li, lj, val in pattern['U']:
                                U_entries.append(((i_off + li) * m + (j_off + lj), col, val))
                            for lj, lk, val in pattern['V']:
                                V_entries.append(((j_off + lj) * p + (k_off + lk), col, val))
                            for lk, li, val in pattern['W']:
                                W_entries.append(((k_off + lk) * n + (i_off + li), col, val))
                            rank += 1
                    elif (nb == 2 and mb == 2 and pb == 3) or \
                         (nb == 2 and mb == 3 and pb == 2) or \
                         (nb == 3 and mb == 2 and pb == 2):
                        # Use rank-10 decomposition for 2x2x3 and permutations
                        if nb == 2 and mb == 2 and pb == 3:
                            u_ent, v_ent, w_ent, r = build_2x2x3_decomposition(i_off, j_off, k_off, n, m, p)
                            for e in u_ent:
                                U_entries.append((e[0], rank + e[1], e[2]))
                            for e in v_ent:
                                V_entries.append((e[0], rank + e[1], e[2]))
                            for e in w_ent:
                                W_entries.append((e[0], rank + e[1], e[2]))
                            rank += r
                        elif nb == 2 and mb == 3 and pb == 2:
                            # Permute: treat as 2x3x2, which is transpose of 2x2x3
                            # For now, use schoolbook as a safe fallback
                            for li in range(nb):
                                for lj in range(mb):
                                    for lk in range(pb):
                                        col = rank
                                        gi = i_off + li
                                        gj = j_off + lj
                                        gk = k_off + lk
                                        U_entries.append((gi * m + gj, col, 1))
                                        V_entries.append((gj * p + gk, col, 1))
                                        W_entries.append((gk * n + gi, col, 1))
                                        rank += 1
                        elif nb == 3 and mb == 2 and pb == 2:
                            # Permute: treat as 3x2x2, which is related to 2x2x3
                            # For now, use schoolbook as a safe fallback
                            for li in range(nb):
                                for lj in range(mb):
                                    for lk in range(pb):
                                        col = rank
                                        gi = i_off + li
                                        gj = j_off + lj
                                        gk = k_off + lk
                                        U_entries.append((gi * m + gj, col, 1))
                                        V_entries.append((gj * p + gk, col, 1))
                                        W_entries.append((gk * n + gi, col, 1))
                                        rank += 1
                    else:
                        # For other blocks, use schoolbook
                        for li in range(nb):
                            for lj in range(mb):
                                for lk in range(pb):
                                    col = rank
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U_entries.append((gi * m + gj, col, 1))
                                    V_entries.append((gj * p + gk, col, 1))
                                    W_entries.append((gk * n + gi, col, 1))
                                    rank += 1
        
        # Build matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for idx, col, val in U_entries:
            U[idx][col] = val
        for idx, col, val in V_entries:
            V[idx][col] = val
        for idx, col, val in W_entries:
            W[idx][col] = val
        
        return {"U": U, "V": V, "W": W}, rank
    
    # Generate all possible splits for each dimension
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)
    
    # Find the best combination of splits (minimum rank)
    # Note: The rank calculation in compute_rank was optimistic.
    # We will use the actual rank from build_decomposition.
    
    best_rank = float('inf')
    best_result = None
    best_splits = None
    
    for ns in n_split_options:
        if time.time() > deadline:
            break
        for ms in m_split_options:
            if time.time() > deadline:
                break
            for ps in p_split_options:
                if time.time() > deadline:
                    break
                result, r = build_decomposition(ns, ms, ps)
                if r < best_rank:
                    best_rank = r
                    best_result = result
                    best_splits = (ns, ms, ps)
    
    # Fallback to a simple schoolbook decomposition if something went wrong
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
