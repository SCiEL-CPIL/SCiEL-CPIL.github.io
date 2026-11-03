"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Strassen patterns for 2x2x2 (7 rank-1 terms)
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
    
    def build_2x3x2_decomposition(i_off, j_off, k_off, n, m, p):
        """
        Build rank-10 decomposition for 2x3x2 block.
        This is the transpose/permutation of 2x2x3.
        A is 2x3, B is 3x2, C is 2x2.
        """
        U_entries = []
        V_entries = []
        W_entries = []
        
        # We'll use a systematic approach:
        # Split the 3 dimension of A and B into 2+1
        # For the 2x2x2 part: use Strassen (rank 7)
        # For the 2x1x2 part: schoolbook (rank 2)
        # For the 1x2x2 part: schoolbook (rank 2)
        # Total: 7 + 2 + 2 = 11, but we can do better with overlap
        
        # Actually, let's just use schoolbook for simplicity first: 2*3*2 = 12
        # Then try to optimize
        
        # Schoolbook fallback:
        for li in range(2):
            for lj in range(3):
                for lk in range(2):
                    col = 0  # will be adjusted
                    U_entries.append((li, lj, 0, 1))  # (li, lj, col, val)
                    V_entries.append((lj, lk, 0, 1))
                    W_entries.append((lk, li, 0, 1))
        
        # Return as list of (row, col, val) for each matrix
        # U is 2*3 = 6 rows, V is 3*2 = 6 rows, W is 2*2 = 4 rows
        # Each has rank columns
        
        result_U = []
        result_V = []
        result_W = []
        col = 0
        for li in range(2):
            for lj in range(3):
                for lk in range(2):
                    result_U.append(((i_off + li) * m + (j_off + lj), col, 1))
                    result_V.append(((j_off + lj) * p + (k_off + lk), col, 1))
                    result_W.append(((k_off + lk) * n + (i_off + li), col, 1))
                    col += 1
        
        return result_U, result_V, result_W, 12
    
    def build_2x2x3_decomposition(i_off, j_off, k_off, n, m, p):
        """
        Build decomposition for 2x2x3 block.
        A is 2x2, B is 2x3, C is 2x3.
        """
        result_U = []
        result_V = []
        result_W = []
        col = 0
        # Schoolbook: 2*2*3 = 12
        for li in range(2):
            for lj in range(2):
                for lk in range(3):
                    result_U.append(((i_off + li) * m + (j_off + lj), col, 1))
                    result_V.append(((j_off + lj) * p + (k_off + lk), col, 1))
                    result_W.append(((k_off + lk) * n + (i_off + li), col, 1))
                    col += 1
        return result_U, result_V, result_W, 12
    
    def build_3x2x2_decomposition(i_off, j_off, k_off, n, m, p):
        """
        Build decomposition for 3x2x2 block.
        A is 3x2, B is 2x2, C is 3x2.
        """
        result_U = []
        result_V = []
        result_W = []
        col = 0
        # Schoolbook: 3*2*2 = 12
        for li in range(3):
            for lj in range(2):
                for lk in range(2):
                    result_U.append(((i_off + li) * m + (j_off + lj), col, 1))
                    result_V.append(((j_off + lj) * p + (k_off + lk), col, 1))
                    result_W.append(((k_off + lk) * n + (i_off + li), col, 1))
                    col += 1
        return result_U, result_V, result_W, 12
    
    def build_3x3x3_decomposition(i_off, j_off, k_off, n, m, p):
        """
        Build decomposition for 3x3x3 block using Laderman's algorithm (rank 23).
        A is 3x3, B is 3x3, C is 3x3.
        """
        # Laderman's algorithm for 3x3x3 matrix multiplication
        # This is a known rank-23 algorithm
        
        # Define the 23 rank-1 terms
        # Each term: (U_entries, V_entries, W_entries)
        # U is 3x3 (9 entries max), V is 3x3, W is 3x3 (indexed as k*n+i)
        
        laderman_terms = [
            # Term 1: (A00+A22)(B00+B22)
            {'U': [(0,0,1), (2,2,1)], 'V': [(0,0,1), (2,2,1)], 'W': [(0,0,1), (2,2,1)]},
            # Term 2: (A20+A22)B00
            {'U': [(2,0,1), (2,2,1)], 'V': [(0,0,1)], 'W': [(0,2,1), (2,2,-1)]},
            # Term 3: A00(B02-B22)
            {'U': [(0,0,1)], 'V': [(0,2,1), (2,2,-1)], 'W': [(2,0,1), (2,2,1)]},
            # Term 4: A22(B20-B00)
            {'U': [(2,2,1)], 'V': [(2,0,1), (0,0,-1)], 'W': [(0,0,1), (0,2,1)]},
            # Term 5: (A00+A02)B22
            {'U': [(0,0,1), (0,2,1)], 'V': [(2,2,1)], 'W': [(0,0,-1), (2,0,1)]},
            # Term 6: (A20-A00)(B00+B02)
            {'U': [(2,0,1), (0,0,-1)], 'V': [(0,0,1), (0,2,1)], 'W': [(2,2,1)]},
            # Term 7: (A02-A22)(B20+B22)
            {'U': [(0,2,1), (2,2,-1)], 'V': [(2,0,1), (2,2,1)], 'W': [(0,0,1)]},
            # Term 8: (A11+A33) -- wait, we only have 3x3, indices 0,1,2
            # Let me use a different known rank-23 algorithm
            
            # Actually, let me use a simpler approach: split 3 into 2+1
            # and use Strassen for the 2x2x2 part
        ]
        
        # Simpler approach: split 3 = 2+1 in each dimension
        # 2x2x2: Strassen rank 7
        # 2x2x1: schoolbook rank 4
        # 2x1x2: schoolbook rank 4
        # 1x2x2: schoolbook rank 4
        # 2x1x1: schoolbook rank 2
        # 1x2x1: schoolbook rank 2
        # 1x1x2: schoolbook rank 2
        # 1x1x1: schoolbook rank 1
        # Total: 7+4+4+4+2+2+2+1 = 26
        
        # Let's implement this split-based approach
        result_U = []
        result_V = []
        result_W = []
        col = 0
        
        # Splits: [2, 1] for each dimension
        n_splits = [2, 1]
        m_splits = [2, 1]
        p_splits = [2, 1]
        
        n_offsets = [0, 2]
        m_offsets = [0, 2]
        p_offsets = [0, 2]
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb = n_splits[i_b]
                    mb = m_splits[j_b]
                    pb = p_splits[k_b]
                    i_off_local = n_offsets[i_b]
                    j_off_local = m_offsets[j_b]
                    k_off_local = p_offsets[k_b]
                    
                    gi_off = i_off + i_off_local
                    gj_off = j_off + j_off_local
                    gk_off = k_off + k_off_local
                    
                    if nb == 2 and mb == 2 and pb == 2:
                        for pattern in strassen_patterns:
                            for li, lj, val in pattern['U']:
                                result_U.append(((gi_off + li) * m + (gj_off + lj), col, val))
                            for lj, lk, val in pattern['V']:
                                result_V.append(((gj_off + lj) * p + (gk_off + lk), col, val))
                            for lk, li, val in pattern['W']:
                                result_W.append(((gk_off + lk) * n + (gi_off + li), col, val))
                            col += 1
                    else:
                        for li in range(nb):
                            for lj in range(mb):
                                for lk in range(pb):
                                    result_U.append(((gi_off + li) * m + (gj_off + lj), col, 1))
                                    result_V.append(((gj_off + lj) * p + (gk_off + lk), col, 1))
                                    result_W.append(((gk_off + lk) * n + (gi_off + li), col, 1))
                                    col += 1
        
        return result_U, result_V, result_W, col
    
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
                    nb = n_splits[i_b]
                    mb = m_splits[j_b]
                    pb = p_splits[k_b]
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
                    elif (nb == 2 and mb == 3 and pb == 2) or \
                         (nb == 3 and mb == 2 and pb == 2) or \
                         (nb == 2 and mb == 2 and pb == 3):
                        # Use split-based decomposition
                        if nb == 2 and mb == 3 and pb == 2:
                            u_ent, v_ent, w_ent, r = build_2x3x2_decomposition(i_off, j_off, k_off, n, m, p)
                        elif nb == 3 and mb == 2 and pb == 2:
                            u_ent, v_ent, w_ent, r = build_3x2x2_decomposition(i_off, j_off, k_off, n, m, p)
                        else:  # nb == 2 and mb == 2 and pb == 3
                            u_ent, v_ent, w_ent, r = build_2x2x3_decomposition(i_off, j_off, k_off, n, m, p)
                        
                        for e in u_ent:
                            U_entries.append((e[0], rank + e[1], e[2]))
                        for e in v_ent:
                            V_entries.append((e[0], rank + e[1], e[2]))
                        for e in w_ent:
                            W_entries.append((e[0], rank + e[1], e[2]))
                        rank += r
                    elif nb == 3 and mb == 3 and pb == 3:
                        u_ent, v_ent, w_ent, r = build_3x3x3_decomposition(i_off, j_off, k_off, n, m, p)
                        for e in u_ent:
                            U_entries.append((e[0], rank + e[1], e[2]))
                        for e in v_ent:
                            V_entries.append((e[0], rank + e[1], e[2]))
                        for e in w_ent:
                            W_entries.append((e[0], rank + e[1], e[2]))
                        rank += r
                    else:
                        # Schoolbook
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
    
    # Generate all possible splits for each dimension (parts of size 1 or 2)
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)
    
    # Find the best combination of splits (minimum rank)
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
