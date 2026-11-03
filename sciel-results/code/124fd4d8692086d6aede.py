"""Matrix multiplication 3,5,5 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Strassen patterns for 2x2x2
    def strassen_222():
        terms = []
        # M1 = (a00+a11)(b00+b11) -> C00: +1, C11: +1
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # M2 = (a10+a11)b00 -> C10: +1, C11: -1
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
        # M3 = a00(b01-b11) -> C01: +1, C11: +1
        terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
        # M4 = a11(b10-b00) -> C00: +1, C10: +1
        terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
        # M5 = (a00+a01)b11 -> C00: -1, C01: +1
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
        # M6 = (a10-a00)(b00+b01) -> C11: +1
        terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
        # M7 = (a01-a11)(b10+b11) -> C00: +1
        terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
        return terms
    
    def schoolbook(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i,j):1}, {(j,k):1}, {(k,i):1}))
        return terms
    
    def get_block_decomp(n_b, m_b, p_b):
        """Get decomposition for a block of size (n_b, m_b, p_b)."""
        if n_b == 2 and m_b == 2 and p_b == 2:
            return strassen_222()
        return schoolbook(n_b, m_b, p_b)
    
    def compute_rank(n_splits, m_splits, p_splits):
        """Compute the total rank for given splits."""
        total_rank = 0
        for n_b in n_splits:
            for m_b in m_splits:
                for p_b in p_splits:
                    terms = get_block_decomp(n_b, m_b, p_b)
                    total_rank += len(terms)
        return total_rank
    
    # Try various split strategies
    split_strategies = [
        ([2, 1], [2, 2, 1], [2, 2, 1]),  # Current: rank 71
        ([2, 1], [3, 2], [3, 2]),
        ([2, 1], [2, 3], [2, 3]),
        ([3], [2, 2, 1], [2, 2, 1]),
        ([2, 1], [5], [5]),
        ([3], [5], [5]),
        ([1, 2], [2, 3], [2, 3]),
        ([2, 1], [1, 4], [1, 4]),
        ([2, 1], [4, 1], [4, 1]),
        ([1, 2], [3, 2], [3, 2]),
        ([2, 1], [2, 1, 2], [2, 1, 2]),
        ([2, 1], [1, 2, 2], [1, 2, 2]),
    ]
    
    # Sort by rank and try the best ones
    ranked_strategies = []
    for ns, ms, ps in split_strategies:
        r = compute_rank(ns, ms, ps)
        ranked_strategies.append((r, ns, ms, ps))
    ranked_strategies.sort()
    
    best_result = None
    best_rank = float('inf')
    
    for rank, n_splits, m_splits, p_splits in ranked_strategies:
        if rank >= best_rank:
            break
        if time.time() > deadline - 5:
            break
        
        # Calculate offsets
        n_offsets = [0]
        for s in n_splits[:-1]:
            n_offsets.append(n_offsets[-1] + s)
        m_offsets = [0]
        for s in m_splits[:-1]:
            m_offsets.append(m_offsets[-1] + s)
        p_offsets = [0]
        for s in p_splits[:-1]:
            p_offsets.append(p_offsets[-1] + s)
        
        # Build the decomposition
        U_entries = []
        V_entries = []
        W_entries = []
        rank = 0
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    
                    i_off = n_offsets[i_b]
                    j_off = m_offsets[j_b]
                    k_off = p_offsets[k_b]
                    
                    terms = get_block_decomp(n_b, m_b, p_b)
                    
                    for U_dict, V_dict, W_dict in terms:
                        col = rank
                        for (li, lj), val in U_dict.items():
                            gi = i_off + li
                            gj = j_off + lj
                            U_entries.append((gi * m + gj, col, val))
                        for (lj, lk), val in V_dict.items():
                            gj = j_off + lj
                            gk = k_off + lk
                            V_entries.append((gj * p + gk, col, val))
                        for (lk, li), val in W_dict.items():
                            gk = k_off + lk
                            gi = i_off + li
                            W_entries.append((gk * n + gi, col, val))
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
        
        # Verify the decomposition
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            if rank <= 70:  # Found improvement
                break
    
    if best_result is None:
        # Fallback to schoolbook
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

