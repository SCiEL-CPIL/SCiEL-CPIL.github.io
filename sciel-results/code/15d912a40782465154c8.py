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
        """Compute the rank for given splits using recursive Strassen."""
        rank = 0
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    rank += _strassen_rank(nb, mb, pb)
        return rank
    
    def _strassen_rank(nb, mb, pb):
        """Recursively compute Strassen rank for block of size nb x mb x pb."""
        if nb == 1 and mb == 1 and pb == 1:
            return 1
        if nb == 1:
            return mb * pb
        if mb == 1:
            return nb * pb
        if pb == 1:
            return nb * mb
        if nb == 2 and mb == 2 and pb == 2:
            return 7
        # Split the largest dimension into 2 + rest
        if nb >= 2:
            r1 = _strassen_rank(2, mb, pb) + _strassen_rank(nb - 2, mb, pb)
        else:
            r1 = float('inf')
        if mb >= 2:
            r2 = _strassen_rank(nb, 2, pb) + _strassen_rank(nb, mb - 2, pb)
        else:
            r2 = float('inf')
        if pb >= 2:
            r3 = _strassen_rank(nb, mb, 2) + _strassen_rank(nb, mb, pb - 2)
        else:
            r3 = float('inf')
        return min(r1, r2, r3, nb * mb * pb)

    def build_decomposition(n_splits, m_splits, p_splits):
        """Build the full decomposition for given splits using recursive Strassen."""
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
        
        def add_block(i_off, j_off, k_off, nb, mb, pb):
            """Recursively add Strassen decomposition for a block."""
            nonlocal rank
            if nb == 1 and mb == 1 and pb == 1:
                col = rank
                gi = i_off
                gj = j_off
                gk = k_off
                U_entries.append((gi * m + gj, col, 1))
                V_entries.append((gj * p + gk, col, 1))
                W_entries.append((gk * n + gi, col, 1))
                rank += 1
                return
            if nb == 1:
                for lj in range(mb):
                    for lk in range(pb):
                        col = rank
                        gi = i_off
                        gj = j_off + lj
                        gk = k_off + lk
                        U_entries.append((gi * m + gj, col, 1))
                        V_entries.append((gj * p + gk, col, 1))
                        W_entries.append((gk * n + gi, col, 1))
                        rank += 1
                return
            if mb == 1:
                for li in range(nb):
                    for lk in range(pb):
                        col = rank
                        gi = i_off + li
                        gj = j_off
                        gk = k_off + lk
                        U_entries.append((gi * m + gj, col, 1))
                        V_entries.append((gj * p + gk, col, 1))
                        W_entries.append((gk * n + gi, col, 1))
                        rank += 1
                return
            if pb == 1:
                for li in range(nb):
                    for lj in range(mb):
                        col = rank
                        gi = i_off + li
                        gj = j_off + lj
                        gk = k_off
                        U_entries.append((gi * m + gj, col, 1))
                        V_entries.append((gj * p + gk, col, 1))
                        W_entries.append((gk * n + gi, col, 1))
                        rank += 1
                return
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
                return
            # Recursively split the largest dimension
            # Try splitting n, m, or p into 2 + rest and pick the one with min rank
            best_rank = float('inf')
            best_split = None
            if nb >= 2:
                r = _strassen_rank(2, mb, pb) + _strassen_rank(nb - 2, mb, pb)
                if r < best_rank:
                    best_rank = r
                    best_split = ('n', r)
            if mb >= 2:
                r = _strassen_rank(nb, 2, pb) + _strassen_rank(nb, mb - 2, pb)
                if r < best_rank:
                    best_rank = r
                    best_split = ('m', r)
            if pb >= 2:
                r = _strassen_rank(nb, mb, 2) + _strassen_rank(nb, mb, pb - 2)
                if r < best_rank:
                    best_rank = r
                    best_split = ('p', r)
            # Fallback to schoolbook
            schoolbook_rank = nb * mb * pb
            if best_split is None or best_rank >= schoolbook_rank:
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
                return
            # Apply the best split
            dim, _ = best_split
            if dim == 'n':
                add_block(i_off, j_off, k_off, 2, mb, pb)
                add_block(i_off + 2, j_off, k_off, nb - 2, mb, pb)
            elif dim == 'm':
                add_block(i_off, j_off, k_off, nb, 2, pb)
                add_block(i_off, j_off + 2, k_off, nb, mb - 2, pb)
            elif dim == 'p':
                add_block(i_off, j_off, k_off, nb, mb, 2)
                add_block(i_off, j_off, k_off + 2, nb, mb, pb - 2)
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]
                    add_block(i_off, j_off, k_off, nb, mb, pb)
        
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
