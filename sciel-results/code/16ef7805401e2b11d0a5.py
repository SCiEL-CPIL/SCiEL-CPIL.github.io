"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen rank-7 decomposition for 2x2x2
    strassen_patterns = [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
    ]
    
    def block_rank(nb, mb, pb):
        """Compute rank for a block of size nb x mb x pb."""
        if nb == 2 and mb == 2 and pb == 2:
            return 7
        # For non-2x2x2 blocks, use schoolbook
        return nb * mb * pb
    
    def generate_splits(dim, max_parts=3):
        """Generate all compositions of dim into at most max_parts parts >= 1."""
        if max_parts == 1:
            yield [dim]
            return
        for first in range(1, dim + 1):
            for rest in generate_splits(dim - first, max_parts - 1):
                yield [first] + rest
    
    def compute_total_rank(n_splits, m_splits, p_splits):
        total = 0
        for nb in n_splits:
            for mb in m_splits:
                for pb in p_splits:
                    total += block_rank(nb, mb, pb)
        return total
    
    def build_decomposition(n_splits, m_splits, p_splits):
        n_offsets = [0]
        for s in n_splits[:-1]:
            n_offsets.append(n_offsets[-1] + s)
        m_offsets = [0]
        for s in m_splits[:-1]:
            m_offsets.append(m_offsets[-1] + s)
        p_offsets = [0]
        for s in p_splits[:-1]:
            p_offsets.append(p_offsets[-1] + s)
        
        max_rank = n * m * p
        U = [[0] * max_rank for _ in range(n * m)]
        V = [[0] * max_rank for _ in range(m * p)]
        W = [[0] * max_rank for _ in range(p * n)]
        rank = 0
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    if time.time() > deadline:
                        break
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    
                    i_off = n_offsets[i_b]
                    j_off = m_offsets[j_b]
                    k_off = p_offsets[k_b]
                    
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        for pattern in strassen_patterns:
                            col = rank
                            for li, lj, val in pattern['U']:
                                gi = i_off + li
                                gj = j_off + lj
                                U[gi * m + gj][col] = val
                            for lj, lk, val in pattern['V']:
                                gj = j_off + lj
                                gk = k_off + lk
                                V[gj * p + gk][col] = val
                            for lk, li, val in pattern['W']:
                                gk = k_off + lk
                                gi = i_off + li
                                W[gk * n + gi][col] = val
                            rank += 1
                    else:
                        for li in range(n_b):
                            for lj in range(m_b):
                                for lk in range(p_b):
                                    col = rank
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    rank += 1
        
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
        return {"U": U, "V": V, "W": W}
    
    # Search over all split combinations to find minimum rank
    best_rank = n * m * p
    best_result = None
    
    n_splits_options = list(generate_splits(n, 3))
    m_splits_options = list(generate_splits(m, 3))
    p_splits_options = list(generate_splits(p, 3))
    
    # Sort by computed rank to try best first
    candidates = []
    for ns in n_splits_options:
        for ms in m_splits_options:
            for ps in p_splits_options:
                r = compute_total_rank(ns, ms, ps)
                if r < best_rank:
                    candidates.append((r, ns, ms, ps))
    
    candidates.sort(key=lambda x: x[0])
    
    for r, ns, ms, ps in candidates:
        if time.time() > deadline - 5:
            break
        result = build_decomposition(ns, ms, ps)
        best_result = result
        best_rank = r
        break  # First candidate is optimal by construction
    
    if best_result is None:
        # Fallback: schoolbook
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        rank = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][rank] = 1
                    V[j*p+k][rank] = 1
                    W[k*n+i][rank] = 1
                    rank += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
