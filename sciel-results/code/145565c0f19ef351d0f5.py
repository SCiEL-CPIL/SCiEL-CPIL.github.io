"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time as _time

def _generate_splits(total):
    """Generate all compositions of 'total' into positive integers."""
    results = []
    def helper(remaining, parts_so_far):
        if remaining == 0:
            results.append(tuple(parts_so_far))
            return
        for part in range(1, remaining + 1):
            parts_so_far.append(part)
            helper(remaining - part, parts_so_far)
            parts_so_far.pop()
    helper(total, [])
    return results

def _block_cost(n_b, m_b, p_b):
    """Cost of a block: Strassen if 2x2x2, else schoolbook."""
    if n_b == 2 and m_b == 2 and p_b == 2:
        return 7
    return n_b * m_b * p_b

def _total_rank(n_splits, m_splits, p_splits):
    total = 0
    for n_b in n_splits:
        for m_b in m_splits:
            for p_b in p_splits:
                total += _block_cost(n_b, m_b, p_b)
    return total

def construct():
    start_time = _time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Search for optimal splits
    n_splits_list = _generate_splits(n)
    m_splits_list = _generate_splits(m)
    p_splits_list = _generate_splits(p)
    
    best_rank = n * m * p
    best_n_splits = (n,)
    best_m_splits = (m,)
    best_p_splits = (p,)
    
    for ns in n_splits_list:
        if _time.time() > deadline:
            break
        for ms in m_splits_list:
            if _time.time() > deadline:
                break
            for ps in p_splits_list:
                r = _total_rank(ns, ms, ps)
                if r < best_rank:
                    best_rank = r
                    best_n_splits = ns
                    best_m_splits = ms
                    best_p_splits = ps
    
    n_splits = list(best_n_splits)
    m_splits = list(best_m_splits)
    p_splits = list(best_p_splits)
    
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
        
    # Initialize matrices
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    rank = 0
    
    # Strassen patterns for 2x2x2
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
    
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
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
# EVOLVE-BLOCK-END
