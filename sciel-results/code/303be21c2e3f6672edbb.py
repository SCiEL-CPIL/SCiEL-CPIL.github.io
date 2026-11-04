import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_terms():
    """Return the 7 Strassen rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (U_dict, V_dict, W_dict) with local indices.
    U_dict: {(i,j): val}, V_dict: {(j,k): val}, W_dict: {(k,i): val}
    Convention: W index is (k,i) so W[k*n+i, c]."""
    return [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

def _get_splits(d):
    """Generate all compositions of d into parts of size 1 or 2."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        if remaining >= 1:
            current.append(1)
            _gen(remaining - 1, current)
            current.pop()
        if remaining >= 2:
            current.append(2)
            _gen(remaining - 2, current)
            current.pop()
    _gen(d, [])
    return results

def _compute_rank(n_splits, m_splits, p_splits):
    """Compute the rank for given splits."""
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

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p):
    """Build the full decomposition for given splits. Returns (U, V, W) as numpy arrays."""
    n_off = [0]
    for s in n_splits[:-1]:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in m_splits[:-1]:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in p_splits[:-1]:
        p_off.append(p_off[-1] + s)
    
    rank = _compute_rank(n_splits, m_splits, p_splits)
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    strassen = _strassen_terms()
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off = n_off[i_b]
                j_off = m_off[j_b]
                k_off = p_off[k_b]
                
                if nb == 2 and mb == 2 and pb == 2:
                    for U_d, V_d, W_d in strassen:
                        for (li, lj), val in U_d.items():
                            U[(i_off + li) * m + (j_off + lj), col] = val
                        for (lj, lk), val in V_d.items():
                            V[(j_off + lj) * p + (k_off + lk), col] = val
                        for (lk, li), val in W_d.items():
                            W[(k_off + lk) * n + (i_off + li), col] = val
                        col += 1
                else:
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                U[(i_off + li) * m + (j_off + lj), col] = 1
                                V[(j_off + lj) * p + (k_off + lk), col] = 1
                                W[(k_off + lk) * n + (i_off + li), col] = 1
                                col += 1
    
    return U, V, W

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Vectorized verification: for each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    prod = U[u_rows, :].astype(np.float64) * V[v_rows, :].astype(np.float64) * W[w_rows, :].astype(np.float64)
    sums = prod.sum(axis=1)
    
    return np.allclose(sums, 1.0, atol=1e-10)

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W

def _count_twos(splits):
    """Count number of 2s in a split tuple."""
    return sum(1 for s in splits if s == 2)

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p  # schoolbook rank = 75
    best_U = best_V = best_W = None
    
    # Pre-compute rank for all combinations
    all_combos = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                rank = _compute_rank(n_splits, m_splits, p_splits)
                all_combos.append((rank, n_splits, m_splits, p_splits))
    
    # Sort by rank ascending
    all_combos.sort(key=lambda x: x[0])
    
    for rank, n_splits, m_splits, p_splits in all_combos:
        if time.time() > deadline:
            break
        
        if rank >= best_rank:
            break
        
        U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            if rank < best_rank:
                best_rank = rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
    
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
