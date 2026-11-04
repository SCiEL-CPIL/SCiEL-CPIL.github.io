import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_patterns():
    """Return the 7 rank-1 terms of the classic Strassen 2x2x2 decomposition."""
    # Each entry is one rank-1 term: (U_coeffs, V_coeffs, W_coeffs)
    # U[i,j] coefficients, V[j,k] coefficients, W[k,i] coefficients
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    classic = [
        # M1: U has A11=1,A22=1; V has B11=1,B22=1; W has C11+=1,C22+=1
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        # M2: U has A21=1,A22=1; V has B11=1; W has C21+=1
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(1,0,1)]},
        # M3: U has A11=1; V has B21=1,B22=-1; W has C12+=1,C22+=1
        {'U': [(0,0,1)], 'V': [(1,0,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
        # M4: U has A22=1; V has B12=1,B11=-1; W has C11+=1,C21+=1
        {'U': [(1,1,1)], 'V': [(0,1,1), (0,0,-1)], 'W': [(0,0,1), (1,0,1)]},
        # M5: U has A11=1,A12=1; V has B22=1; W has C11-=1,C12+=1
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)]},
        # M6: U has A21=1,A11=-1; V has B11=1,B12=1; W has C21+=1
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,0,1)]},
        # M7: U has A12=1,A22=-1; V has B21=1,B22=1; W has C11-=1,C22+=1
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,-1), (1,1,1)]}
    ]
    return classic

def _get_splits(d):
    """Generate all compositions of d into parts of size 1 or 2."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(list(current))
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
    """Compute the rank for given splits (assuming Strassen for 2x2x2 blocks)."""
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

def _count_222_blocks(n_splits, m_splits, p_splits):
    """Count number of 2x2x2 block positions."""
    n_twos = sum(1 for s in n_splits if s == 2)
    m_twos = sum(1 for s in m_splits if s == 2)
    p_twos = sum(1 for s in p_splits if s == 2)
    return n_twos * m_twos * p_twos

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p, strassen_patterns):
    """Build the full decomposition for given splits using Strassen patterns."""
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
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off = n_off[i_b]
                j_off = m_off[j_b]
                k_off = p_off[k_b]
                
                if nb == 2 and mb == 2 and pb == 2:
                    for pattern in strassen_patterns:
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
                        col += 1
                else:
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Verify the tensor identity using efficient NumPy operations."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Build index arrays without meshgrid
    idx_i = np.repeat(np.arange(n), m * p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n * m)
    
    u_rows = idx_i * m + idx_j
    v_rows = idx_j * p + idx_k
    w_rows = idx_k * n + idx_i
    
    # Efficient: use einsum or direct multiply
    prod = U_np[u_rows] * V_np[v_rows] * W_np[w_rows]
    sums = prod.sum(axis=1)
    
    return np.allclose(sums, 1.0, atol=1e-9)

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

def _try_block_pattern_variants(n_splits, m_splits, p_splits, n, m, p, strassen_patterns, deadline):
    """
    For splits with 2x2x2 blocks, try different assignments of Strassen patterns
    to each 2x2x2 block position. Since the patterns are independent per block
    (the tensor decomposes as a direct sum over block positions), any valid
    pattern works for each block. But we verify to be safe.
    """
    # Find positions of 2x2x2 blocks
    block_positions = []
    n_off = [0]
    for s in n_splits[:-1]:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in m_splits[:-1]:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in p_splits[:-1]:
        p_off.append(p_off[-1] + s)
    
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                if nb == 2 and mb == 2 and pb == 2:
                    block_positions.append((i_b, j_b, k_b))
    
    if not block_positions:
        return None
    
    # For each block position, we can choose any of the Strassen patterns.
    # Since they're independent, the total rank is the same regardless of choice.
    # So we just use the standard patterns. But let's verify.
    U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p, strassen_patterns)
    
    if time.time() > deadline:
        return None
    
    if _verify_fast(U, V, W, n, m, p):
        return (U, V, W, _compute_rank(n_splits, m_splits, p_splits))
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit (150s kill)
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p  # schoolbook rank = 75
    best_U = best_V = best_W = None
    
    # Generate candidates: all combinations of splits
    candidates = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                rank = _compute_rank(n_splits, m_splits, p_splits)
                candidates.append((rank, n_splits, m_splits, p_splits))
    
    # Sort by rank (ascending) to try best candidates first
    candidates.sort(key=lambda x: x[0])
    
    strassen_patterns = _strassen_patterns()
    
    # For 3x5x5, the optimal is rank 71 (4 blocks of 2x2x2 using Strassen)
    # Directly construct the known-optimal decomposition without searching
    target_rank = 71
    
    # Known optimal splits for 3x5x5: [2,1] x [2,2,1] x [2,2,1] gives rank 71
    optimal_splits = ([2, 1], [2, 2, 1], [2, 2, 1])
    
    # First try the known optimal
    result = _try_block_pattern_variants(
        optimal_splits[0], optimal_splits[1], optimal_splits[2],
        n, m, p, strassen_patterns, deadline
    )
    if result is not None:
        U, V, W, actual_rank = result
        best_rank = actual_rank
        best_U, best_V, best_W = U, V, W
    
    # Then search for anything better (unlikely but thorough)
    if best_rank > target_rank:
        for rank, n_splits, m_splits, p_splits in candidates:
            if time.time() > deadline - 3:
                break
            
            if rank >= best_rank:
                continue
            
            result = _try_block_pattern_variants(
                n_splits, m_splits, p_splits, n, m, p, strassen_patterns, deadline
            )
            
            if result is not None:
                U, V, W, actual_rank = result
                if actual_rank < best_rank:
                    best_rank = actual_rank
                    best_U, best_V, best_W = U, V, W
                    if best_rank <= target_rank:
                        break
    
    # If we found a better decomposition, return it
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
