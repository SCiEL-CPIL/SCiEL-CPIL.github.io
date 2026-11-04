import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_patterns():
    """Return the 7 Strassen patterns for 2x2x2 matrix multiplication.
    
    Each pattern has U, V, W as lists of (index1, index2, value) tuples.
    U: (i, j, val) for A[i,j]
    V: (j, k, val) for B[j,k]
    W: (k, i, val) for C[k,i]  -- using trace convention W[k*n+i, r]
    """
    return [
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
    """Compute the rank for given splits using direct sum structure.
    
    The key insight: the decomposition is a direct sum over blocks.
    For each triple (nb, mb, pb) where nb from n_splits, mb from m_splits, pb from p_splits,
    we need rank(nb*mb*pb) for that block. But actually the structure is:
    we partition rows into blocks of sizes n_splits, columns into m_splits, 
    and output into p_splits. The tensor is the direct sum over all (i_b, j_b, k_b) 
    of the nb x mb x pb multiplication tensor.
    
    Wait - that's not right. The matrix multiplication tensor T_{ijk} = delta_{i,k} for 
    C = A*B where A is n x m, B is m x p, C is n x p.
    T_{i,j,k} = 1 if the (i,k) entry of A*B is the (j)th term, i.e., T_{i,j,k} = 1 for all valid.
    
    Actually the tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    This is because C[i,k] = sum_j A[i,j]*B[j,k], so the tensor has T[i,j,k] = 1.
    
    When we partition into blocks, the tensor decomposes as a direct sum only if 
    the partition is compatible. For a block partition, the (i_b, j_b, k_b) block 
    contributes the full nb x mb x p_b all-ones tensor, which has rank nb*mb*pb (schoolbook).
    But if nb=mb=pb=2, we can use Strassen to get rank 7 instead of 8.
    """
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
    
    rank = _compute_rank(n_splits, m_splits, p_splits)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_patterns()
    
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m + j, :], V_np[j*p + k, :] * W_np[k*n + i, :])
                if not np.isclose(s, 1.0):
                    return False
    return True

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

def _try_split(n_splits, m_splits, p_splits, n, m, p):
    """Try a specific combination of splits and return (rank, U, V, W) or None."""
    rank = _compute_rank(n_splits, m_splits, p_splits)
    U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
    if _verify_decomposition(U, V, W, n, m, p):
        return (rank, U, V, W)
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p  # schoolbook rank = 75
    best_U = best_V = best_W = None
    
    # Sort splits by potential rank contribution to find good solutions early
    # A split with more 2s is better (more Strassen blocks possible)
    def _split_score(splits):
        # Count number of 2s - more 2s means more Strassen savings
        return -sum(1 for s in splits if s == 2)
    
    n_splits_list.sort(key=_split_score)
    m_splits_list.sort(key=_split_score)
    p_splits_list.sort(key=_split_score)
    
    # Try all combinations of splits, prioritizing those with more 2s
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                if time.time() > deadline - 5:
                    break
                
                rank = _compute_rank(n_splits, m_splits, p_splits)
                
                if rank < best_rank:
                    result = _try_split(n_splits, m_splits, p_splits, n, m, p)
                    if result is not None:
                        r, U, V, W = result
                        if r < best_rank:
                            best_rank = r
                            best_U, best_V, best_W = U, V, W
    
    # If we found a better decomposition, return it
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
