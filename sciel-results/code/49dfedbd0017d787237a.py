import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_patterns():
    """Return the 7 Strassen patterns for 2x2x2 matrix multiplication."""
    return [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
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
    for nb, mb, pb in zip(n_splits, m_splits, p_splits):
        if nb == 2 and mb == 2 and pb == 2:
            rank += 7
        else:
            rank += nb * mb * pb
    return rank

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p):
    """Build the full decomposition for given splits, returning numpy arrays."""
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
                            U[(i_off + li) * m + (j_off + lj), col] = val
                        for lj, lk, val in pattern['V']:
                            V[(j_off + lj) * p + (k_off + lk), col] = val
                        for lk, li, val in pattern['W']:
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
    """Verify the tensor identity using vectorized NumPy operations.
    
    We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    
    # Reshape for efficient einsum
    # U has shape (n*m, rank) -> reshape to (n, m, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using einsum: T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    
    # Check all entries are 1
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def _entries_valid_fast(U, V, W):
    """Check integer or half-integer entries (vectorized)."""
    for mat in (U, V, W):
        vals = mat.astype(np.float64)
        # Check if all values are integers or half-integers
        doubled = 2.0 * vals
        if not np.allclose(doubled, np.round(doubled), atol=1e-10):
            return False
    return True

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as numpy arrays for schoolbook multiplication."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)

    best_rank = n * m * p
    best_UVW = None

    # Precompute all split combinations and sort by predicted rank
    all_combos = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                r = _compute_rank(n_splits, m_splits, p_splits)
                all_combos.append((r, n_splits, m_splits, p_splits))
    all_combos.sort(key=lambda x: x[0])

    for r, n_splits, m_splits, p_splits in all_combos:
        # Prune: if predicted rank >= best, skip (sorted, so we can break)
        if r >= best_rank:
            break
        if time.time() > deadline:
            break
        
        U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
        
        # Fast entry validation
        if not _entries_valid_fast(U, V, W):
            continue
        
        # Fast vectorized verification
        if _verify_decomposition_fast(U, V, W, n, m, p):
            best_rank = r
            best_UVW = (U, V, W)

    if best_UVW is not None:
        U, V, W = best_UVW
    else:
        U, V, W = _schoolbook_factors(n, m, p)

    # Convert numpy arrays to lists of lists (required by interface)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
