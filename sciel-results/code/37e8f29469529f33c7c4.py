import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_factors():
    """Return (U, V, W) as numpy arrays for 2x2x2 Strassen, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with m=p=n=2.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products for 2x2 matrices
    # M1 = (A11+A22)(B11+B22)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1; W[1,0] = -1; W[2,0] = -1
    
    # M2 = (A21+A22)B11
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = 1
    
    # M3 = A11(B21-B22)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[0,2] = 1; W[1,2] = 1
    
    # M4 = A22(B11-B21)
    U[3,3] = 1
    V[0,3] = 1; V[2,3] = -1
    W[2,3] = 1; W[3,3] = 1
    
    # M5 = (A11+A12)B22
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1; W[3,4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = -1; W[2,5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[1,6] = 1; W[3,6] = -1
    
    return U, V, W

def _schoolbook_factors_2d(n, m, p):
    """Return (U, V, W) numpy arrays for schoolbook n x m times m x p.
    U shape (n*m, n*m*p), V shape (m*p, n*m*p), W shape (p*n, n*m*p).
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _get_splits(d, max_part=2):
    """Generate all compositions of d into parts of size 1 or max_part."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        for part in range(1, min(max_part, remaining) + 1):
            current.append(part)
            _gen(remaining - part, current)
            current.pop()
    _gen(d, [])
    return results

def _compute_rank(n_splits, m_splits, p_splits):
    """Compute the rank for given splits using known optimal sub-ranks."""
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
    """Build the full decomposition for given splits. Returns numpy arrays."""
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
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off = n_off[i_b]
                j_off = m_off[j_b]
                k_off = p_off[k_b]
                
                if nb == 2 and mb == 2 and pb == 2:
                    sU, sV, sW = _strassen_factors()
                    for r in range(7):
                        for gi in range(2):
                            for gj in range(2):
                                U[(i_off + gi) * m + (j_off + gj), col + r] += sU[gi * 2 + gj, r]
                        for gj in range(2):
                            for gk in range(2):
                                V[(j_off + gj) * p + (k_off + gk), col + r] += sV[gj * 2 + gk, r]
                        for gk in range(2):
                            for gi in range(2):
                                W[(k_off + gk) * n + (i_off + gi), col + r] += sW[gk * 2 + gi, r]
                    col += 7
                else:
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj, col] = 1
                                V[gj * p + gk, col] = 1
                                W[gk * n + gi, col] = 1
                                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition using numpy."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
    U, V, W = _schoolbook_factors_2d(n, m, p)
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p  # schoolbook rank = 75
    best_U = best_V = best_W = None
    
    # Try all combinations of splits, sorted by estimated rank
    candidates = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                rank = _compute_rank(n_splits, m_splits, p_splits)
                candidates.append((rank, n_splits, m_splits, p_splits))
    
    # Sort by rank to find best solutions early
    candidates.sort(key=lambda x: x[0])
    
    for rank, n_splits, m_splits, p_splits in candidates:
        if time.time() > deadline - 5:
            break
        
        if rank >= best_rank:
            continue
        
        U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
        
        if _verify_decomposition(U, V, W, n, m, p):
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
    
    # If we found a better decomposition, return it
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
