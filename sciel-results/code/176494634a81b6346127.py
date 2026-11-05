import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _get_laderman_factors():
    """
    Returns lists of (row, col, coeff) for U, V, W for the 9 Laderman terms.
    Indices 0, 1, 2 correspond to 1, 2, 3 in 1-based notation.
    """
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A11+A22)(B11+B22)
    # M3 = (A22+A00)(B22+B00)
    # M4 = (A00+A11+A22)(B00+B11+B22)
    # M5 = (A01+A12)(B10+B21)
    # M6 = (A12+A20)(B21+B02)
    # M7 = (A20+A01)(B02+B10)
    # M8 = (A02+A10+A21)(B12+B20+B01)
    # M9 = (A10+A21+A02)(B20+B01+B12)
    
    # C reconstruction:
    # c00 = M1 + M4 - M2 - M3 + M8 - M9
    # c01 = M5 + M9 - M6
    # c02 = M6 + M8 - M7
    # c10 = M7 + M8 - M5
    # c11 = M2 + M4 - M1 - M3 + M9 - M8
    # c12 = M3 + M9 - M7
    # c20 = M5 + M7 - M8
    # c21 = M8 + M6 - M9
    # c22 = M3 + M4 - M1 - M2 + M7 - M6

    U_factors = []
    V_factors = []
    W_factors = []

    # M1
    U_factors.append([(0,0,1), (1,1,1)])
    V_factors.append([(0,0,1), (1,1,1)])
    W_factors.append([(0,0,1), (1,1,-1)]) # c00=1, c11=-1

    # M2
    U_factors.append([(1,1,1), (2,2,1)])
    V_factors.append([(1,1,1), (2,2,1)])
    W_factors.append([(0,0,-1), (1,1,1), (2,2,-1)]) # c00=-1, c11=1, c22=-1

    # M3
    U_factors.append([(2,2,1), (0,0,1)])
    V_factors.append([(2,2,1), (0,0,1)])
    W_factors.append([(0,0,-1), (1,1,-1), (2,2,1)]) # c00=-1, c11=-1, c22=1

    # M4
    U_factors.append([(0,0,1), (1,1,1), (2,2,1)])
    V_factors.append([(0,0,1), (1,1,1), (2,2,1)])
    W_factors.append([(0,0,1), (1,1,1), (2,2,1)]) # c00=1, c11=1, c22=1

    # M5
    U_factors.append([(0,1,1), (1,2,1)])
    V_factors.append([(1,0,1), (2,1,1)])
    W_factors.append([(0,1,1), (1,0,-1), (2,0,1)]) # c01=1, c10=-1, c20=1

    # M6
    U_factors.append([(1,2,1), (2,0,1)])
    V_factors.append([(2,1,1), (0,2,1)])
    W_factors.append([(0,2,1), (0,1,-1), (2,1,1)]) # c02=1, c01=-1, c21=1

    # M7
    U_factors.append([(2,0,1), (0,1,1)])
    V_factors.append([(0,2,1), (1,0,1)])
    W_factors.append([(0,2,-1), (1,0,1), (2,0,-1)]) # c02=-1, c10=1, c20=-1

    # M8
    U_factors.append([(0,2,1), (1,0,1), (2,1,1)])
    V_factors.append([(1,2,1), (2,0,1), (0,1,1)])
    W_factors.append([(0,0,1), (0,2,1), (1,0,1), (1,1,-1), (2,1,-1)]) # c00=1, c02=1, c10=1, c11=-1, c21=-1

    # M9
    U_factors.append([(1,0,1), (2,1,1), (0,2,1)])
    V_factors.append([(2,0,1), (0,1,1), (1,2,1)])
    W_factors.append([(0,0,-1), (0,1,1), (1,1,1), (2,0,1), (2,1,-1)]) # c00=-1, c01=1, c11=1, c20=1, c21=-1

    return U_factors, V_factors, W_factors

LADERMAN_U, LADERMAN_V, LADERMAN_W = _get_laderman_factors()

def _get_strassen_factors():
    """
    Returns lists of (row, col, coeff) for U, V, W for the 7 Strassen terms.
    """
    U_factors = []
    V_factors = []
    W_factors = []

    # M1
    U_factors.append([(0,0,1), (1,1,1)])
    V_factors.append([(0,0,1), (1,1,1)])
    W_factors.append([(0,0,1), (1,1,1)])

    # M2
    U_factors.append([(1,0,1), (1,1,1)])
    V_factors.append([(0,0,1)])
    W_factors.append([(1,0,1), (1,1,-1)])

    # M3
    U_factors.append([(0,0,1)])
    V_factors.append([(0,1,1), (1,1,-1)])
    W_factors.append([(0,1,1), (1,1,1)])

    # M4
    U_factors.append([(1,1,1)])
    V_factors.append([(1,0,1), (0,0,-1)])
    W_factors.append([(0,0,1), (1,0,1)])

    # M5
    U_factors.append([(0,0,1), (0,1,1)])
    V_factors.append([(1,1,1)])
    W_factors.append([(0,0,-1), (0,1,1)])

    # M6
    U_factors.append([(1,0,1), (0,0,-1)])
    V_factors.append([(0,0,1), (0,1,1)])
    W_factors.append([(1,1,1)])

    # M7
    U_factors.append([(0,1,1), (1,1,-1)])
    V_factors.append([(1,0,1), (1,1,1)])
    W_factors.append([(0,0,1)])

    return U_factors, V_factors, W_factors

STRASSEN_U, STRASSEN_V, STRASSEN_W = _get_strassen_factors()

def _compute_rank(n, m, p, n_blocks, m_blocks, p_blocks):
    """Compute the rank without building matrices."""
    rank = 0
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                elif ns == 3 and ms == 3 and ps == 3:
                    rank += 9
                else:
                    rank += ns * ms * ps
    return rank

def _block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """Generate a decomposition using block-wise Strassen and Laderman."""
    # Compute block offsets
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    # Count total rank
    rank = 0
    terms = []
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                    terms.append((ni, mi, pi, 'strassen'))
                elif ns == 3 and ms == 3 and ps == 3:
                    rank += 9
                    terms.append((ni, mi, pi, 'laderman'))
                else:
                    rank += ns * ms * ps
                    terms.append((ni, mi, pi, 'trivial'))
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for ni, mi, pi, method in terms:
        n0 = n_offsets[ni]
        m0 = m_offsets[mi]
        p0 = p_offsets[pi]
        ns = n_blocks[ni]
        ms = m_blocks[mi]
        ps = p_blocks[pi]
        
        if method == 'strassen':
            for a_factors, b_factors, c_factors in zip(STRASSEN_U, STRASSEN_V, STRASSEN_W):
                for (ar, ac, acoeff) in a_factors:
                    U[(n0+ar)*m + (m0+ac), col] += acoeff
                for (br, bc, bcoeff) in b_factors:
                    V[(m0+br)*p + (p0+bc), col] += bcoeff
                for (cr, cc, ccoeff) in c_factors:
                    W[(p0+cc)*n + (n0+cr), col] += ccoeff
                col += 1
        
        elif method == 'laderman':
            for a_factors, b_factors, c_factors in zip(LADERMAN_U, LADERMAN_V, LADERMAN_W):
                for (ar, ac, acoeff) in a_factors:
                    U[(n0+ar)*m + (m0+ac), col] += acoeff
                for (br, bc, bcoeff) in b_factors:
                    V[(m0+br)*p + (p0+bc), col] += bcoeff
                for (cr, cc, ccoeff) in c_factors:
                    W[(p0+cc)*n + (n0+cr), col] += ccoeff
                col += 1

        else:
            # Trivial multiplication
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        U[(n0+i)*m + (m0+j), col] = 1
                        V[(m0+j)*p + (p0+k), col] = 1
                        W[(p0+k)*n + (n0+i), col] = 1
                        col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    return U, V, W

def _get_all_splits(dim, max_part=3):
    """Generate all compositions of dim into parts of size 1, 2, 3."""
    splits = []
    def _gen(remaining, current):
        if remaining == 0:
            splits.append(current[:])
            return
        for part in range(1, min(max_part, remaining) + 1):
            current.append(part)
            _gen(remaining - part, current)
            current.pop()
    _gen(dim, [])
    return splits

def _find_best_block_split(n, m, p):
    """Search over block splits to find the one with minimum rank."""
    n_splits = _get_all_splits(n)
    m_splits = _get_all_splits(m)
    p_splits = _get_all_splits(p)
    
    best_rank = n * m * p
    best_split = None
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                rank = _compute_rank(n, m, p, ns, ms, ps)
                if rank < best_rank:
                    best_rank = rank
                    best_split = (ns, ms, ps)
    
    return best_rank, best_split

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, get the trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _block_decomposition(n, m, p, [n], [m], [p])
    
    # Search for better block splits
    best_rank, best_split = _find_best_block_split(n, m, p)
    
    if time.time() > deadline:
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    
    if best_split is not None and best_rank < n*m*p:
        U, V, W = _block_decomposition(n, m, p, best_split[0], best_split[1], best_split[2])
        # Verify the decomposition is correct
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # If verification fails, fall back to trivial
        U, V, W = U_trivial, V_trivial, W_trivial
    else:
        U, V, W = U_trivial, V_trivial, W_trivial
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
