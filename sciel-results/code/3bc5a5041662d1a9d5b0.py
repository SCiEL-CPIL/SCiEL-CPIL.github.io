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

def _strassen_rank(size):
    """Return rank for a 2x2x2 Strassen block."""
    return 7

def _block_rank(ns, ms, ps):
    """Return rank for a single block of given sizes."""
    if ns == 2 and ms == 2 and ps == 2:
        return 7
    return ns * ms * ps

def _get_compositions(dim, max_part=2):
    """Generate all compositions of dim using parts up to max_part."""
    comps = []
    def _gen(rem, cur):
        if rem == 0:
            comps.append(cur[:])
            return
        for part in range(1, min(max_part, rem) + 1):
            cur.append(part)
            _gen(rem - part, cur)
            cur.pop()
    _gen(dim, [])
    return comps

def _find_best_composition(n, m, p):
    """Find the composition of n, m, p that minimizes total rank."""
    n_comps = _get_compositions(n)
    m_comps = _get_compositions(m)
    p_comps = _get_compositions(p)
    
    best_rank = n * m * p
    best = (None, None, None)
    
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                # Check that all blocks are valid (same number of blocks)
                if len(nc) != len(mc) or len(nc) != len(pc):
                    continue
                total_rank = 0
                valid = True
                for i in range(len(nc)):
                    if nc[i] == 2 and mc[i] == 2 and pc[i] == 2:
                        total_rank += 7
                    else:
                        total_rank += nc[i] * mc[i] * pc[i]
                if total_rank < best_rank:
                    best_rank = total_rank
                    best = (nc, mc, pc)
    
    return best_rank, best

def _build_decomposition(n, m, p, n_comp, m_comp, p_comp):
    """Build the full decomposition from compositions."""
    if n_comp is None:
        n_comp = [n]
        m_comp = [m]
        p_comp = [p]
    
    # Compute total rank
    total_rank = 0
    for i in range(len(n_comp)):
        ns, ms, ps = n_comp[i], m_comp[i], p_comp[i]
        if ns == 2 and ms == 2 and ps == 2:
            total_rank += 7
        else:
            total_rank += ns * ms * ps
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i in range(len(n_comp)):
        ns, ms, ps = n_comp[i], m_comp[i], p_comp[i]
        # Compute offsets
        n_off = sum(n_comp[:i])
        m_off = sum(m_comp[:i])
        p_off = sum(p_comp[:i])
        
        if ns == 2 and ms == 2 and ps == 2:
            # Strassen 2x2x2
            # Standard Strassen factors for 2x2 matrices
            # We need to map (i,j) -> U index, (j,k) -> V index, (k,i) -> W index
            # i in [0,1], j in [0,1], k in [0,1]
            # U index: (n_off + i) * m + (m_off + j)
            # V index: (m_off + j) * p + (p_off + k)
            # W index: (p_off + k) * n + (n_off + i)
            
            # Strassen 7 terms:
            # M1 = (A11-A22)(B21+B22)
            # M2 = (A11+A12)B22
            # M3 = A11(B11-B22)
            # M4 = A22(B11+B12)
            # M5 = (A12-A22)(B11+B21)
            # M6 = (A11+A22)(B12-B21)
            # M7 = (A21-A11)(B11+B22)
            
            strassen_terms = [
                # (U factors, V factors, W factors)
                # Each factor is a list of (row, col, coeff)
                # Term 1: M1 = (A11-A22)(B21+B22)
                # A part: A11 - A22
                [(0,0,1),(1,1,-1)],
                [(0,1,1),(1,1,1)],
                [(0,0,1),(1,0,1)],
                # Term 2: M2 = (A11+A12)B22
                [(0,0,1),(0,1,1)],
                [(1,1,1)],
                [(0,1,1),(1,1,-1)],
                # Term 3: M3 = A11(B11-B22)
                [(0,0,1)],
                [(0,0,1),(1,1,-1)],
                [(0,0,1),(1,0,-1)],
                # Term 4: M4 = A22(B11+B12)
                [(1,1,1)],
                [(0,0,1),(0,1,1)],
                [(1,0,1),(1,1,1)],
                # Term 5: M5 = (A12-A22)(B11+B21)
                [(0,1,1),(1,1,-1)],
                [(0,0,1),(1,0,1)],
                [(0,0,1),(0,1,1)],
                # Term 6: M6 = (A11+A22)(B12-B21)
                [(0,0,1),(1,1,1)],
                [(0,1,1),(1,0,-1)],
                [(1,1,1),(0,0,1)],
                # Term 7: M7 = (A21-A11)(B11+B22)
                [(1,0,1),(0,0,-1)],
                [(0,0,1),(1,1,1)],
                [(0,1,1),(1,1,1)],
            ]
            
            for a_fac, b_fac, c_fac in strassen_terms:
                for (ar, ac, acoeff) in a_fac:
                    U[(n_off + ar) * m + (m_off + ac), col] += acoeff
                for (br, bc, bcoeff) in b_fac:
                    V[(m_off + br) * p + (p_off + bc), col] += bcoeff
                for (cr, cc, ccoeff) in c_fac:
                    W[(p_off + cc) * n + (n_off + cr), col] += ccoeff
                col += 1
        else:
            # Trivial block
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        U[(n_off + i) * m + (m_off + j), col] = 1
                        V[(m_off + j) * p + (p_off + k), col] = 1
                        W[(p_off + k) * n + (n_off + i), col] = 1
                        col += 1
    
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Find best composition
    best_rank, (n_comp, m_comp, p_comp) = _find_best_composition(n, m, p)
    
    # Build decomposition
    U, V, W, col = _build_decomposition(n, m, p, n_comp, m_comp, p_comp)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to trivial
        U, V, W, col = _build_decomposition(n, m, p, [n], [m], [p])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
