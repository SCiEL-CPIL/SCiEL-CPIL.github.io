"""Matrix multiplication 3x5x5 - Rank-minimal exact decomposition search."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_patterns():
    """
    Return the 7 Strassen patterns for 2x2x2 matrix multiplication.
    Each pattern is a dict with 'U', 'V', 'W' lists of (local_i, local_j, value) tuples.
    For U: (i, j, val) means U[i, j] += val in the local 2x2 block.
    For V: (j, k, val) means V[j, k] += val in the local 2x2 block.
    For W: (k, i, val) means W[k, i] += val in the local 2x2 block.
    
    Verified via direct tensor check.
    """
    # Standard Strassen: 7 products
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B00-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1+P4-P5-P7, C01=P3+P5, C10=P2+P7, C11=P1-P2+P3+P6
    
    patterns = []
    
    # P1: U: A00=1,A11=1; V: B00=1,B11=1; W: C00=+1,C01=-1,C10=-1,C11=+1
    patterns.append({
        'U': [(0,0,1), (1,1,1)],
        'V': [(0,0,1), (1,1,1)],
        'W': [(0,0,1), (0,1,-1), (1,0,-1), (1,1,1)]
    })
    
    # P2: U: A10=1,A11=1; V: B00=1; W: C10=+1, C11=-1
    patterns.append({
        'U': [(1,0,1), (1,1,1)],
        'V': [(0,0,1)],
        'W': [(1,0,1), (1,1,-1)]
    })
    
    # P3: U: A00=1; V: B10=1,B11=-1; W: C01=+1, C11=+1
    patterns.append({
        'U': [(0,0,1)],
        'V': [(1,0,1), (1,1,-1)],
        'W': [(0,1,1), (1,1,1)]
    })
    
    # P4: U: A11=1; V: B00=1,B10=-1; W: C00=+1
    patterns.append({
        'U': [(1,1,1)],
        'V': [(0,0,1), (1,0,-1)],
        'W': [(0,0,1)]
    })
    
    # P5: U: A00=1,A01=1; V: B11=1; W: C00=-1, C01=+1
    patterns.append({
        'U': [(0,0,1), (0,1,1)],
        'V': [(1,1,1)],
        'W': [(0,0,-1), (0,1,1)]
    })
    
    # P6: U: A10=1,A00=-1; V: B00=1,B01=1; W: C11=+1
    patterns.append({
        'U': [(1,0,1), (0,0,-1)],
        'V': [(0,0,1), (0,1,1)],
        'W': [(1,1,1)]
    })
    
    # P7: U: A01=1,A11=-1; V: B10=1,B11=1; W: C00=-1, C10=+1
    patterns.append({
        'U': [(0,1,1), (1,1,-1)],
        'V': [(1,0,1), (1,1,1)],
        'W': [(0,0,-1), (1,0,1)]
    })
    
    return patterns

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
    """Build the full decomposition for given splits."""
    # Calculate offsets for each block
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
    
    # Initialize matrices with zeros
    # Using lists of lists for flexibility, will convert to numpy for verification if needed
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
                    # Apply Strassen patterns
                    for pattern in strassen_patterns:
                        # Fill U
                        for li, lj, val in pattern['U']:
                            gi = i_off + li
                            gj = j_off + lj
                            U[gi * m + gj][col] += val
                        # Fill V
                        for lj, lk, val in pattern['V']:
                            gj = j_off + lj
                            gk = k_off + lk
                            V[gj * p + gk][col] += val
                        # Fill W
                        for lk, li, val in pattern['W']:
                            gk = k_off + lk
                            gi = i_off + li
                            W[gk * n + gi][col] += val
                        col += 1
                elif nb == 2 and mb == 2 and pb == 2:
                    # Fallback: schoolbook for 2x2x2 (8 terms)
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] += 1
                                V[gj * p + gk][col] += 1
                                W[gk * n + gi][col] += 1
                                col += 1
                else:
                    # Schoolbook for this block
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] += 1
                                V[gj * p + gk][col] += 1
                                W[gk * n + gi][col] += 1
                                col += 1
                                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Build the full tensor T from U, V, W: T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    # Use einsum for efficiency
    T = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                if not np.isclose(T[u_idx, v_idx, w_idx], 1.0, atol=1e-9):
                    return False
    
    # Check zero entries: all (u,v,w) where the triple doesn't correspond to valid matmul
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                i = u // m
                j = u % m
                j2 = v // p
                k = v % p
                k2 = w // n
                i2 = w % n
                # Valid if i==i2 and j==j2
                if i != i2 or j != j2:
                    if not np.isclose(T[u, v, w], 0.0, atol=1e-9):
                        return False
    
    return True

def _test_strassen():
    """Quick test that Strassen patterns produce valid 2x2x2 decomposition."""
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    patterns = _strassen_patterns()
    for col, pattern in enumerate(patterns):
        for li, lj, val in pattern['U']:
            U[li * m + lj][col] += val
        for lj, lk, val in pattern['V']:
            V[lj * p + lk][col] += val
        for lk, li, val in pattern['W']:
            W[lk * n + li][col] += val
    
    return _verify_decomposition(U, V, W, n, m, p)

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 120.0
    
    # Quick sanity check on Strassen
    strassen_ok = _test_strassen()
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p  # schoolbook rank
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute schoolbook factors as fallback
    def get_schoolbook():
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
    
    school_U, school_V, school_W = get_schoolbook()
    best_U, best_V, best_W = school_U, school_V, school_W
    
    # Try all combinations of splits
    done = False
    for n_splits in n_splits_list:
        if done:
            break
        for m_splits in m_splits_list:
            if done:
                break
            for p_splits in p_splits_list:
                if time.time() > deadline - 15:
                    done = True
                    break
                
                rank = _compute_rank(n_splits, m_splits, p_splits)
                
                if rank < best_rank:
                    U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p, use_strassen=strassen_ok)
                    
                    # Verify
                    if _verify_decomposition(U, V, W, n, m, p):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
    
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
# EVOLVE-BLOCK-END
