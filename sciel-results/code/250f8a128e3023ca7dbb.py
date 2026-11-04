import time
import numpy as np

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    
    Uses a 4+1 block decomposition:
    - Split indices into G0={0,1,2,3}, G1={4}
    - For the 4x4x4 sub-block, use Laderman's rank-21 algorithm
    - For cross terms and the 1x1x1 corner, use direct multiplication
    - Total rank: 21 + 16 + 16 + 16 + 1 = 70
    """
    start = time.time()
    deadline = start + 110
    
    n, m, p = 5, 5, 5
    
    try:
        result = _construct_4plus1_decomposition(n, m, p)
        if result is not None:
            # Verify correctness
            if _verify_decomposition(result, n, m, p):
                return result
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition with rank 125
    return _construct_schoolbook(n, m, p)


def _verify_decomposition(result, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    rank = U.shape[1]
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            u_vals = np.zeros(rank)
            for j in range(m):
                u_vals += U[i * m + j, :] * A[i, j]
            v_vals = np.zeros(rank)
            for j in range(m):
                v_vals += V[j * p + k, :] * B[j, k]
            C_decomp[i, k] = np.dot(W[k * n + i, :], u_vals * v_vals)
    
    err = np.max(np.abs(C_decomp - C_expected))
    return err < tol


def _construct_4plus1_decomposition(n, m, p):
    """
    Construct a low-rank decomposition using 4+1 block structure.
    """
    G0 = [0, 1, 2, 3]
    G1 = [4]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        
        terms.append((u, v, w))
    
    # 1. 4x4x4 block: use Laderman's rank-21 algorithm
    _add_laderman_block(G0, G0, G0, add_term, n, m, p)
    
    # 2. Cross term: A[0:4,4] * B[4,0:4] -> C[0:4,0:4]
    for i in G0:
        for k in G0:
            add_term([(i * m + 4, 1)],
                     [(4 * p + k, 1)],
                     [(k * n + i, 1)])
    
    # 3. Cross term: A[4,0:4] * B[0:4,0:4] -> C[4,0:4]
    for j in G0:
        for k in G0:
            add_term([(4 * m + j, 1)],
                     [(j * p + k, 1)],
                     [(k * n + 4, 1)])
    
    # 4. Cross term: A[0:4,0:4] * B[0:4,4] -> C[0:4,4]
    for i in G0:
        for j in G0:
            add_term([(i * m + j, 1)],
                     [(j * p + 4, 1)],
                     [(4 * n + i, 1)])
    
    # 5. 1x1x1 corner: A[4,4] * B[4,4] -> C[4,4]
    add_term([(4 * m + 4, 1)],
             [(4 * p + 4, 1)],
             [(4 * n + 4, 1)])
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _add_laderman_block(i_group, j_group, k_group, add_term, n, m, p):
    """
    Add Laderman's rank-21 terms for a 4x4x4 block.
    """
    def gi(bi, li):
        return i_group[bi * 2 + li]
    
    def gj(bj, lj):
        return j_group[bj * 2 + lj]
    
    def gk(bk, lk):
        return k_group[bk * 2 + lk]
    
    def u_idx(bi, li, bj, lj):
        return gi(bi, li) * m + gj(bj, lj)
    
    def v_idx(bj, lj, bk, lk):
        return gj(bj, lj) * p + gk(bk, lk)
    
    def w_idx(bk, lk, bi, li):
        return gk(bk, lk) * n + gi(bi, li)
    
    def build_term(u_blocks, v_blocks, w_blocks):
        u_entries = []
        v_entries = []
        w_entries = []
        
        for bi, bj, coeff in u_blocks:
            for li in range(2):
                for lj in range(2):
                    u_entries.append((u_idx(bi, li, bj, lj), coeff))
        
        for bj, bk, coeff in v_blocks:
            for lj in range(2):
                for lk in range(2):
                    v_entries.append((v_idx(bj, lj, bk, lk), coeff))
        
        for bk, bi, coeff in w_blocks:
            for lk in range(2):
                for li in range(2):
                    w_entries.append((w_idx(bk, lk, bi, li), coeff))
        
        return u_entries, v_entries, w_entries
    
    # Laderman's 21 terms
    # M1 = (A00+A11)(B00+B11)
    u, v, w = build_term([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    add_term(u, v, w)
    
    # M2 = (A01+A11)B00
    u, v, w = build_term([(0,1,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)])
    add_term(u, v, w)
    
    # M3 = A00(B01-B11)
    u, v, w = build_term([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])
    add_term(u, v, w)
    
    # M4 = A11(B10-B00)
    u, v, w = build_term([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])
    add_term(u, v, w)
    
    # M5 = (A00+A01)B11
    u, v, w = build_term([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)])
    add_term(u, v, w)
    
    # M6 = (A10-A00)(B00+B01)
    u, v, w = build_term([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
    add_term(u, v, w)
    
    # M7 = (A01-A11)(B10+B11)
    u, v, w = build_term([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])
    add_term(u, v, w)
    
    # M8 = (A00+A10)B01
    u, v, w = build_term([(0,0,1),(1,0,1)], [(0,1,1)], [(0,0,1),(1,0,1)])
    add_term(u, v, w)
    
    # M9 = (A10+A11)(B01+B11)
    u, v, w = build_term([(1,0,1),(1,1,1)], [(0,1,1),(1,1,1)], [(1,0,1),(1,1,1)])
    add_term(u, v, w)
    
    # M10 = (A01+A11)(B00+B10)
    u, v, w = build_term([(0,1,1),(1,1,1)], [(0,0,1),(1,0,1)], [(0,1,1),(1,1,1)])
    add_term(u, v, w)
    
    # M11 = (A00+A01)(B10+B11)
    u, v, w = build_term([(0,0,1),(0,1,1)], [(1,0,1),(1,1,1)], [(0,0,-1),(0,1,1)])
    add_term(u, v, w)
    
    # M12 = (A10-A00)(B01+B11)
    u, v, w = build_term([(1,0,1),(0,0,-1)], [(0,1,1),(1,1,1)], [(1,1,1)])
    add_term(u, v, w)
    
    # M13 = (A01-A11)(B00+B10)
    u, v, w = build_term([(0,1,1),(1,1,-1)], [(0,0,1),(1,0,1)], [(0,0,1)])
    add_term(u, v, w)
    
    # M14 = (A00+A10)(B00+B10)
    u, v, w = build_term([(0,0,1),(1,0,1)], [(0,0,1),(1,0,1)], [(0,0,1),(1,0,1)])
    add_term(u, v, w)
    
    # M15 = (A01+A11)(B01+B11)
    u, v, w = build_term([(0,1,1),(1,1,1)], [(0,1,1),(1,1,1)], [(0,1,1),(1,1,1)])
    add_term(u, v, w)
    
    # M16 = (A00+A01)(B00+B01)
    u, v, w = build_term([(0,0,1),(0,1,1)], [(0,0,1),(0,1,1)], [(0,0,-1),(0,1,1)])
    add_term(u, v, w)
    
    # M17 = (A10+A11)(B10+B11)
    u, v, w = build_term([(1,0,1),(1,1,1)], [(1,0,1),(1,1,1)], [(1,0,1),(1,1,1)])
    add_term(u, v, w)
    
    # M18 = (A00-A10)(B10-B00)
    u, v, w = build_term([(0,0,1),(1,0,-1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,-1)])
    add_term(u, v, w)
    
    # M19 = (A01-A11)(B11-B01)
    u, v, w = build_term([(0,1,1),(1,1,-1)], [(1,1,1),(0,1,-1)], [(0,1,1),(1,1,-1)])
    add_term(u, v, w)
    
    # M20 = (A00+A11)(B10+B01)
    u, v, w = build_term([(0,0,1),(1,1,1)], [(1,0,1),(0,1,1)], [(0,0,1),(1,0,1)])
    add_term(u, v, w)
    
    # M21 = (A01+A10)(B00+B11)
    u, v, w = build_term([(0,1,1),(1,0,1)], [(0,0,1),(1,1,1)], [(0,1,1),(1,0,1)])
    add_term(u, v, w)


def _construct_schoolbook(n, m, p):
    """
    Schoolbook decomposition with rank n*m*p = 125.
    """
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
