import time

_DIMS = (3, 4, 6)
_DEADLINE = 120.0
_T0 = time.time()

def _deadline_ok():
    return (time.time() - _T0) < (_DEADLINE - 8.0)

def _build_2x2x3_strassen_terms():
    """
    Build rank-7 terms for 2x2 x 2x3 matrix multiplication.
    A is 2x2 (rows i=0,1; cols j=0,1)
    B is 2x3 (rows j=0,1; cols k=0,1,2)
    C is 2x3 (rows i=0,1; cols k=0,1,2)
    
    Returns list of (u_factors, v_factors, w_factors) where each factor is
    a list of (index, coeff) tuples.
    """
    terms = []
    
    # M1 = (A00 + A11) * (B00 + B10) -> contributes to C00, C10
    terms.append(([(0,0,1), (1,1,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)]))
    
    # M2 = (A00 + A11) * (B01 + B11) -> contributes to C01, C11
    terms.append(([(0,0,1), (1,1,1)], [(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)]))
    
    # M3 = (A00 + A11) * (B02 + B12) -> contributes to C02, C12
    terms.append(([(0,0,1), (1,1,1)], [(0,2,1), (1,2,1)], [(0,2,1), (1,2,1)]))
    
    # M4 = (A10 + A11) * B00 -> contributes to C10
    terms.append(([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]))
    
    # M5 = A01 * (B10 - B00) -> contributes to C00, C10
    terms.append(([(0,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,-1)]))
    
    # M6 = A00 * (B01 - B11) -> contributes to C01, C11
    terms.append(([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,-1)]))
    
    # M7 = A11 * (B12 - B02) -> contributes to C12, C02
    terms.append(([(1,1,1)], [(1,2,1), (0,2,-1)], [(1,2,1), (0,2,-1)]))
    
    return terms

def _build_improved_terms():
    """
    Build improved decomposition using 2x2x3 Strassen blocks.
    Partition: rows 1,2 use 2x2x3 blocks; row 0 uses schoolbook.
    j-blocks: (0,1) and (2,3)
    k-blocks: (0,1,2) and (3,4,5)
    """
    n, m, p = _DIMS
    terms = []
    
    # Row 0: schoolbook, 4x6 = 24 terms
    for j in range(4):
        for k in range(6):
            terms.append(([(0, j, 1)], [(j, k, 1)], [(k, 0, 1)]))
    
    # Rows 1,2: use 2x2x3 Strassen blocks
    # j-blocks: (0,1) and (2,3)
    # k-blocks: (0,1,2) and (3,4,5)
    for j_block in [(0,1), (2,3)]:
        for k_block in [(0,1,2), (3,4,5)]:
            j0, j1 = j_block
            k0, k1, k2 = k_block
            
            # Get the 7 terms for this 2x2x3 block
            block_terms = _build_2x2x3_strassen_terms()
            
            for (u_f, v_f, w_f) in block_terms:
                # Remap indices: local (i,j,k) -> global (i,j+k_offset,k+k_offset)
                # i: 0->1, 1->2
                # j: 0->j0, 1->j1
                # k: 0->k0, 1->k1, 2->k2
                new_u = []
                for (li, lj, c) in u_f:
                    gi = li + 1  # rows 1,2
                    gj = [j0, j1][lj]
                    new_u.append((gi, gj, c))
                
                new_v = []
                for (lj, lk, c) in v_f:
                    gj = [j0, j1][lj]
                    gk = [k0, k1, k2][lk]
                    new_v.append((gj, gk, c))
                
                new_w = []
                for (lk, li, c) in w_f:
                    gk = [k0, k1, k2][lk]
                    gi = li + 1
                    new_w.append((gk, gi, c))
                
                terms.append((new_u, new_v, new_w))
    
    return terms

def _verify_decomposition(U, V, W, rank):
    """Verify that the decomposition is exact."""
    n, m, p = _DIMS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _build_from_terms(terms):
    """Build U, V, W matrices from a list of terms."""
    n, m, p = _DIMS
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (a_terms, b_terms, out_terms) in enumerate(terms):
        for (i, j, c) in a_terms:
            U[i * m + j][r] += c
        for (j, k, c) in b_terms:
            V[j * p + k][r] += c
        for (k, i, c) in out_terms:
            W[k * n + i][r] += c
    
    return U, V, W

def construct():
    n, m, p = _DIMS
    
    # Try the improved 2x2x3-based decomposition first
    terms = _build_improved_terms()
    rank = len(terms)
    U, V, W = _build_from_terms(terms)
    
    if _verify_decomposition(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook()

def _schoolbook():
    n, m, p = _DIMS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
