"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition.

Strategy: split B into [B1 (4x4), B2 (4x4), B3 (4x1)].
  C = [A*B1, A*B2, A*B3]
  A*B1: Strassen rank 49
  A*B2: Strassen rank 49
  A*B3: schoolbook rank 16
  Total rank: 114

Strassen for 4x4*4x4:
  Split A and B into 2x2 blocks.
  7 block products M1..M7, each a 2x2*2x2 multiplication.
  Each 2x2*2x2 uses Strassen (rank 7).
  Total: 7 * 7 = 49 terms.
"""
import time

_START = time.time()
_DEADLINE = 130.0  # safety margin below 150s kill


def _time_ok():
    return time.time() - _START < _DEADLINE


def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    """Verify sum_c U[u,c]*V[v,c]*W[io,c] == T[io,u,v] for all (io,u,v)."""
    rank = len(U[0])
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[io][c]
                if s != T.get((io, u, v), 0):
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2_terms():
    """Return 7 tuples (u_pattern, v_pattern, w_pattern) for 2x2*2x2 Strassen.
    
    u_pattern: dict mapping (i,j) -> coefficient
    v_pattern: dict mapping (j,k) -> coefficient  
    w_pattern: dict mapping (i,k) -> coefficient
    
    Standard Strassen:
    P1 = (A00+A11)*(B00+B11)
    P2 = (A10+A11)*B00
    P3 = A00*(B01-B11)
    P4 = A11*(B10-B00)
    P5 = (A00+A01)*B11
    P6 = (A10-A00)*(B00+B01)
    P7 = (A01-A11)*(B10+B11)
    
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    """
    terms = []
    
    # P1 = (A00+A11)*(B00+B11)
    # Contributes to C00 and C11
    terms.append((
        {(0,0): 1, (1,1): 1},   # u: A00+A11
        {(0,0): 1, (1,1): 1},   # v: B00+B11
        {(0,0): 1, (1,1): 1}    # w: C00 + C11
    ))
    
    # P2 = (A10+A11)*B00
    # Contributes to C10 and C11 (with -1)
    terms.append((
        {(1,0): 1, (1,1): 1},   # u: A10+A11
        {(0,0): 1},             # v: B00
        {(1,0): 1, (1,1): -1}   # w: C10 - C11
    ))
    
    # P3 = A00*(B01-B11)
    # Contributes to C01 and C11
    terms.append((
        {(0,0): 1},             # u: A00
        {(0,1): 1, (1,1): -1},  # v: B01-B11
        {(0,1): 1, (1,1): 1}    # w: C01 + C11
    ))
    
    # P4 = A11*(B10-B00)
    # Contributes to C00 and C10
    terms.append((
        {(1,1): 1},             # u: A11
        {(1,0): 1, (0,0): -1},  # v: B10-B00
        {(0,0): 1, (1,0): 1}    # w: C00 + C10
    ))
    
    # P5 = (A00+A01)*B11
    # Contributes to C00 (with -1) and C01
    terms.append((
        {(0,0): 1, (0,1): 1},   # u: A00+A01
        {(1,1): 1},             # v: B11
        {(0,0): -1, (0,1): 1}   # w: -C00 + C01
    ))
    
    # P6 = (A10-A00)*(B00+B01)
    # Contributes to C11
    terms.append((
        {(1,0): 1, (0,0): -1},  # u: A10-A00
        {(0,0): 1, (0,1): 1},   # v: B00+B01
        {(1,1): 1}              # w: C11
    ))
    
    # P7 = (A01-A11)*(B10+B11)
    # Contributes to C00
    terms.append((
        {(0,1): 1, (1,1): -1},  # u: A01-A11
        {(1,0): 1, (1,1): 1},   # v: B10+B11
        {(0,0): 1}              # w: C00
    ))
    
    return terms


def _strassen_4x4_4x4_terms():
    """Return 49 tuples (u, v, w) for 4x4*4x4 Strassen decomposition.
    
    u: dict mapping (i,j) -> coeff for A[i,j]
    v: dict mapping (j,k) -> coeff for B[j,k]  
    w: dict mapping (i,k) -> coeff for C[i,k]
    """
    # Top-level Strassen: 7 block products
    # A blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
    # B blocks: B00=B[0:2,0:2], B01=B[0:2,2:4], B10=B[2:4,0:2], B11=B[2:4,2:4]
    
    # Block-level Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mx, we need to decompose the 2x2*2x2 product using Strassen.
    # The 2x2 blocks have their own 2x2 structure.
    
    # Let's define the 7 inner Strassen terms for a generic 2x2*2x2:
    # These operate on local indices (li, lj) for the 2x2 blocks.
    inner_terms = _strassen_2x2_terms()
    
    # Now compose: for each top-level Strassen product, for each inner Strassen term,
    # combine the coefficients.
    
    # Top-level products: (A_block_pattern, B_block_pattern, C_block_pattern)
    # Where block_pattern maps (bi, bj) -> coeff for block (bi,bj)
    
    top_products = [
        # M1: (A00+A11)*(B00+B11) -> C00, C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A10+A11)*B00 -> C10, -C11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: A00*(B01-B11) -> C01, C11
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: A11*(B10-B00) -> C00, C10
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A00+A01)*B11 -> -C00, C01
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A10-A00)*(B00+B01) -> C11
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A01-A11)*(B10+B11) -> C00
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    all_terms = []
    
    for (a_blocks, b_blocks, c_blocks) in top_products:
        for (u_inner, v_inner, w_inner) in inner_terms:
            u = {}
            v = {}
            w = {}
            
            # Combine A coefficients
            for (bi, bj), ca in a_blocks.items():
                for (li, lj), ci in u_inner.items():
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u[(i, j)] = u.get((i, j), 0) + ca * ci
            
            # Combine B coefficients
            for (bi, bj), cb in b_blocks.items():
                for (li, lj), ci in v_inner.items():
                    j = 2 * bi + li
                    k = 2 * bj + lj
                    v[(j, k)] = v.get((j, k), 0) + cb * ci
            
            # Combine C coefficients
            for (bi, bj), cc in c_blocks.items():
                for (li, lj), ci in w_inner.items():
                    i = 2 * bi + li
                    k = 2 * bj + lj
                    w[(i, k)] = w.get((i, k), 0) + cc * ci
            
            all_terms.append((u, v, w))
    
    return all_terms


def _rank114():
    """Build rank-114 decomposition for 4x4 * 4x9.
    
    Splits B into B1 (4x4, cols 0-3), B2 (4x4, cols 4-7), B3 (4x1, col 8).
    C = [A*B1, A*B2, A*B3]
    
    A*B1: Strassen rank 49
    A*B2: Strassen rank 49  
    A*B3: schoolbook rank 16
    Total: 114
    """
    n, m, p = 4, 4, 9
    rank = 114
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * rank for _ in range(p * n)]  # 36 x 114
    
    # Get Strassen terms for 4x4*4x4
    strassen_terms = _strassen_4x4_4x4_terms()
    
    # Part 1: A*B1 (B1 = B[:, 0:4])
    # Output C[:, 0:4]
    for c_idx, (u_pat, v_pat, w_pat) in enumerate(strassen_terms):
        for (i, j), cu in u_pat.items():
            if cu != 0:
                U[i * m + j][c_idx] = cu
        for (j, k), cv in v_pat.items():
            if cv != 0:
                # B1 col k -> B col k
                V[j * p + k][c_idx] = cv
        for (i, k), cw in w_pat.items():
            if cw != 0:
                # C col k -> C col k
                W[k * n + i][c_idx] = cw
    
    # Part 2: A*B2 (B2 = B[:, 4:8])
    # Output C[:, 4:8]
    offset = 49
    for c_idx, (u_pat, v_pat, w_pat) in enumerate(strassen_terms):
        for (i, j), cu in u_pat.items():
            if cu != 0:
                U[i * m + j][c_idx + offset] = cu
        for (j, k), cv in v_pat.items():
            if cv != 0:
                # B2 col k -> B col (4+k)
                V[j * p + (4 + k)][c_idx + offset] = cv
        for (i, k), cw in w_pat.items():
            if cw != 0:
                # C col (4+k)
                W[(4 + k) * n + i][c_idx + offset] = cw
    
    # Part 3: A*B3 (B3 = B[:, 8:9], single column)
    # Schoolbook: C[i,8] = sum_j A[i,j] * B[j,8]
    # Rank = 4*4*1 = 16
    offset2 = 98
    for i in range(n):
        for j in range(m):
            c_idx = offset2 + i * m + j
            U[i * m + j][c_idx] = 1
            V[j * p + 8][c_idx] = 1
            W[8 * n + i][c_idx] = 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 4, 4, 9
    T = _build_tensor(n, m, p)
    
    # Try rank-114 decomposition
    if _time_ok():
        cand = _rank114()
        if _exact_check(cand["U"], cand["V"], cand["W"], T, n, m, p):
            return cand
    
    # Fallback: schoolbook (rank 144)
    best = _schoolbook(n, m, p)
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        return best
    
    return best
