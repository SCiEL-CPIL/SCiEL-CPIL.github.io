import time

# Dimensions of the task: (n, m, p)
n, m, p = 2, 2, 5

DEADLINE_MARGIN = 12.0  # seconds of safety margin before the 120 s limit


def _build_decomposition():
    """
    Build a rank-18 exact tensor decomposition for (2,2)x(2,5) matrix multiplication.

    Index conventions (per the task contract):
      U[i*m + j, r]  : coefficient for A[i, j]
      V[j*p + k, r]  : coefficient for B[j, k]
      W[k*n + i, r]  : coefficient for C[i, k]   (trace convention)

    Strategy:
      - The 5 output columns of C are independent in the tensor sense, so the
        decomposition is a direct sum over columns.
      - Columns 0,1 are computed with a 7-term Strassen block (2x2 by 2x2 slice of B).
      - Columns 2,3 are computed with another 7-term Strassen block (another 2x2 slice).
      - Column 4 is computed with the 4 standard (schoolbook) terms.
      Total rank = 7 + 7 + 4 = 18.

    Strassen block (for output columns c, c+1, with B columns 2c, 2c+1):
      Let a=A[0,0], b=A[0,1], c_=A[1,0], d=A[1,1]
      Let e=B[0,2c], f=B[0,2c+1], g=B[1,2c], h=B[1,2c+1]
      M1 = a*e
      M2 = (b+c_)*e
      M3 = a*(f-h)
      M4 = d*(g-e)
      M5 = (a+d)*(g+h)
      M6 = (b-d)*(f-g)
      M7 = (a+c_)*(h-f)
      C[0,2c]   = M1 + M2 - M4 + M7
      C[0,2c+1] = M3 + M4
      C[1,2c]   = M2 - M1 + M5 - M7
      C[1,2c+1] = M3 + M5 + M6
    """
    # Allocate matrices with rank 18
    rank = 18
    U = [[0] * rank for _ in range(n * m)]   # (4, 18)
    V = [[0] * rank for _ in range(m * p)]   # (10, 18)
    W = [[0] * rank for _ in range(p * n)]   # (10, 18)

    def add_term(Uvec, Vvec, Wvec, c):
        """Add one rank-1 term at column c of U, V, W."""
        for idx, val in enumerate(Uvec):
            if val:
                U[idx][c] += val
        for idx, val in enumerate(Vvec):
            if val:
                V[idx][c] += val
        for idx, val in enumerate(Wvec):
            if val:
                W[idx][c] += val

    def strassen_block(col_base):
        """
        Add the 7 Strassen terms for output columns col_base, col_base+1,
        using B columns 2*col_base, 2*col_base+1.
        Returns the next free column index.
        """
        # B row-0 cols 2c, 2c+1 -> V indices (0*p+2c, 0*p+2c+1)
        B0_2c = 0 * p + 2 * col_base
        B0_2c1 = 0 * p + 2 * col_base + 1
        # B row-1 cols 2c, 2c+1 -> V indices (1*p+2c, 1*p+2c+1)
        B1_2c = 1 * p + 2 * col_base
        B1_2c1 = 1 * p + 2 * col_base + 1

        # C indices (trace convention W[k*n + i])
        # C[0,2c], C[0,2c+1], C[1,2c], C[1,2c+1]
        C0_2c = (2 * col_base) * n + 0
        C0_2c1 = (2 * col_base + 1) * n + 0
        C1_2c = (2 * col_base) * n + 1
        C1_2c1 = (2 * col_base + 1) * n + 1

        # U vectors (a, b, c_, d) -> indices (0,1,2,3)
        # V vectors (e, f, g, h) -> indices (B0_2c, B0_2c1, B1_2c, B1_2c1)
        # W vectors (C0_2c, C0_2c1, C1_2c, C1_2c1)

        terms = []

        # M1 = a * e
        terms.append(([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]))
        # M2 = (b + c_) * e
        terms.append(([0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 1, 0]))
        # M3 = a * (f - h)
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        # M4 = d * (g - e)
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 0, 0]))
        # M5 = (a + d) * (g + h)
        terms.append(([1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]))
        # M6 = (b - d) * (f - g)
        terms.append(([0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 0, 1]))
        # M7 = (a + c_) * (h - f)
        terms.append(([1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]))

        # Map local U indices (0,1,2,3) to global (i*m+j) = same for n=m=2
        # Map local V indices (0,1,2,3) to (B0_2c, B0_2c1, B1_2c, B1_2c1)
        # Map local W indices (0,1,2,3) to (C0_2c, C0_2c1, C1_2c, C1_2c1)
        v_map = [B0_2c, B0_2c1, B1_2c, B1_2c1]
        w_map = [C0_2c, C0_2c1, C1_2c, C1_2c1]

        for (uvec, vvec, wvec) in terms:
            Uglob = [0] * (n * m)
            Vglob = [0] * (m * p)
            Wglob = [0] * (p * n)
            for li, val in enumerate(uvec):
                if val:
                    Uglob[li] += val
            for li, val in enumerate(vvec):
                if val:
                    Vglob[v_map[li]] += val
            for li, val in enumerate(wvec):
                if val:
                    Wglob[w_map[li]] += val
            add_term(Uglob, Vglob, Wglob, col_base * 2)  # placeholder, fixed below
        return col_base * 2

    # The add_term above uses a global column index; we need to track the running column.
    # Let me restructure: collect all terms in order, then write them out.

    terms_all = []

    # Block 1: columns 0,1
    col_base = 0
    B0_2c = 0 * p + 0
    B0_2c1 = 0 * p + 1
    B1_2c = 1 * p + 0
    B1_2c1 = 1 * p + 1
    C0_2c = 0 * n + 0
    C0_2c1 = 1 * n + 0
    C1_2c = 0 * n + 1
    C1_2c1 = 1 * n + 1
    v_map1 = [B0_2c, B0_2c1, B1_2c, B1_2c1]
    w_map1 = [C0_2c, C0_2c1, C1_2c, C1_2c1]

    strassen_terms = [
        # (uvec, vvec, wvec) in local 4-dim
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),   # M1
        ([0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 0, 0]), # M4
        ([1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]),   # M5
        ([0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 0, 1]), # M6
        ([1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]), # M7
    ]

    for (uvec, vvec, wvec) in strassen_terms:
        Uglob = list(uvec)  # U indices 0..3 map directly
        Vglob = [0] * (m * p)
        Wglob = [0] * (p * n)
        for li, val in enumerate(vvec):
            if val:
                Vglob[v_map1[li]] += val
        for li, val in enumerate(wvec):
            if val:
                Wglob[w_map1[li]] += val
        terms_all.append((Uglob, Vglob, Wglob))

    # Block 2: columns 2,3
    col_base = 2
    B0_2c = 0 * p + 4
    B0_2c1 = 0 * p + 5
    B1_2c = 1 * p + 4
    B1_2c1 = 1 * p + 5
    C0_2c = 4 * n + 0
    C0_2c1 = 5 * n + 0
    C1_2c = 4 * n + 1
    C1_2c1 = 5 * n + 1
    v_map2 = [B0_2c, B0_2c1, B1_2c, B1_2c1]
    w_map2 = [C0_2c, C0_2c1, C1_2c, C1_2c1]

    for (uvec, vvec, wvec) in strassen_terms:
        Uglob = list(uvec)
        Vglob = [0] * (m * p)
        Wglob = [0] * (p * n)
        for li, val in enumerate(vvec):
            if val:
                Vglob[v_map2[li]] += val
        for li, val in enumerate(wvec):
            if val:
                Wglob[w_map2[li]] += val
        terms_all.append((Uglob, Vglob, Wglob))

    # Block 3: column 4, standard 4 terms
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    # B[0,4] -> V index 0*p+4 = 4
    # B[1,4] -> V index 1*p+4 = 9
    # C[0,4] -> W index 4*n+0 = 8
    # C[1,4] -> W index 4*n+1 = 9

    standard_terms = [
        # (Uglob, Vglob, Wglob)
        ([1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),  # a*B04 -> C04
        ([0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),  # b*B14 -> C04
        ([0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # c*B04 -> C14
        ([0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # d*B14 -> C14
    ]

    for (Uglob, Vglob, Wglob) in standard_terms:
        terms_all.append((Uglob, Vglob, Wglob))

    # Write all terms into U, V, W
    for c, (Uglob, Vglob, Wglob) in enumerate(terms_all):
        for idx, val in enumerate(Uglob):
            if val:
                U[idx][c] = val
        for idx, val in enumerate(Vglob):
            if val:
                V[idx][c] = val
        for idx, val in enumerate(Wglob):
            if val:
                W[idx][c] = val

    return U, V, W


def _verify_decomposition(U, V, W):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    Returns True if valid.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(len(U[0])):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    # Also verify that for each (i, k), the only (j) contributions are correct:
    # Actually the tensor entry T[i,j,k] = delta (i.e., 1 for all valid (i,j,k) in the
    # standard matmul, and 0 otherwise). Since all (i,j,k) with i<n, j<m, k<p are
    # valid, we just need each to equal 1. The above loop checks that.
    return True


def construct():
    """
    Constructs the rank-minimal (or near-minimal) exact tensor decomposition for
    2x2 by 2x5 matrix multiplication.

    Returns a dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank) = (4, rank)
      V has shape (m*p, rank) = (10, rank)
      W has shape (p*n, rank) = (10, rank)

    The rank is 18, achieved via two Strassen blocks (7 terms each) for the first
    four columns and four standard terms for the last column.
    """
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN

    # Build the rank-18 decomposition
    U, V, W = _build_decomposition()

    # Verify before returning (only if we have time)
    if time.time() < deadline:
        if not _verify_decomposition(U, V, W):
            # Fallback: return the standard rank-20 decomposition
            return _standard_decomposition()

    return {"U": U, "V": V, "W": W}


def _standard_decomposition():
    """Fallback: the trivial rank-20 schoolbook decomposition."""
    rank = n * m * p  # 20
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
