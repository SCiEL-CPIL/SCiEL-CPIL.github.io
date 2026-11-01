"""Matrix multiplication 2x2 by 2x13 via Strassen-based decomposition."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEATHLINE = start_time + 130.0  # safety margin before 150s wall-clock kill

    n, m, p = DIMENSIONS  # 2, 2, 13

    # Build U, V, W as lists of lists; each row initially [0]
    U_rows = []
    V_rows = []
    W_rows = []

    # Strassen terms for a 2x2 x 2x2 -> 2x2 block
    # Local A indices: (0,0),(0,1),(1,0),(1,1) -> U row = i*2+j
    # Local B indices: (0,0),(0,1),(1,0),(1,1) -> V row = j*13 + (k or k+1)
    # Local C indices: (0,0),(0,1),(1,0),(1,1) -> W row = (k or k+1)*2 + i
    #
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B10+B11)       -> C01 += M3
    # M4 = A11(B01-B11)       -> C01 += M4, C11 -= M4
    # M5 = (A00+A01)B11       -> C00 += M5, C10 += M5
    # M6 = (A10-A00)(B00+B01) -> C10 -= M6
    # M7 = (A01-A11)(B10+B11) -> C01 -= M7, C11 += M7

    def strassen_coeffs():
        """Return list of (U_local, V_local, W_local) tuples for 7 Strassen terms.
        Each is a dict mapping local index -> coefficient."""
        terms = []
        # M1: (A00+A11)*(B00+B11) -> C00+=1, C11+=1
        terms.append((
            {0: 1, 3: 1},            # A00, A11
            {0: 1, 3: 1},            # B00, B11
            {0: 1, 3: 1},            # C00, C11
        ))
        # M2: (A10+A11)*B00 -> C10+=1
        terms.append((
            {2: 1, 3: 1},            # A10, A11
            {0: 1},                  # B00
            {2: 1},                  # C10
        ))
        # M3: A00*(B10+B11) -> C01+=1
        terms.append((
            {0: 1},                  # A00
            {1: 1, 3: 1},            # B10, B11
            {1: 1},                  # C01
        ))
        # M4: A11*(B01-B11) -> C01+=1, C11-=1
        terms.append((
            {3: 1},                  # A11
            {1: 1, 3: -1},           # B01, -B11
            {1: 1, 3: -1},           # C01, -C11
        ))
        # M5: (A00+A01)*B11 -> C00+=1, C10+=1
        terms.append((
            {0: 1, 1: 1},            # A00, A01
            {3: 1},                  # B11
            {0: 1, 2: 1},            # C00, C10
        ))
        # M6: (A10-A00)*(B00+B01) -> C10-=1
        terms.append((
            {2: 1, 0: -1},           # A10, -A00
            {0: 1, 1: 1},            # B00, B01
            {2: -1},                 # -C10
        ))
        # M7: (A01-A11)*(B10+B11) -> C01-=1, C11+=1
        terms.append((
            {1: 1, 3: -1},           # A01, -A11
            {2: 1, 3: 1},            # B10, B11
            {1: -1, 3: 1},           # -C01, C11
        ))
        return terms

    S = strassen_coeffs()

    # Schoolbook terms for a single column (2x2 x 2x1)
    def schoolbook_coeffs():
        """Return list of (U_local, V_local, W_local) for 4 schoolbook terms."""
        terms = []
        terms.append(({0: 1}, {0: 1}, {0: 1}))  # A00*B0 -> C0
        terms.append(({1: 1}, {1: 1}, {1: 1}))  # A01*B1 -> C1
        terms.append(({2: 1}, {0: 1}, {2: 1}))  # A10*B0 -> C10
        terms.append(({3: 1}, {1: 1}, {3: 1}))  # A11*B1 -> C11
        return terms

    SB = schoolbook_coeffs()

    # Determine column grouping: 6 pairs (cols 0-11) + 1 single (col 12)
    # Total terms: 6*7 + 4 = 46
    total_terms = 6 * 7 + 4
    rank = total_terms

    # Initialize
    U = [[0] * rank for _ in range(n * m)]   # 4 x rank
    V = [[0] * rank for _ in range(m * p)]   # 26 x rank
    W = [[0] * rank for _ in range(p * n)]   # 26 x rank

    col = 0

    # Process 6 pairs of columns
    for pair_idx in range(6):
        k0 = pair_idx * 2
        k1 = pair_idx * 2 + 1
        for t_idx in range(7):
            u_local, v_local, w_local = S[t_idx]
            # U: local index (i*2+j) -> global row i*2+j
            for u_idx, coeff in u_local.items():
                U[u_idx][col] += coeff
            # V: local B index -> global row j*13 + k
            # B local: 0=B(0,k0), 1=B(0,k1), 2=B(1,k0), 3=B(1,k1)
            for v_idx, coeff in v_local.items():
                # v_idx: 0->(0,k0), 1->(0,k1), 2->(1,k0), 3->(1,k1)
                vj = v_idx // 2  # j
                vk = k0 + (v_idx % 2)  # k
                V[vj * p + vk][col] += coeff
            # W: local C index -> global row k*2 + i
            # C local: 0=C(0,k0), 1=C(0,k1), 2=C(1,k0), 3=C(1,k1)
            for w_idx, coeff in w_local.items():
                wi = w_idx // 2  # i
                wk = k0 + (w_idx % 2)  # k
                W[wk * n + wi][col] += coeff
            col += 1

    # Process 1 single column (col 12)
    k0 = 12
    for t_idx in range(4):
        u_local, v_local, w_local = SB[t_idx]
        for u_idx, coeff in u_local.items():
            U[u_idx][col] += coeff
        # V: local B index -> global row j*13 + k0
        # B local: 0=B(0,k0), 1=B(1,k0)
        for v_idx, coeff in v_local.items():
            vj = v_idx  # j (0 or 1)
            V[vj * p + k0][col] += coeff
        # W: local C index -> global row k0*2 + i
        # C local: 0=C(0,k0), 1=C(0,k1=none), 2=C(1,k0), 3=C(1,k1=none)
        # For single column: C indices 0 and 2 are used
        for w_idx, coeff in w_local.items():
            wi = w_idx // 2  # i
            W[k0 * n + wi][col] += coeff
        col += 1

    elapsed = time.time() - start_time
    if time.time() > DEATHLINE:
        # Shouldn't happen since construction is O(1), but guard anyway
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
