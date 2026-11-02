<<<<<<< SEARCH
    # Build candidate decompositions
    best_result = None
    best_rank = float('inf')

    candidates = []

    # Candidate 1: original Strassen split (n: 2+1)
    c1 = _build_strassen_3x4x8()
    if _verify(c1['U'], c1['V'], c1['W']) and _verify_half_integer(c1['U'], c1['V'], c1['W']):
        candidates.append((88, c1))

    # Candidate 2: alternative split (n: 1+2)
    if time.time() < deadline:
        c2 = _build_strassen_alt_split()
        if _verify(c2['U'], c2['V'], c2['W']) and _verify_half_integer(c2['U'], c2['V'], c2['W']):
            candidates.append((88, c2))

    # Candidate 3: m-split (m: 2+2)
    if time.time() < deadline:
        c3 = _build_strassen_m_split()
        if _verify(c3['U'], c3['V'], c3['W']) and _verify_half_integer(c3['U'], c3['V'], c3['W']):
            candidates.append((88, c3))
=======
    def _build_strassen_2x2x4_3x4x8():
        """3x4x8: split n=3 into 2+1, p=8 into 4+4.
        2x4x4 blocks use 2x2x4 Strassen-Strassen (rank 28 each), total 56.
        1x4x4 schoolbook: 16. Total: 72."""
        rank = 72
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        # For each p-half (4 wide):
        for ph in range(2):
            # 2x4x4 part (n=2, m=4, p=4) using Strassen-Strassen
            # Split m=4 into 2+2, p=4 into 2+2 within this half
            # For each m-strip j1 in {0,1}:
            #   2x2x4 subproblem on (n=2, m-strip=2, p-half=4)
            #   Use 2x2x4 Strassen-Strassen (rank 14)
            for j1 in range(2):
                ss = _build_strassen_2x2x4()
                for r in range(14):
                    for i in range(2):
                        for j2 in range(2):
                            U[i * m + (2 * j1 + j2)][col] = ss['U'][i * 2 + j2][r]
                    for j2 in range(2):
                        for k in range(4):
                            V[(2 * j1 + j2) * p + (4 * ph + k)][col] = ss['V'][j2 * 4 + k][r]
                    for k in range(4):
                        for i in range(2):
                            W[(4 * ph + k) * n + i][col] = ss['W'][k * 2 + i][r]
                    col += 1
            # 1x4x4 schoolbook part
            for j in range(4):
                for k in range(4):
                    U[2 * m + j][col] = 1
                    V[j * p + (4 * ph + k)][col] = 1
                    W[(4 * ph + k) * n + 2][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build candidate decompositions
    best_result = None
    best_rank = float('inf')

    candidates = []

    # Candidate 1: Strassen-Strassen 3x4x8 (rank 72)
    if time.time() < deadline:
        c0 = _build_strassen_2x2x4_3x4x8()
        r0 = _fast_rank_check(c0['U'], c0['V'], c0['W'])
        if r0 > 0 and _verify(c0['U'], c0['V'], c0['W']) and _verify_half_integer(c0['U'], c0['V'], c0['W']):
            candidates.append((r0, c0))

    # Candidate 2: original Strassen split (n: 2+1)
    c1 = _build_strassen_3x4x8()
    r1 = _fast_rank_check(c1['U'], c1['V'], c1['W'])
    if r1 > 0 and _verify(c1['U'], c1['V'], c1['W']) and _verify_half_integer(c1['U'], c1['V'], c1['W']):
        candidates.append((r1, c1))

    # Candidate 3: alternative split (n: 1+2)
    if time.time() < deadline:
        c2 = _build_strassen_alt_split()
        r2 = _fast_rank_check(c2['U'], c2['V'], c2['W'])
        if r2 > 0 and _verify(c2['U'], c2['V'], c2['W']) and _verify_half_integer(c2['U'], c2['V'], c2['W']):
            candidates.append((r2, c2))

    # Candidate 4: m-split (m: 2+2)
    if time.time() < deadline:
        c3 = _build_strassen_m_split()
        r3 = _fast_rank_check(c3['U'], c3['V'], c3['W'])
        if r3 > 0 and _verify(c3['U'], c3['V'], c3['W']) and _verify_half_integer(c3['U'], c3['V'], c3['W']):
            candidates.append((r3, c3))
>>>>>>> REPLACE

