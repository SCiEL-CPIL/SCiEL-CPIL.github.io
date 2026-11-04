"""Rewritten program: direct linear-algebra rank minimization for M_{2,4,6}."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    N, M, P = n * m, m * p, p * n  # 8, 24, 12

    def build_strassen():
        """Rank-42 block-wise Strassen decomposition (fallback / seed)."""
        rank = 42
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]

        def add_block(a_idx, b_idx, w_idx, c):
            # M1=(a00+a11)(b00+b11)
            U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
            V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
            W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
            # M2=(a10+a11)b00
            U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
            V[b_idx[0]][c+1] += 1
            W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
            # M3=a00(b01-b11)
            U[a_idx[0]][c+2] += 1
            V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
            W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
            # M4=a11(b10-b00)
            U[a_idx[3]][c+3] += 1
            V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
            W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
            # M5=(a00+a01)b11
            U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
            V[b_idx[3]][c+4] += 1
            W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
            # M6=(a10-a00)(b00+b01)
            U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
            V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
            W[w_idx[3]][c+5] += 1
            # M7=(a01-a11)(b10+b11)
            U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
            V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
            W[w_idx[0]][c+6] += 1

        a_L = [0, 1, 4, 5]
        a_R = [2, 3, 6, 7]
        blocks = [
            ([0, 1, 6, 7], [12, 13, 18, 19], [0, 2, 1, 3], 0),
            ([2, 3, 8, 9], [14, 15, 20, 21], [4, 6, 5, 7], 14),
            ([4, 5, 10, 11], [16, 17, 22, 23], [8, 10, 9, 11], 28),
        ]
        for bt, bb, w, c in blocks:
            add_block(a_L, bt, w, c)
            add_block(a_R, bb, w, c + 7)
        return U, V, W

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    # ---- Build candidate factor-pattern tensors ----
    # A "pattern" is a (u,v,w) triple of 0/+-1 vectors that is a valid rank-1
    # tensor factor (its only non-zero entries align on the multiplication
    # tensor). We generate Strassen-style patterns plus direct (i,j,k) terms.
    # The multiplication tensor has support {(i*m+j, j*p+k, k*n+i)}.
    # For a pattern to contribute only to valid entries, the set of (i,j,k)
    # where u[i*m+j]*v[j*p+k]*w[k*n+i] != 0 must be a subset of valid entries
    # (always true) AND the product must be 1 on the entries we want to cover.

    def pattern_tensor(u, v, w):
        """Flatten u⊗v⊗w restricted to valid multiplication entries."""
        t = np.zeros(N * M * P)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t[(i*m+j) * (M*P) + (j*p+k) * P + (k*n+i)] = \
                        u[i*m+j] * v[j*p+k] * w[k*n+i]
        return t

    # Candidate patterns:
    # 1) Direct terms (i,j,k): u=e_{i*m+j}, v=e_{j*p+k}, w=e_{k*n+i}
    # 2) Strassen-style: for each 2x2 sub-block, the 7 patterns.
    # We enumerate all "small support" patterns with entries in {-1,0,1}
    # that are valid (product is 1 on covered entries, 0 elsewhere).

    patterns = []  # list of (u, v, w) as lists
    seen = set()

    # Direct terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*N; v = [0]*M; w = [0]*P
                u[i*m+j] = 1; v[j*p+k] = 1; w[k*n+i] = 1
                key = (tuple(u), tuple(v), tuple(w))
                if key not in seen:
                    seen.add(key); patterns.append((u, v, w))

    # Strassen patterns on 2x2 blocks (A sub-block x B sub-block x C sub-block)
    # A is 2x4 -> sub-blocks: rows {0,1}, cols {0,1},{2,3}
    # B is 4x6 -> sub-blocks: rows {0,1},{2,3}, cols {0,1},{2,3},{4,5}
    # C is 6x2 -> sub-blocks: rows {0,1},{2,3},{4,5}, cols {0,1}
    # A 2x2 Strassen pattern needs A block (2x2), B block (2x2), C block (2x2).
    a_col_blocks = [[0,1],[2,3]]   # col pairs for A
    b_row_blocks = [[0,1],[2,3]]   # row pairs for B
    b_col_blocks = [[0,1],[2,3],[4,5]]  # col pairs for B
    c_row_blocks = [[0,1],[2,3],[4,5]]  # row pairs for C
    c_col_blocks = [[0,1]]         # col pair for C

    def strassen7(a_cols, b_rows, b_cols, c_rows, c_cols):
        """Yield the 7 Strassen patterns for a 2x2 block."""
        # local indices: a_{lr,lc} with lr in {0,1}, lc in {0,1}
        # A element (row lr, col lc) -> global A index lr*m + (a_cols[0]+lc)
        # B element (row lr, col lc) -> global B index (b_rows[0]+lr)*p + (b_cols[0]+lc)
        # C element (row lc, col lr) -> global C index (c_rows[0]+lc)*n + (c_cols[0]+lr)
        def aidx(lr, lc): return lr*m + (a_cols[0]+lc)
        def bidx(lr, lc): return (b_rows[0]+lr)*p + (b_cols[0]+lc)
        def cidx(rc, cc): return (c_rows[0]+rc)*n + (c_cols[0]+cc)

        defs = [
            # (u local, v local, w local) each is dict local->coeff
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
        ]
        for ud, vd, wd in defs:
            u = [0]*N; v = [0]*M; w = [0]*P
            for (lr,lc),c in ud.items(): u[aidx(lr,lc)] = c
            for (lr,lc),c in vd.items(): v[bidx(lr,lc)] = c
            for (rc,cc),c in wd.items(): w[cidx(rc,cc)] = c
            # validate: product must be 1 on covered entries
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = u[i*m+j]*v[j*p+k]*w[k*n+i]
                        # covered entries must be exactly the block's 4 C entries
                        if prod != 0:
                            # check this C entry is in the block
                            if not (c_rows[0] <= k < c_rows[0]+2 and c_cols[0] <= i < c_cols[0]+2):
                                ok = False
                            if prod != 1:
                                ok = False
            if ok:
                yield (u, v, w)

    for ac in a_col_blocks:
        for br in b_row_blocks:
            for bc in b_col_blocks:
                for cr in c_row_blocks:
                    for cc in c_col_blocks:
                        # C block rows must correspond to B cols, C cols to A cols
                        # C row = B col index, C col = A row index (always 0,1)
                        # so cr must equal bc (the B col pair)
                        if cr != bc:
                            continue
                        for (u, v, w) in strassen7(ac, br, bc, cr, cc):
                            key = (tuple(u), tuple(v), tuple(w))
                            if key not in seen:
                                seen.add(key); patterns.append((u, v, w))

    # Build the pattern matrix: rows = patterns, cols = tensor entries
    K = len(patterns)
    Tdim = N * M * P
    Pmat = np.zeros((K, Tdim))
    for idx, (u, v, w) in enumerate(patterns):
        Pmat[idx] = pattern_tensor(u, v, w)

    # Target tensor
    target = np.zeros(Tdim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i*m+j)*(M*P) + (j*p+k)*P + (k*n+i)] = 1

    # ---- Greedy + refinement to find a small subset of patterns summing to target ----
    # We want the smallest set of patterns whose (signed) sum equals target.
    # Since patterns are integer, we solve: find x in {0,1}^K with Pmat @ x = target.
    # Use a greedy basis approach: work in the column space, track the current
    # residual, and at each step pick the pattern that reduces the residual
    # (allowing +/-1) most. Then clean up with integer least squares on the
    # chosen support.

    def try_support(support):
        """Given a set of pattern indices, solve integer coeffs so sum = target.
        Returns coeffs list (integer) if exact, else None."""
        A = Pmat[list(support)].T  # Tdim x s
        # solve A c = target over reals, then check integrality
        try:
            c, res, rank, sv = np.linalg.lstsq(A, target, rcond=None)
        except np.linalg.LinAlgError:
            return None
        # check exactness
        if np.max(np.abs(A @ c - target)) > 1e-6:
            return None
        # round to integers
        ci = np.round(c).astype(int)
        if np.max(np.abs(A @ ci - target)) > 1e-9:
            return None
        return ci

    # Greedy construction
    def greedy_build():
        residual = target.copy()
        chosen = []
        for step in range(60):
            if time.time() > deadline - 20:
                break
            # score each unused pattern by how well it aligns with residual
            # use signed inner product magnitude
            scores = np.abs(Pmat @ residual)
            # exclude already chosen
            for ci in chosen:
                scores[ci] = 0
            best = int(np.argmax(scores))
            if scores[best] < 1e-9:
                break
            # decide sign
            dot = Pmat[best] @ residual
            sign = 1 if dot >= 0 else -1
            # try adding with this sign; accept if residual norm decreases
            new_res = residual - sign * Pmat[best]
            if np.linalg.norm(new_res) < np.linalg.norm(residual) - 1e-12:
                residual = new_res
                chosen.append(best)
            if np.linalg.norm(residual) < 1e-9:
                break
        return chosen, residual

    best_result = None
    best_rank = None

    # First try the greedy support and clean it up
    chosen, residual = greedy_build()
    if chosen:
        coeffs = try_support(chosen)
        if coeffs is not None:
            nz = np.nonzero(coeffs)[0]
            support = [chosen[i] for i in nz]
            cvals = coeffs[nz]
            rank = len(support)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_result = (support, cvals)

    # Try random subset refinement: randomize the greedy order a few times
    rng = np.random.RandomState(0)
    for trial in range(40):
        if time.time() > deadline - 25:
            break
        # perturb greedy: randomly mask some patterns
        mask = np.ones(K, dtype=bool)
        nmask = rng.randint(0, K//3)
        idxs = rng.choice(K, size=nmask, replace=False)
        mask[idxs] = False
        residual = target.copy()
        chosen = []
        for step in range(60):
            if time.time() > deadline - 20:
                break
            scores = np.abs(Pmat @ residual)
            scores[~mask] = 0
            for ci in chosen:
                scores[ci] = 0
            best = int(np.argmax(scores))
            if scores[best] < 1e-9:
                break
            dot = Pmat[best] @ residual
            sign = 1 if dot >= 0 else -1
            new_res = residual - sign * Pmat[best]
            if np.linalg.norm(new_res) < np.linalg.norm(residual) - 1e-12:
                residual = new_res
                chosen.append(best)
            if np.linalg.norm(residual) < 1e-9:
                break
        if chosen:
            coeffs = try_support(chosen)
            if coeffs is not None:
                nz = np.nonzero(coeffs)[0]
                support = [chosen[i] for i in nz]
                cvals = coeffs[nz]
                rank = len(support)
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_result = (support, cvals)

    # Build final matrices from best_result
    if best_result is not None:
        support, cvals = best_result
        rank = len(support)
        U = [[0]*rank for _ in range(N)]
        V = [[0]*rank for _ in range(M)]
        W = [[0]*rank for _ in range(P)]
        for col, (pi, cv) in enumerate(zip(support, cvals)):
            u, v, w = patterns[pi]
            for a in range(N):
                U[a][col] = u[a] * cv
            for b in range(M):
                V[b][col] = v[b] * cv
            for c in range(P):
                W[c][col] = w[c] * cv
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}

    # Fallback: Strassen rank-42
    U, V, W = build_strassen()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Last resort: schoolbook
    rank = N * M * P
    U = [[0]*rank for _ in range(N)]
    V = [[0]*rank for _ in range(M)]
    W = [[0]*rank for _ in range(P)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*M*P + j*P + k
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
    return {"U": U, "V": V, "W": W}
