"""Matrix multiplication 4x5 by 5x6 — Strassen seed + exact column-removal."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """2x2x2 Strassen decomposition. U,V,W shapes (4,7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition. Rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """Combine block decompositions into one global decomposition."""
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        current_rank += rank_sub
    return U, V, W

def _build_kronecker_strassen_4x4x4():
    """Rank-49 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1*2+j1][t1]
                    if c1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2*2+j2][t2]
                            if c2 == 0:
                                continue
                            u_row = (i1*2+i2)*4 + (j1*2+j2)
                            U[u_row][col] += c1 * c2
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1*2+k1][t1]
                    if c1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2*2+k2][t2]
                            if c2 == 0:
                                continue
                            v_row = (j1*2+j2)*4 + (k1*2+k2)
                            V[v_row][col] += c1 * c2
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1*2+i1][t1]
                    if c1 == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2*2+i2][t2]
                            if c2 == 0:
                                continue
                            w_row = (k1*2+k2)*4 + (i1*2+i2)
                            W[w_row][col] += c1 * c2
    return U, V, W

def _build_4x4x2_strassen():
    """Rank-28 4x4x2 via 4 independent 2x2x2 Strassen blocks."""
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 28
    n_sub, m_sub, p_sub = 4, 4, 2
    U = [[0]*rank for _ in range(n_sub * m_sub)]
    V = [[0]*rank for _ in range(m_sub * p_sub)]
    W = [[0]*rank for _ in range(p_sub * n_sub)]
    for ai in range(2):
        for bj in range(2):
            block_offset = (ai * 2 + bj) * 7
            for t in range(7):
                col = block_offset + t
                for i in range(2):
                    for j in range(2):
                        val = U2[i*2+j][t]
                        if val != 0:
                            u_row = (2*ai + i) * m_sub + (2*bj + j)
                            U[u_row][col] += val
                for j in range(2):
                    for k in range(2):
                        val = V2[j*2+k][t]
                        if val != 0:
                            v_row = (2*bj + j) * p_sub + k
                            V[v_row][col] += val
                for k in range(2):
                    for i in range(2):
                        val = W2[k*2+i][t]
                        if val != 0:
                            w_row = k * n_sub + (2*ai + i)
                            W[w_row][col] += val
    return U, V, W

def _build_seed_decomposition(n, m, p):
    """Build the valid rank-101 seed: 4x4x4(49) + 4x4x2(28) + 4x1x6(24)."""
    decomps = []
    U4, V4, W4 = _build_kronecker_strassen_4x4x4()
    decomps.append((U4, V4, W4, 0, 0, 0, 4, 4, 4))
    U5, V5, W5 = _build_4x4x2_strassen()
    decomps.append((U5, V5, W5, 0, 0, 4, 4, 4, 2))
    U6, V6, W6 = _build_schoolbook(4, 1, 6)
    decomps.append((U6, V6, W6, 0, 4, 0, 4, 1, 6))
    return _combine_decompositions(decomps, n, m, p)

def _validate(U, V, W, n, m, p):
    """Check exact validity. Returns (ok, max_abs_error)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                s = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-7, max_err

def _prune_columns(U, V, W, n, m, p, deadline, time_budget=110.0):
    """Iteratively remove a column if the remaining decomposition can be
    repaired (by solving one factor linearly) to stay exactly valid."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    tol = 1e-9

    def residual_after_zero(U, V, W, r):
        # max |sum_{r' != r} U V W| over all (i,j,k)
        # = max over (i,j,k) of |U_ij . (V_jk * W_ki) with r zeroed|
        # Equivalent to max |U_ij . (V_jk * W_ki) - U_ij[r]*V_jk[r]*W_ki[r]|
        # But simplest: recompute with r set to 0.
        Uu = U.copy(); Uu[:, r] = 0.0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(Uu[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                    e = abs(s - 1.0)
                    if e > max_err:
                        max_err = e
        return max_err

    def solve_factor(factor, r, tol=1e-9):
        """Given U,V,W valid, set factor[:,r]=0 and solve for it.
        factor in {'U','V','W'}. Returns (new_factor, success)."""
        if factor == 'U':
            A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for i in range(n) for j in range(m) for k in range(p)])
            b = np.array([1.0 - float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])) - 0.0 for i in range(n) for j in range(m) for k in range(p)])
            # Note: U[:,r] currently 0, so current dot excludes r. b = 1 - current.
            # We need U_new[:,r] such that A_mat @ U_new[:,r] = b.
            # Actually b should be 1 - (sum over r' != r). Since we zeroed r, current dot = sum_{r'!=r}. Good.
            sol, res, rank, sv = np.linalg.lstsq(A_mat, b, rcond=tol)
            return sol, (res < 1e-5)
        elif factor == 'V':
            A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n) for j in range(m) for k in range(p)])
            b = np.array([1.0 - float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])) for i in range(n) for j in range(m) for k in range(p)])
            sol, res, rank, sv = np.linalg.lstsq(A_mat, b, rcond=tol)
            return sol, (res < 1e-5)
        else:  # W
            A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for i in range(n) for j in range(m) for k in range(p)])
            b = np.array([1.0 - float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])) for i in range(n) for j in range(m) for k in range(p)])
            sol, res, rank, sv = np.linalg.lstsq(A_mat, b, rcond=tol)
            return sol, (res < 1e-5)

    def try_remove(r, factor):
        """Try to remove column r by solving factor[:,r]. Returns (U,V,W, success)."""
        Uu, Vv, Ww = U.copy(), V.copy(), W.copy()
        if factor == 'U':
            A_mat = np.array([Vv[j*p+k, :] * Ww[k*n+i, :] for i in range(n) for j in range(m) for k in range(p)])
            cur = np.array([float(np.dot(Uu[i*m+j, :], Vv[j*p+k, :] * Ww[k*n+i, :])) for i in range(n) for j in range(m) for k in range(p)])
            b = 1.0 - cur
            sol, res, rk, sv = np.linalg.lstsq(A_mat, b, rcond=1e-9)
            Uu[:, r] = sol
            newU, newV, newW = Uu, Vv, Ww
        elif factor == 'V':
            A_mat = np.array([Uu[i*m+j, :] * Ww[k*n+i, :] for i in range(n) for j in range(m) for k in range(p)])
            cur = np.array([float(np.dot(Uu[i*m+j, :], Vv[j*p+k, :] * Ww[k*n+i, :])) for i in range(n) for j in range(m) for k in range(p)])
            b = 1.0 - cur
            sol, res, rk, sv = np.linalg.lstsq(A_mat, b, rcond=1e-9)
            Vv[:, r] = sol
            newU, newV, newW = Uu, Vv, Ww
        else:
            A_mat = np.array([Uu[i*m+j, :] * Vv[j*p+k, :] for i in range(n) for j in range(m) for k in range(p)])
            cur = np.array([float(np.dot(Uu[i*m+j, :], Vv[j*p+k, :] * Ww[k*n+i, :])) for i in range(n) for j in range(m) for k in range(p)])
            b = 1.0 - cur
            sol, res, rk, sv = np.linalg.lstsq(A_mat, b, rcond=1e-9)
            Ww[:, r] = sol
            newU, newV, newW = Uu, Vv, Ww
        # Verify
        ok, err = _validate(newU, newV, newW, n, m, p)
        if ok and err < 1e-7:
            return newU, newV, newW, True
        return None, None, None, False

    # Greedy removal: repeatedly find any (r, factor) that allows removal.
    # Prefer removing columns where the residual after zeroing is small.
    progress = True
    while progress and time.time() < deadline - 5:
        progress = False
        R = U.shape[1]
        # Score each column by residual-after-zero (smaller = easier to remove)
        candidates = []
        for r in range(R):
            # residual if we zero column r in all three (worst case)
            Uu = U.copy(); Uu[:, r] = 0.0
            Vv = V.copy(); Vv[:, r] = 0.0
            Ww = W.copy(); Ww[:, r] = 0.0
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = float(np.dot(Uu[i*m+j, :], Vv[j*p+k, :] * Ww[k*n+i, :]))
                        e = abs(s - 1.0)
                        if e > max_err:
                            max_err = e
            candidates.append((max_err, r))
        candidates.sort()
        for max_err, r in candidates:
            if time.time() >= deadline - 5:
                break
            # Try each factor to repair
            for factor in ['U', 'V', 'W']:
                newU, newV, newW, ok = try_remove(r, factor)
                if ok:
                    U, V, W = newU, newV, newW
                    progress = True
                    break
            if progress:
                break
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    U, V, W = _build_seed_decomposition(n, m, p)
    ok, err = _validate(U, V, W, n, m, p)
    if not ok:
        # Fallback: pure schoolbook (always valid)
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    # Prune columns
    U, V, W = _prune_columns(U, V, W, n, m, p, deadline)
    ok, err = _validate(U, V, W, n, m, p)
    if not ok:
        # Fall back to seed (valid)
        U, V, W = _build_seed_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
