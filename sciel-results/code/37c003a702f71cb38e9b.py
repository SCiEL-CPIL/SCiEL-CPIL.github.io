"""
Rank-minimal decomposition for M_{2,2,11}.
Baseline: rank-39 (5 Strassen blocks + 1 schoolbook).
Search: try to find rank < 39 via iterative refinement starting from
        column-deleted rank-39 matrices, with multiple seeds and a
        time-bounded budget.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 11]


def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float verification."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += (Fraction(str(U[i * m + j][c])) *
                          Fraction(str(V[j * p + k][c])) *
                          Fraction(str(W[k * n + i][c])))
                if s != 1:
                    return False
    return True


def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1

        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1

        # m1 = (A00+A11)*(B[0,k0]+B[1,k1]) -> C[0,k0] + C[1,k1]
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1

        # m2 = (A10+A11)*B[0,k0] -> C[1,k0] - C[1,k1]
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1

        # m3 = A00*(B[0,k1]-B[1,k1]) -> C[0,k1] + C[1,k1]
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1

        # m4 = A11*(B[1,k0]-B[0,k0]) -> C[0,k0] + C[1,k0]
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1

        # m5 = (A00+A01)*B[1,k1] -> -C[0,k0] + C[0,k1]
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1

        # m6 = (A10-A00)*(B[0,k0]+B[0,k1]) -> C[1,k1]
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1

        # m7 = (A01-A11)*(B[1,k0]+B[1,k1]) -> C[0,k0]
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1

    # Single column k=10: schoolbook 4 terms
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1

    U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
    U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
    U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
    U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1

    assert r == rank
    return U, V, W


def _build_index_structures(n, m, p):
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)

    groups_a = {a: (ta_arr == a) for a in range(n * m)}
    groups_b = {b: (tb_arr == b) for b in range(m * p)}
    groups_c = {c: (tc_arr == c) for c in range(p * n)}

    return ta_arr, tb_arr, tc_arr, groups_a, groups_b, groups_c


def _compute_error(U, V, W, ta_arr, tb_arr, tc_arr):
    rank = U.shape[1]
    sums = np.zeros(len(ta_arr))
    for c in range(rank):
        sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
    return np.sum((sums - 1.0) ** 2)


def _polish(U, V, W, n, m, p, deadline, max_rounds=2000, idx_struct=None):
    rank = U.shape[1]
    if idx_struct is None:
        ta_arr, tb_arr, tc_arr, groups_a, groups_b, groups_c = _build_index_structures(n, m, p)
    else:
        ta_arr, tb_arr, tc_arr, groups_a, groups_b, groups_c = idx_struct

    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for it in range(max_rounds):
        if time.time() > deadline:
            break

        for a in range(n * m):
            mask = groups_a[a]
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            prod = V[tb_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]

        for b in range(m * p):
            mask = groups_b[b]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            prod = U[ta_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]

        for c_idx in range(p * n):
            mask = groups_c[c_idx]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            prod = U[ta_sub, :] * V[tb_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]

        if it % 10 == 0:
            err = _compute_error(U, V, W, ta_arr, tb_arr, tc_arr)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-14:
                break

    return best_U, best_V, best_W, best_err


def _try_round(U, V, W, n, m, p):
    rU = np.round(2 * U) / 2.0
    rV = np.round(2 * V) / 2.0
    rW = np.round(2 * W) / 2.0

    if np.allclose(U, rU, atol=1e-5) and np.allclose(V, rV, atol=1e-5) and np.allclose(W, rW, atol=1e-5):
        if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
            return rU.tolist(), rV.tolist(), rW.tolist()
    return None


def _try_low_rank_random(rank, n, m, p, deadline, max_rounds=1500, seed=42):
    """Try to find a decomposition of given rank using random init + iterative refinement."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)

    ta_arr, tb_arr, tc_arr, groups_a, groups_b, groups_c = _build_index_structures(n, m, p)

    for it in range(max_rounds):
        if time.time() > deadline:
            break

        for a in range(n * m):
            mask = groups_a[a]
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            prod = V[tb_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]

        for b in range(m * p):
            mask = groups_b[b]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            prod = U[ta_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]

        for c_idx in range(p * n):
            mask = groups_c[c_idx]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            prod = U[ta_sub, :] * V[tb_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]

        if it % 50 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            if err < 1e-12:
                return U.tolist(), V.tolist(), W.tolist()

    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Construct and verify rank-39 baseline
    U39, V39, W39 = _construct_rank39()
    if not _verify_float(U39, V39, W39, n, m, p):
        # Fallback: schoolbook
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

    idx_struct = _build_index_structures(n, m, p)
    U39_arr = np.array(U39, dtype=float)
    V39_arr = np.array(V39, dtype=float)
    W39_arr = np.array(W39, dtype=float)

    # Strategy 1: Try rank 38 by deleting one column from rank-39 and polishing
    # Try all 39 possible deletions with limited time each
    best_result = None
    per_attempt_time = 3.0  # seconds per deletion attempt

    for c in range(39):
        if time.time() > deadline - 20:
            break
        U_new = np.delete(U39_arr, c, axis=1)
        V_new = np.delete(V39_arr, c, axis=1)
        W_new = np.delete(W39_arr, c, axis=1)

        pU, pV, pW, perr = _polish(U_new, V_new, W_new, n, m, p,
                                    deadline=min(deadline - 20, time.time() + per_attempt_time),
                                    max_rounds=800, idx_struct=idx_struct)
        if perr < 1e-10:
            res = _try_round(pU, pV, pW, n, m, p)
            if res:
                best_result = res
                break

    if best_result:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

    # Strategy 2: Try rank 37-34 with random initializations
    for rank in [37, 36, 35, 34]:
        if time.time() > deadline - 15:
            break
        for seed in [rank * 13 + 7, rank * 31 + 3, rank * 7 + 11]:
            if time.time() > deadline - 10:
                break
            result = _try_low_rank_random(rank, n, m, p,
                                          deadline=min(deadline - 10, time.time() + 8.0),
                                          max_rounds=1000, seed=seed)
            if result is not None:
                U2, V2, W2 = result
                if _verify_float(U2, V2, W2, n, m, p):
                    # Try to round
                    rU = np.array(U2)
                    rV = np.array(V2)
                    rW = np.array(W2)
                    res = _try_round(rU, rV, rW, n, m, p)
                    if res:
                        return {"U": res[0], "V": res[1], "W": res[2]}
                    else:
                        return {"U": U2, "V": V2, "W": W2}

    # Return the verified rank-39 baseline
    return {"U": U39, "V": V39, "W": W39}
