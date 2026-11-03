"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 5, 7]

# Strassen coefficients for the 7 rank-1 terms of 2x2x2 multiplication.
# Term r: U = u_r (coefficients of a_{00},a_{01},a_{10},a_{11}),
#         V = v_r (coefficients of b_{00},b_{01},b_{10},b_{11}),
#         W = w_r (coefficients of c_{00},c_{01},c_{10},c_{11}).
# c_{pq} = sum_r w_r[p*2+q] * (sum u_r[ab] a_ab) * (sum v_r[cd] b_cd)
STRASSEN = [
    # (u, v, w)
    ((1, 0, 0, 0), (0, 0, 1, 1), (1, 0, 0, 0)),   # a00*(b10+b11) -> c00
    ((1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 1)),   # a00*b11 -> c11
    ((0, 1, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)),   # a01*b00 -> c01
    ((0, 0, 1, 1), (0, 0, 1, 0), (1, 0, 0, 0)),   # (a10+a11)*b10 -> c00
    ((0, 0, 1, -1), (0, 1, 0, 0), (0, 1, 0, 0)),  # (a10-a11)*(b01) -> c01
    ((0, 0, 0, 1), (1, 1, 0, 0), (0, 0, 1, 0)),   # a11*(b00+b01) -> c10
    ((0, 0, 1, 0), (1, 0, -1, 0), (1, 0, 1, 0)),  # a10*(b00-b10) -> c00+c10
]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def _verify(result, n, m, p, seed=12345):
    rng = np.random.default_rng(seed)
    rank = len(result['U'][0])
    for _ in range(3):
        A = rng.integers(-5, 6, size=(n, m))
        B = rng.integers(-5, 6, size=(m, p))
        C = A @ B
        Cde = np.zeros((n, p))
        for r in range(rank):
            ua = np.zeros((n, 1))
            vb = np.zeros((1, p))
            for i in range(n):
                for j in range(m):
                    ua[i, 0] += result['U'][i * m + j][r] * A[i, j]
            for j in range(m):
                for k in range(p):
                    vb[0, k] += result['V'][j * p + k][r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    Cde[i, k] += result['W'][k * n + i][r] * ua[i, 0] * vb[0, k]
        if not np.array_equal(Cde, C):
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110  # safety margin vs 150s kill

    # --- Build the structured Strassen + residual decomposition ---
    U = {}
    V = {}
    W = {}

    def add_term(u_idx, v_idx, w_idx, uu, vv, ww):
        U[u_idx] = U.get(u_idx, 0) + uu
        V[v_idx] = V.get(v_idx, 0) + vv
        W[w_idx] = W.get(w_idx, 0) + ww

    # Six disjoint 2x2x2 blocks.
    # Block b: rows I=[2b,2b+1], cols K=[2b,2b+1], middle J=[2b,2b+1].
    for b in range(6):
        I = [2 * b, 2 * b + 1]
        J = [2 * b, 2 * b + 1]
        K = [2 * b, 2 * b + 1]
        for (u, v, w) in STRASSEN:
            for (ai, aj), uu in zip(
                [(0, 0), (0, 1), (1, 0), (1, 1)], u):
                if uu != 0:
                    add_term(I[ai] * m + J[aj],
                             J[aj] * p + K[ai],
                             0, uu, 0, 0)
            for (bj, bk), vv in zip(
                [(0, 0), (0, 1), (1, 0), (1, 1)], v):
                if vv != 0:
                    add_term(0, J[bj] * p + K[bk],
                             0, 0, vv, 0)
            for (ck, ci), ww in zip(
                [(0, 0), (0, 1), (1, 0), (1, 1)], w):
                if ww != 0:
                    add_term(0, 0, K[ck] * n + I[ci],
                             0, 0, ww)

    # Residual terms for the leftover middle column j=5 (b=6 -> j=10,11 out of range).
    for i in range(n):
        for k in range(p):
            add_term(i * m + 5, 5 * p + k, k * n + i, 1, 1, 1)

    cols = sorted(set(U.keys()) | set(V.keys()) | set(W.keys()))
    rank = len(cols)
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    for c, idx in enumerate(cols):
        U_list[idx][c] = U.get(idx, 0)
        V_list[idx][c] = V.get(idx, 0)
        W_list[idx][c] = W.get(idx, 0)

    result = {"U": U_list, "V": V_list, "W": W_list}

    # --- Verify; fall back to schoolbook if inconsistent ---
    if time.time() - start_time < time_limit:
        try:
            if _verify(result, n, m, p):
                return result
        except Exception:
            pass

    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])

    assert len(result['U']) == n * m
    assert len(result['V']) == m * p
    assert len(result['W']) == p * n

    import random
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]

    C_decomp = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            val = 0
            for r in range(rank):
                ua = 0
                for jj in range(m):
                    ua += result['U'][i * m + jj][r] * A[i][jj]
                vb = 0
                for jj in range(m):
                    vb += result['V'][jj * p + k][r] * B[jj][k]
                val += result['W'][k * n + i][r] * ua * vb
            C_decomp[i][k] = val

    if C_decomp != C:
        print("Verification failed!", file=sys.stderr)
    else:
        print(f"Verification passed. rank={rank}", file=sys.stderr)
