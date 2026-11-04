import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 130.0  # safety margin from 150s kill

# Correct Strassen 2x2x2 multipliers.
# A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = A*B
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6
STR_U = [
    {(0, 0): 1, (1, 1): 1},  # M1 (a00+a11)
    {(1, 0): 1, (1, 1): 1},  # M2 (a10+a11)
    {(0, 0): 1},             # M3 a00
    {(1, 1): 1},             # M4 a11
    {(0, 0): 1, (0, 1): 1},  # M5 (a00+a01)
    {(1, 0): 1, (0, 0): -1}, # M6 (a10-a00)
    {(0, 1): 1, (1, 1): -1}, # M7 (a01-a11)
]
STR_V = [
    {(0, 0): 1, (1, 1): 1},  # (b00+b11)
    {(0, 0): 1},             # b00
    {(0, 1): 1, (1, 1): -1}, # (b01-b11)
    {(1, 0): 1, (0, 0): -1}, # (b10-b00)
    {(1, 1): 1},             # b11
    {(0, 0): 1, (0, 1): 1},  # (b00+b01)
    {(1, 0): 1, (1, 1): 1},  # (b10+b11)
]
# W[r] = dict {(i,k): coeff} : multiplier r's contribution to C[i,k]
STR_W = [
    {(0, 0): 1, (1, 1): 1},            # M1 -> C00(+), C11(+)
    {(1, 0): 1, (1, 1): -1},           # M2 -> C10(+), C11(-)
    {(0, 1): 1, (1, 1): 1},            # M3 -> C01(+), C11(+)
    {(0, 0): 1, (1, 0): 1},            # M4 -> C00(+), C10(+)
    {(0, 0): -1, (0, 1): 1},           # M5 -> C00(-), C01(+)
    {(1, 1): 1},                       # M6 -> C11(+)
    {(0, 0): 1},                       # M7 -> C00(+)
]


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def time_left():
        return DEADLINE_S - (time.time() - start_time)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def validate(U, V, W, tol=1e-8):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.ones((n, m, p))
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i, j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        return np.max(np.abs(T_computed - T_expected)) < tol

    def build_68_rank():
        """Deterministic rank-68 decomposition.

        Part 1: 4 disjoint 2x2 Strassen blocks (i in {0,1}, j in {0,1},
                k in {2t,2t+1}) -> 4 * 7 = 28 terms.
        Part 2: boundary terms (j=2 or i=2) -> 40 schoolbook terms.
        Total 68.
        """
        rank = 68
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for t in range(4):
            k0 = 2 * t
            k1 = 2 * t + 1
            for r in range(7):
                for (ii, jj), coeff in STR_U[r].items():
                    U[ii * m + jj, col] += coeff
                for (jj_l, kk_l), coeff in STR_V[r].items():
                    V[jj_l * p + (k0 + kk_l), col] += coeff
                for (ii_l, kk_l), coeff in STR_W[r].items():
                    W[(k0 + kk_l) * n + ii_l, col] += coeff
                col += 1
        # boundary: i in {0,1}, j=2
        for i in range(2):
            for k in range(p):
                U[i * m + 2, col] = 1.0
                V[2 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
        # boundary: i=2, j in {0,1}
        for j in range(2):
            for k in range(p):
                U[2 * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + 2, col] = 1.0
                col += 1
        # boundary: i=2, j=2
        for k in range(p):
            U[2 * m + 2, col] = 1.0
            V[2 * p + k, col] = 1.0
            W[k * n + 2, col] = 1.0
            col += 1
        assert col == rank, (col, rank)
        return U, V, W

    def try_reduce(U0, V0, W0):
        """Try to lower rank by finding a term that is a half-integer linear
        combination of the others, then absorb it. Returns (U,V,W) of best rank
        found (may be unchanged)."""
        rank = U0.shape[1]
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_rank = rank
        term_tensors = np.zeros((rank, n, m, p))
        for c in range(rank):
            u_mat = U0[:, c].reshape(n, m)
            v_mat = V0[:, c].reshape(m, p)
            w_mat = W0[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i, j]) < 1e-15:
                        continue
                    for k in range(p):
                        term_tensors[c, i, j, k] = u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        term_flat = term_tensors.reshape(rank, -1)
        for c in range(rank):
            if time_left() < 4:
                break
            others = [x for x in range(rank) if x != c]
            A = term_flat[others].T
            b = term_flat[c]
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                if np.max(np.abs(A @ x - b)) > 1e-9:
                    continue
                x_rounded = np.round(x * 2) / 2
                if np.max(np.abs(A @ x_rounded - b)) > 1e-9:
                    continue
                # term c = sum x_i term_i (half-integer coefficients).
                # Absorb: for each i, new term_i = term_i - x_i*term_c, drop term c.
                new_rank = rank - 1
                nU = np.zeros((nm, new_rank))
                nV = np.zeros((mp, new_rank))
                nW = np.zeros((pn, new_rank))
                for new_c, old_c in enumerate(others):
                    nU[:, new_c] = U0[:, old_c] - x_rounded[new_c] * U0[:, c]
                    nV[:, new_c] = V0[:, old_c] - x_rounded[new_c] * V0[:, c]
                    nW[:, new_c] = W0[:, old_c] - x_rounded[new_c] * W0[:, c]
                if validate(nU, nV, nW):
                    best_U, best_V, best_W = nU, nV, nW
                    best_rank = new_rank
                    # rebuild tensors for continued search
                    term_tensors = np.zeros((best_rank, n, m, p))
                    for cc in range(best_rank):
                        um = best_U[:, cc].reshape(n, m)
                        vm = best_V[:, cc].reshape(m, p)
                        wm = best_W[:, cc].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                if abs(um[i, j]) < 1e-15:
                                    continue
                                for k in range(p):
                                    term_tensors[cc, i, j, k] = um[i, j] * vm[j, k] * wm[k, i]
                    term_flat = term_tensors.reshape(best_rank, -1)
                    rank = best_rank
                    U0, V0, W0 = best_U, best_V, best_W
                    c = 0
            except Exception:
                continue
        return best_U, best_V, best_W, best_rank

    U, V, W = build_68_rank()
    if not validate(U, V, W):
        return schoolbook()

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = 68

    # Attempt rank reduction via linear-dependency absorption.
    if time_left() > 25:
        rU, rV, rW, rrank = try_reduce(U, V, W)
        if rrank < best_rank and validate(rU, rV, rW):
            best_rank = rrank
            result = {"U": rU.tolist(), "V": rV.tolist(), "W": rW.tolist()}

    return result
