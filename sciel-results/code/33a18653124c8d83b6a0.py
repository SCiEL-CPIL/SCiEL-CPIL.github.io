"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify(U, V, W):
        """Verify exact decomposition under trace convention W[k*n+i, r]."""
        U3 = np.array(U, dtype=float).reshape(n, m, -1)
        V3 = np.array(V, dtype=float).reshape(m, p, -1)
        W3 = np.array(W, dtype=float).reshape(p, n, -1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :]))
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1.0
                    V[j * p + k][c] = 1.0
                    W[k * n + i][c] = 1.0
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Build rank-32 decomposition: 4 Strassen (2x2x2) blocks + 1 schoolbook (2x2x1)
    def build_rank32():
        rank = 32
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            U[0][col] = 1.0; U[3][col] = 1.0
            V[0 * p + k0][col] = 1.0; V[1 * p + k1][col] = 1.0
            W[k0 * n + 0][col] = 1.0; W[k1 * n + 1][col] = 1.0
            col += 1
            U[2][col] = 1.0; U[3][col] = 1.0
            V[0 * p + k0][col] = 1.0
            W[k0 * n + 1][col] = 1.0; W[k1 * n + 1][col] = -1.0
            col += 1
            U[0][col] = 1.0
            V[0 * p + k1][col] = 1.0; V[1 * p + k1][col] = -1.0
            W[k1 * n + 0][col] = 1.0; W[k1 * n + 1][col] = 1.0
            col += 1
            U[3][col] = 1.0
            V[1 * p + k0][col] = 1.0; V[0 * p + k0][col] = -1.0
            W[k0 * n + 0][col] = 1.0; W[k0 * n + 1][col] = 1.0
            col += 1
            U[0][col] = 1.0; U[1][col] = 1.0
            V[1 * p + k1][col] = 1.0
            W[k0 * n + 0][col] = -1.0; W[k1 * n + 0][col] = 1.0
            col += 1
            U[2][col] = 1.0; U[0][col] = -1.0
            V[0 * p + k0][col] = 1.0; V[0 * p + k1][col] = 1.0
            W[k1 * n + 1][col] = 1.0
            col += 1
            U[1][col] = 1.0; U[3][col] = -1.0
            V[1 * p + k0][col] = 1.0; V[1 * p + k1][col] = 1.0
            W[k0 * n + 0][col] = 1.0
            col += 1

        def add_schoolbook(k):
            nonlocal col
            U[0][col] = 1.0; V[0 * p + k][col] = 1.0; W[k * n + 0][col] = 1.0; col += 1
            U[1][col] = 1.0; V[1 * p + k][col] = 1.0; W[k * n + 0][col] = 1.0; col += 1
            U[2][col] = 1.0; V[0 * p + k][col] = 1.0; W[k * n + 1][col] = 1.0; col += 1
            U[3][col] = 1.0; V[1 * p + k][col] = 1.0; W[k * n + 1][col] = 1.0; col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_schoolbook(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U, "V": V, "W": W}

    result = build_rank32()
    if not verify(result["U"], result["V"], result["W"]):
        return schoolbook()
    best = result
    best_rank = 32

    # ---- Refinement: try to reduce rank via smooth nonlinear search ----
    def residuals(x, r):
        U3 = x[:4 * r].reshape(n, m, r)
        V3 = x[4 * r:4 * r + 18 * r].reshape(m, p, r)
        W3 = x[4 * r + 18 * r:4 * r + 18 * r + 18 * r].reshape(p, n, r)
        out = np.empty(36)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    out[idx] = float(np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])) - 1.0
                    idx += 1
        return out

    def pack(U, V, W, r):
        return np.concatenate([
            np.array(U, dtype=float).reshape(-1),
            np.array(V, dtype=float).reshape(-1),
            np.array(W, dtype=float).reshape(-1),
        ][:4 * r + 36 * r])

    def unpack(x, r):
        U = x[:4 * r].reshape(n, m, r).reshape(n * m, r)
        V = x[4 * r:4 * r + 18 * r].reshape(m, p, r).reshape(m * p, r)
        W = x[4 * r + 18 * r:4 * r + 36 * r].reshape(p, n, r).reshape(p * n, r)
        return U, V, W

    def try_rank(r, x0, time_budget):
        if time.time() > deadline - time_budget:
            return None
        try:
            from scipy.optimize import least_squares
        except Exception:
            return None
        t0 = time.time()
        try:
            sol = least_squares(
                residuals, x0, args=(r,), method='lm',
                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000,
            )
        except Exception:
            return None
        if time.time() - t0 > time_budget:
            return None
        if sol.success or np.linalg.norm(residuals(sol.x, r)) < 1e-8:
            U, V, W = unpack(sol.x, r)
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    # Try to reduce the schoolbook tail by solving a global rank-31 fit.
    for r_try in (31, 30):
        if time.time() > deadline - 20:
            break
        x0 = pack(result["U"][:32] and [row[:r_try] for row in result["U"]],
                  [row[:r_try] for row in result["V"]],
                  [row[:r_try] for row in result["W"]], r_try)
        found = try_rank(r_try, x0, 18.0)
        if found is not None:
            best = found
            best_rank = r_try
            break

    return best
# EVOLVE-BLOCK-END
