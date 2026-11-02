import numpy as np
import time
from itertools import product
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook fallback (rank 18)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    def verify(U, V, W, R):
        """Verify exact decomposition using the structural constraint."""
        for i in range(n):
            for k in range(p):
                wv = W[k * n + i]
                M = np.zeros((m, m))
                for j in range(m):
                    uv = U[i * m + j]
                    for jp in range(m):
                        vv = V[jp * p + k]
                        M[j, jp] = np.dot(wv, uv * vv)
                if not np.allclose(M, np.eye(m), atol=1e-7):
                    return False
        return True

    def to_half_int(M):
        return np.round(np.array(M, dtype=float) * 2) / 2

    # Build rank-1 update decomposition from coefficient vectors.
    # u: (n, R) row i -> u_i; a: (m, R) col j -> a_j
    # b: (m, R) row j -> b_j; v: (p, R) col k -> v_k
    # w: (p, R) row k -> w_k; c: (n, R) col i -> c_i
    def build_from_coeffs(u, a, b, v, w, c, R):
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = u[i, r] * a[j, r]
            for j in range(m):
                for k in range(p):
                    V[j * p + k, r] = b[j, r] * v[k, r]
            for k in range(p):
                for i in range(n):
                    W[k * n + i, r] = w[k, r] * c[i, r]
        return U, V, W

    def search_integer(R, max_val=3, max_restarts=40):
        """Search for integer coefficient vectors giving exact rank-R decomposition."""
        # Each coefficient vector entry in [-max_val, max_val]
        # We need u(n), a(m), b(m), v(p), w(p), c(n) each of length R
        # Total variables = R*(n+m+m+p+p+n) = R*(2n+2m+2p) = R*12
        # That's too many for brute force. Instead, use random integer starts + local refinement.
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            rng = np.random.RandomState(restart * 1000 + R)
            # Random integer coefficients in [-max_val, max_val]
            u = rng.randint(-max_val, max_val + 1, size=(n, R))
            a = rng.randint(-max_val, max_val + 1, size=(m, R))
            b = rng.randint(-max_val, max_val + 1, size=(m, R))
            v = rng.randint(-max_val, max_val + 1, size=(p, R))
            w = rng.randint(-max_val, max_val + 1, size=(p, R))
            c = rng.randint(-max_val, max_val + 1, size=(n, R))
            # Ensure at least one nonzero per rank slot to avoid trivial
            for r in range(R):
                if u[:, r].sum() == 0 and a[:, r].sum() == 0:
                    u[0, r] = 1
            U, V, W = build_from_coeffs(u, a, b, v, w, c, R)
            if verify(U, V, W, R):
                return (to_half_int(U), to_half_int(V), to_half_int(W), R)
        return None

    # Polynomial optimization with structural residual (54 constraints)
    def structural_residual(x, R):
        U = x[:nm * R].reshape(nm, R)
        V = x[nm * R: nm * R + mp * R].reshape(mp, R)
        W = x[nm * R + mp * R: nm * R + mp * R + pn * R].reshape(pn, R)
        res = []
        for i in range(n):
            for k in range(p):
                wv = W[k * n + i, :]
                for j in range(m):
                    uv = U[i * m + j, :]
                    for jp in range(m):
                        vv = V[jp * p + k, :]
                        res.append(np.dot(wv, uv * vv) - (1.0 if j == jp else 0.0))
        return np.array(res)

    def optimize_rank(R, max_restarts=15):
        """Use least_squares on structural residual, then round to half-integers."""
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            rng = np.random.RandomState(restart * 7919 + R)
            x0 = rng.randn(R * (nm + mp + pn)) * 0.5
            try:
                sol = least_squares(lambda x: structural_residual(x, R), x0,
                                    method='lm', max_nfev=500, xtol=1e-12, ftol=1e-12)
            except Exception:
                continue
            if sol.cost > 1e-10:
                continue
            U = sol.x[:nm * R].reshape(nm, R)
            V = sol.x[nm * R: nm * R + mp * R].reshape(mp, R)
            W = sol.x[nm * R + mp * R: nm * R + mp * R + pn * R].reshape(pn, R)
            # Round to half-integers and verify
            U_r, V_r, W_r = to_half_int(U), to_half_int(V), to_half_int(W)
            if verify(U_r, V_r, W_r, R):
                return (U_r, V_r, W_r, R)
            # Try a few scalings
            for _ in range(5):
                sc = rng.uniform(0.5, 2.0)
                Us, Vs, Ws = U * sc, V / sc, W
                Ur, Vr, Wr = to_half_int(Us), to_half_int(Vs), to_half_int(Ws)
                if verify(Ur, Vr, Wr, R):
                    return (Ur, Vr, Wr, R)
        return None

    # Try ranks from 15 upward (known lower bound for 2x3x3 is around 15-17)
    for R in range(15, 18):
        if time.time() > deadline - 10:
            break
        # First try integer coefficient search (fast, structured)
        result = search_integer(R, max_val=3, max_restarts=30)
        if result is not None:
            U, V, W, r = result
            if r < len(best['U'][0]):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
        # Then try polynomial optimization
        result = optimize_rank(R, max_restarts=10)
        if result is not None:
            U, V, W, r = result
            if r < len(best['U'][0]):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break

    return best
