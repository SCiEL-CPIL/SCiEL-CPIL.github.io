import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 11]
DEADLINE = 145.0
SAFETY = 5.0

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS

    def build_rank39():
        rank = 5 * 7 + 1 * 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0; v_f = k1; v_g = p + k0; v_h = p + k1
            w_c00 = k0 * n; w_c10 = k0 * n + 1
            w_c01 = k1 * n; w_c11 = k1 * n + 1
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        k2 = 10
        v_e2 = k2; v_g2 = p + k2
        w_c02 = k2 * n; w_c12 = k2 * n + 1
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        return {"U": U, "V": V, "W": W}

    best = build_rank39()

    def verify(U, V, W):
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        T = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        M = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    M[a, b, c] = 1.0
        return np.allclose(T, M, atol=1e-9, rtol=0.0)

    def round_halfint(mat):
        return np.round(np.asarray(mat, dtype=float) * 2.0) / 2.0

    def build_target_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = build_target_tensor()
    T_flat = T_target.flatten()
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    def try_rank38_structured():
        """Try rank 38 by merging one Strassen term with schoolbook tail."""
        if time.time() - t_start > DEADLINE - 30:
            return None
        rank = 38
        T = T_target
        T_f = T.flatten()

        def loss(params):
            U = params[:dim_u * rank].reshape(dim_u, rank)
            V = params[dim_u * rank: dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W = params[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
            T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.sum((T_rec.flatten() - T_f) ** 2)

        # Initialize from rank-39 with one term removed and perturbed
        base = build_rank39()
        U_base = np.array(base["U"], dtype=float)
        V_base = np.array(base["V"], dtype=float)
        W_base = np.array(base["W"], dtype=float)

        rng = np.random.default_rng(42)
        best_loss = float('inf')
        best_params = None

        # Strategy: remove one of the last 4 schoolbook terms and try to absorb
        # its contribution into nearby Strassen terms
        n_restarts = 8
        for idx in range(n_restarts):
            if time.time() - t_start > DEADLINE - 10:
                break

            # Start from rank-39, zero out one column, then optimize
            U_init = U_base.copy()
            V_init = V_base.copy()
            W_init = W_base.copy()

            # Remove column 38 (last schoolbook term) and perturb neighbors
            U_init[:, 38] = 0
            V_init[:, 38] = 0
            W_init[:, 38] = 0

            # Perturb the remaining terms slightly
            U_init += rng.normal(0, 0.1, U_init.shape)
            V_init += rng.normal(0, 0.1, V_init.shape)
            W_init += rng.normal(0, 0.1, W_init.shape)

            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            res = minimize(loss, params_init, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
            if best_loss < 1e-8:
                break

        if best_params is None or best_loss > 1e-6:
            return None

        U = best_params[:dim_u * rank].reshape(dim_u, rank)
        V = best_params[dim_u * rank: dim_u * rank + dim_v * rank].reshape(dim_v, rank)
        W = best_params[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
        U_r = round_halfint(U)
        V_r = round_halfint(V)
        W_r = round_halfint(W)
        if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        return None

    def try_rank38_general():
        """General random restarts for rank 38."""
        if time.time() - t_start > DEADLINE - 20:
            return None
        rank = 38
        T_f = T_flat

        def loss(params):
            U = params[:dim_u * rank].reshape(dim_u, rank)
            V = params[dim_u * rank: dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W = params[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
            T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.sum((T_rec.flatten() - T_f) ** 2)

        rng = np.random.default_rng(123)
        best_loss = float('inf')
        best_params = None
        n_restarts = 6
        for idx in range(n_restarts):
            if time.time() - t_start > DEADLINE - 10:
                break
            U_init = rng.normal(0, 1, (dim_u, rank))
            V_init = rng.normal(0, 1, (dim_v, rank))
            W_init = rng.normal(0, 1, (dim_w, rank))
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            res = minimize(loss, params_init, method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
            if best_loss < 1e-8:
                break

        if best_params is None or best_loss > 1e-6:
            return None

        U = best_params[:dim_u * rank].reshape(dim_u, rank)
        V = best_params[dim_u * rank: dim_u * rank + dim_v * rank].reshape(dim_v, rank)
        W = best_params[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
        U_r = round_halfint(U)
        V_r = round_halfint(V)
        W_r = round_halfint(W)
        if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        return None

    # Try structured rank-38 first
    if time.time() - t_start < DEADLINE - 25:
        r38s = try_rank38_structured()
        if r38s:
            return r38s

    # Try general rank-38
    if time.time() - t_start < DEADLINE - 15:
        r38g = try_rank38_general()
        if r38g:
            return r38g

    return best
