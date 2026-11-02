import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 2, 12
    t0 = time.time()
    deadline = 135  # safety margin before 150s kill

    def remaining():
        return deadline - (time.time() - t0)

    # --- Fallback: rank 42 (Strassen per 2x2 block of the 2x12) ---
    def get_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(24)]
        W = [[0] * rank for _ in range(24)]
        strassen_coeffs = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        r = 0
        for b in range(6):
            for s in range(7):
                u, v, w = strassen_coeffs[s]
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = u[i_loc * 2 + j_loc]
                        if val != 0:
                            U[i_loc * 2 + j_loc][r] = val
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = v[j_loc * 2 + k_loc]
                        if val != 0:
                            V[j_loc * 12 + (2 * b + k_loc)][r] = val
                for k_loc in range(2):
                    for i_loc in range(2):
                        val = w[k_loc * 2 + i_loc]
                        if val != 0:
                            W[(2 * b + k_loc) * 2 + i_loc][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    # --- Vectorized objective for 2x2x3 ---
    def make_obj_2x2x3(R):
        # T[i,j,k] = sum_s U[i*2+j,s] V[j*3+k,s] W[i*3+k,s]
        # Precompute: for each (i,j,k), the row indices
        ii = np.repeat(np.repeat(np.arange(2), 2), 3)  # 12
        jj = np.repeat(np.tile(np.arange(2), 2), 3)
        kk = np.tile(np.arange(3), 4)
        u_row = ii * 2 + jj
        v_row = jj * 3 + kk
        w_row = ii * 3 + kk
        target = np.ones(12)

        def objective(params):
            U = params[:4*R].reshape(4, R)
            V = params[4*R:4*R+6*R].reshape(6, R)
            W = params[4*R+6*R:].reshape(6, R)
            T = (U[u_row] * V[v_row] * W[w_row]).sum(axis=1)
            d = T - target
            return float(d @ d)
        return objective

    # --- Vectorized objective for 2x2x4 ---
    def make_obj_2x2x4(R):
        ii = np.repeat(np.repeat(np.arange(2), 2), 4)
        jj = np.repeat(np.tile(np.arange(2), 2), 4)
        kk = np.tile(np.arange(4), 4)
        u_row = ii * 2 + jj
        v_row = jj * 4 + kk
        w_row = ii * 4 + kk
        target = np.ones(16)

        def objective(params):
            U = params[:4*R].reshape(4, R)
            V = params[4*R:4*R+8*R].reshape(8, R)
            W = params[4*R+8*R:].reshape(8, R)
            T = (U[u_row] * V[v_row] * W[w_row]).sum(axis=1)
            d = T - target
            return float(d @ d)
        return objective

    def verify_2x2x3(U, V, W, R):
        ii = np.repeat(np.repeat(np.arange(2), 2), 3)
        jj = np.repeat(np.tile(np.arange(2), 2), 3)
        kk = np.tile(np.arange(3), 4)
        T = (U[ii*2+jj] * V[jj*3+kk] * W[ii*3+kk]).sum(axis=1)
        return float(np.max(np.abs(T - 1)))

    def verify_2x2x4(U, V, W, R):
        ii = np.repeat(np.repeat(np.arange(2), 2), 4)
        jj = np.repeat(np.tile(np.arange(2), 2), 4)
        kk = np.tile(np.arange(4), 4)
        T = (U[ii*2+jj] * V[jj*4+kk] * W[ii*4+kk]).sum(axis=1)
        return float(np.max(np.abs(T - 1)))

    def snap_halfint(X):
        return np.round(X * 2) / 2

    def try_snaps(U, V, W, verify_fn, R):
        # Try several snap grids
        for scale in [2, 4, 8]:
            Us = np.round(U * scale) / scale
            Vs = np.round(V * scale) / scale
            Ws = np.round(W * scale) / scale
            if verify_fn(Us, Vs, Ws, R) < 1e-9:
                return Us, Vs, Ws
        return None

    def search_2x2x3(R, num_restarts, budget):
        if remaining() < 5:
            return None
        obj = make_obj_2x2x3(R)
        dim = 4*R + 6*R + 6*R
        best_err = float('inf')
        best_p = None

        for restart in range(num_restarts):
            if remaining() < 5:
                break
            if restart % 3 == 0:
                params = np.random.uniform(-1, 1, dim)
            elif restart % 3 == 1:
                params = np.random.normal(0, 0.7, dim)
            else:
                params = np.zeros(dim)
                nn = max(1, dim // 4)
                nz = np.random.choice(dim, nn, replace=False)
                params[nz] = np.random.uniform(-1, 1, nn)

            res = minimize(obj, params, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 1500, 'ftol': 1e-12})
            if res.fun < best_err:
                best_err = res.fun
                best_p = res.x.copy()
            if best_err < 1e-10:
                break

        if best_err < 1e-8 and best_p is not None:
            U = best_p[:4*R].reshape(4, R)
            V = best_p[4*R:4*R+6*R].reshape(6, R)
            W = best_p[4*R+6*R:].reshape(6, R)
            snapped = try_snaps(U, V, W, verify_2x2x3, R)
            if snapped is not None:
                Us, Vs, Ws = snapped
                # Build full 2x2x12 as 4 blocks
                full_rank = R * 4
                full_U = [[0.0] * full_rank for _ in range(4)]
                full_V = [[0.0] * full_rank for _ in range(24)]
                full_W = [[0.0] * full_rank for _ in range(24)]
                for block in range(4):
                    off = block * R
                    for s in range(R):
                        r = off + s
                        for i in range(2):
                            for j in range(2):
                                full_U[i*2+j][r] = float(Us[i*2+j, s])
                        for j in range(2):
                            for k in range(3):
                                full_V[j*12 + (block*3 + k)][r] = float(Vs[j*3+k, s])
                        for i in range(2):
                            for k in range(3):
                                full_W[(block*3 + k)*2 + i][r] = float(Ws[i*3+k, s])
                return {"U": full_U, "V": full_V, "W": full_W}
        return None

    def search_2x2x4(R, num_restarts, budget):
        if remaining() < 5:
            return None
        obj = make_obj_2x2x4(R)
        dim = 4*R + 8*R + 8*R
        best_err = float('inf')
        best_p = None

        for restart in range(num_restarts):
            if remaining() < 5:
                break
            if restart % 3 == 0:
                params = np.random.uniform(-1, 1, dim)
            elif restart % 3 == 1:
                params = np.random.normal(0, 0.7, dim)
            else:
                params = np.zeros(dim)
                nn = max(1, dim // 4)
                nz = np.random.choice(dim, nn, replace=False)
                params[nz] = np.random.uniform(-1, 1, nn)

            res = minimize(obj, params, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 1500, 'ftol': 1e-12})
            if res.fun < best_err:
                best_err = res.fun
                best_p = res.x.copy()
            if best_err < 1e-10:
                break

        if best_err < 1e-8 and best_p is not None:
            U = best_p[:4*R].reshape(4, R)
            V = best_p[4*R:4*R+8*R].reshape(8, R)
            W = best_p[4*R+8*R:].reshape(8, R)
            snapped = try_snaps(U, V, W, verify_2x2x4, R)
            if snapped is not None:
                Us, Vs, Ws = snapped
                full_rank = R * 3
                full_U = [[0.0] * full_rank for _ in range(4)]
                full_V = [[0.0] * full_rank for _ in range(24)]
                full_W = [[0.0] * full_rank for _ in range(24)]
                for block in range(3):
                    off = block * R
                    for s in range(R):
                        r = off + s
                        for i in range(2):
                            for j in range(2):
                                full_U[i*2+j][r] = float(Us[i*2+j, s])
                        for j in range(2):
                            for k in range(4):
                                full_V[j*12 + (block*4 + k)][r] = float(Vs[j*4+k, s])
                        for i in range(2):
                            for k in range(4):
                                full_W[(block*4 + k)*2 + i][r] = float(Ws[i*4+k, s])
                return {"U": full_U, "V": full_V, "W": full_W}
        return None

    # Try rank 7 (2x2x3) first -> total rank 28
    res = search_2x2x3(R=7, num_restarts=40, budget=60)
    if res:
        return res

    # Rank 8 (2x2x3) -> total 32
    res = search_2x2x3(R=8, num_restarts=25, budget=30)
    if res:
        return res

    # Rank 9 (2x2x4) -> total 27
    res = search_2x2x4(R=9, num_restarts=20, budget=30)
    if res:
        return res

    # Rank 10 (2x2x4) -> total 30
    res = search_2x2x4(R=10, num_restarts=15, budget=20)
    if res:
        return res

    # Fallback
    return get_strassen_42()
