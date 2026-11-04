import numpy as np
import time
from scipy.optimize import least_squares

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def check_decomposition(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_optimize_rank(R, max_restarts=8):
        if time.time() > deadline - 25:
            return None

        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W

        u_indices, v_indices, w_indices = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_indices.append(i * m + j)
                    v_indices.append(j * p + k)
                    w_indices.append(k * n + i)
        u_indices = np.array(u_indices)
        v_indices = np.array(v_indices)
        w_indices = np.array(w_indices)

        def residuals(x):
            U = x[:size_U].reshape((n*m, R))
            V = x[size_U:size_U+size_V].reshape((m*p, R))
            W = x[size_U+size_V:].reshape((p*n, R))
            U_sel = U[u_indices, :]
            V_sel = V[v_indices, :]
            W_sel = W[w_indices, :]
            res = np.sum(U_sel * V_sel * W_sel, axis=1) - 1.0
            return res

        def try_initialization(x0):
            if time.time() > deadline - 20:
                return None
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=500,
                                       ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                return None

            if result.cost < 1e-8:
                U = result.x[:size_U].reshape((n*m, R))
                V = result.x[size_U:size_U+size_V].reshape((m*p, R))
                W = result.x[size_U+size_V:].reshape((p*n, R))

                for scale in [2, 1]:
                    U_snap = np.round(U * scale) / scale
                    V_snap = np.round(V * scale) / scale
                    W_snap = np.round(W * scale) / scale
                    if check_decomposition(U_snap, V_snap, W_snap):
                        return (U_snap, V_snap, W_snap)
                
                # If snapping fails, try to polish with tighter tolerance
                try:
                    result2 = least_squares(residuals, result.x, method='lm', max_nfev=200,
                                            ftol=1e-14, xtol=1e-14, gtol=1e-14)
                    if result2.cost < 1e-12:
                         U = result2.x[:size_U].reshape((n*m, R))
                         V = result2.x[size_U:size_U+size_V].reshape((m*p, R))
                         W = result2.x[size_U+size_V:].reshape((p*n, R))
                         for scale in [2, 1]:
                             U_snap = np.round(U * scale) / scale
                             V_snap = np.round(V * scale) / scale
                             W_snap = np.round(W * scale) / scale
                             if check_decomposition(U_snap, V_snap, W_snap):
                                 return (U_snap, V_snap, W_snap)
                except Exception:
                    pass

        # Generate initializations
        # 1. Schoolbook minus (60-R) columns
        if R < 60:
            U_s, V_s, W_s = get_schoolbook()
            # Remove last (60-R) columns
            U_init = U_s[:, :R].copy()
            V_init = V_s[:, :R].copy()
            W_init = W_s[:, :R].copy()
            # Perturb slightly to escape the exact schoolbook manifold if needed?
            # Actually, starting exactly on the manifold of a higher rank solution
            # is a good start for lower rank if the removed columns can be absorbed.
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            result = try_initialization(x0)
            if result is not None:
                return result

        # 2. Random restarts
        for attempt in range(max_restarts):
            if time.time() > deadline - 20:
                break
            U_init = np.random.randn(n*m, R) * 0.5
            V_init = np.random.randn(m*p, R) * 0.5
            W_init = np.random.randn(p*n, R) * 0.5
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            result = try_initialization(x0)
            if result is not None:
                return result

        return None

    for R in range(59, 44, -1):
        if time.time() > deadline - 30:
            break
        result = try_optimize_rank(R, max_restarts=4)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
