import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _verify(U, V, W, n=2, m=2, p=5):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _build_tensor(n=2, m=2, p=5):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _get_strassen_block_np(n=2, m=2, p=5, col0=0, col1=1):
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))

    U[0, 0] = 1; U[3, 0] = 1
    V[0 * p + col0, 0] = 1; V[1 * p + col1, 0] = 1
    W[col0 * n + 0, 0] = 1; W[col1 * n + 1, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0 * p + col0, 1] = 1
    W[col0 * n + 1, 1] = 1; W[col1 * n + 1, 1] = -1

    U[0, 2] = 1
    V[0 * p + col1, 2] = 1; V[1 * p + col1, 2] = -1
    W[col1 * n + 0, 2] = 1; W[col1 * n + 1, 2] = 1

    U[3, 3] = 1
    V[1 * p + col0, 3] = 1; V[0 * p + col0, 3] = -1
    W[col0 * n + 0, 3] = 1; W[col0 * n + 1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[1 * p + col1, 4] = 1
    W[col0 * n + 0, 4] = -1; W[col1 * n + 0, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[0 * p + col0, 5] = 1; V[0 * p + col1, 5] = 1
    W[col1 * n + 1, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[1 * p + col0, 6] = 1; V[1 * p + col1, 6] = 1
    W[col0 * n + 0, 6] = 1

    return U, V, W

def _tensor_residual(U, V, W, T):
    return T - np.einsum('ur,vr,wr->uvw', U, V, W)

def _als_refine(U, V, W, T, R, n, m, p, max_iter=100, tol=1e-12, deadline=None):
    prev_err = np.inf
    best_U, best_V, best_W, best_err = U.copy(), V.copy(), W.copy(), np.inf

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n * m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol

        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m * p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol

        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p * n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol

        err = np.linalg.norm(_tensor_residual(U, V, W, T))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return best_U, best_V, best_W, best_err

def _scipy_polish(U, V, W, T, R, n, m, p, deadline=None):
    # Flatten
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return np.inf
        U_t = x[:n*m*R].reshape(n*m, R)
        V_t = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W_t = x[n*m*R + m*p*R :].reshape(p*n, R)
        res = _tensor_residual(U_t, V_t, W_t, T)
        return np.sum(res**2)

    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < 1e-8:
            x_opt = res.x
            U_opt = x_opt[:n*m*R].reshape(n*m, R)
            V_opt = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W_opt = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
            err = np.linalg.norm(_tensor_residual(U_opt, V_opt, W_opt, T))
            return U_opt, V_opt, W_opt, err
    except Exception:
        pass
    return U, V, W, np.linalg.norm(_tensor_residual(U, V, W, T))

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    result = np.einsum('ur,vr,wr->uvw', U, V, W)
    if not np.allclose(result, T, atol=1e-6):
        return None

    for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom

        result_s = np.einsum('ur,vr,wr->uvw', U_s, V_s, W_s)
        if np.allclose(result_s, T, atol=1e-5):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]

            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-8:
                            mat[i][j] = int(round(mat[i][j]))

            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    return None

def _structured_init(n, m, p, R, strategy, trial):
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    U_str01, V_str01, W_str01 = _get_strassen_block_np(n, m, p, 0, 1)
    U_str23, V_str23, W_str23 = _get_strassen_block_np(n, m, p, 2, 3)
    
    scale = 0.3 + 0.1 * (trial % 5)

    if strategy == 0:
        if R >= 7:
            U[:, :7] = U_str01
            V[:, :7] = V_str01
            W[:, :7] = W_str01
        if R > 7:
            U[:, 7:] = np.random.randn(n * m, R - 7) * scale
            V[:, 7:] = np.random.randn(m * p, R - 7) * scale
            W[:, 7:] = np.random.randn(p * n, R - 7) * scale
    elif strategy == 1:
        U = np.random.randn(n * m, R) * scale
        V = np.random.randn(m * p, R) * scale
        W = np.random.randn(p * n, R) * scale
    elif strategy == 2:
        if R >= 7:
            U[:, :7] = U_str01
            V[:, :7] = V_str01
            W[:, :7] = W_str01
        if R >= 14:
            U[:, 7:14] = U_str23
            V[:, 7:14] = V_str23
            W[:, 7:14] = W_str23
        if R > 14:
            U[:, 14:] = np.random.randn(n * m, R - 14) * 0.2
            V[:, 14:] = np.random.randn(m * p, R - 14) * 0.2
            W[:, 14:] = np.random.randn(p * n, R - 14) * 0.2
    else:
        U = np.random.randn(n * m, R) * scale
        V = np.random.randn(m * p, R) * scale
        W = np.random.randn(p * n, R) * scale
        if R >= 7:
            U[:, :7] += U_str01 * 0.5
            V[:, :7] += V_str01 * 0.5
            W[:, :7] += W_str01 * 0.5

    return U, V, W

def _search_rank(n, m, p, R, timeout=20, n_trials=100):
    start_time = time.time()
    deadline = start_time + timeout
    T = _build_tensor(n, m, p)

    for trial in range(n_trials):
        if time.time() > deadline:
            break

        strategy = trial % 4
        U, V, W = _structured_init(n, m, p, R, strategy, trial)

        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=50, deadline=deadline)
        
        if err < 1e-4:
            U, V, W, err = _scipy_polish(U, V, W, T, R, n, m, p, deadline=deadline)

        if err < 1e-8:
            result = _check_and_rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result

    return None

def _construct_rank18(n=2, m=2, p=5):
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    U[:, :7] = U_str
    V[:, :7] = V_str
    W[:, :7] = W_str

    U_str2, V_str2, W_str2 = _get_strassen_block_np(n, m, p, 2, 3)
    U[:, 7:14] = U_str2
    V[:, 7:14] = V_str2
    W[:, 7:14] = W_str2

    U[0, 14] = 1; V[0 * p + 4, 14] = 1; W[4 * n + 0, 14] = 1
    U[1, 15] = 1; V[1 * p + 4, 15] = 1; W[4 * n + 0, 15] = 1
    U[2, 16] = 1; V[0 * p + 4, 16] = 1; W[4 * n + 1, 16] = 1
    U[3, 17] = 1; V[1 * p + 4, 17] = 1; W[4 * n + 1, 17] = 1

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    if not _verify(U_list, V_list, W_list, n, m, p):
        raise ValueError("Rank-18 construction failed verification")

    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT

    # Try Rank 17
    remaining = deadline - time.time() - 40
    if remaining > 10:
        t17_budget = min(40, remaining)
        result = _search_rank(n, m, p, 17, timeout=t17_budget, n_trials=150)
        if result is not None:
            return result

    # Try Rank 16
    remaining = deadline - time.time() - 30
    if remaining > 10:
        t16_budget = min(30, remaining)
        result = _search_rank(n, m, p, 16, timeout=t16_budget, n_trials=150)
        if result is not None:
            return result

    # Try Rank 15
    remaining = deadline - time.time() - 20
    if remaining > 10:
        t15_budget = min(20, remaining)
        result = _search_rank(n, m, p, 15, timeout=t15_budget, n_trials=100)
        if result is not None:
            return result

    # Fallback Rank 18 (Fast and Guaranteed)
    return _construct_rank18(n, m, p)
