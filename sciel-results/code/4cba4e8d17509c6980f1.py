import time
import numpy as np

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
    """Compute residual tensor T - U·V·W."""
    return T - np.einsum('ur,vr,wr->uvw', U, V, W)


def _als_refine(U, V, W, T, R, n, m, p, max_iter=200, tol=1e-14, deadline=None,
                smooth=False, smooth_beta=0.1):
    """ALS refinement with optional smooth surrogate and high-precision polish."""
    prev_err = np.inf
    best_U, best_V, best_W, best_err = U.copy(), V.copy(), W.copy(), np.inf

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n * m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol

        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m * p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol

        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p * n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol

        # Smooth surrogate: add small regularization toward previous best if err stalls
        if smooth and it > 20 and prev_err - np.linalg.norm(_tensor_residual(U, V, W, T)) < 1e-12:
            U = (1 - smooth_beta) * U + smooth_beta * best_U
            V = (1 - smooth_beta) * V + smooth_beta * best_V
            W = (1 - smooth_beta) * W + smooth_beta * best_W

        err = np.linalg.norm(_tensor_residual(U, V, W, T))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    # High-precision polish: tighter tolerance, more iterations
    if best_err < 1e-6 and deadline is not None and time.time() < deadline - 2:
        U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        for it in range(50):
            if time.time() > deadline - 1:
                break
            VvW = np.einsum('vr,wr->vwr', V, W)
            VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
            for u in range(n * m):
                b = T[u].reshape(-1)
                sol, *_ = np.linalg.lstsq(VvW_flat, b, rcond=None)
                U[u] = sol
            Uw = np.einsum('ur,wr->uwr', U, W)
            Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
            for v in range(m * p):
                b = T[:, v, :].reshape(-1)
                sol, *_ = np.linalg.lstsq(Uw_flat, b, rcond=None)
                V[v] = sol
            UV = np.einsum('ur,vr->uvr', U, V)
            UV_flat = UV.reshape(-1, R)
            for w in range(p * n):
                b = T[:, :, w].reshape(-1)
                sol, *_ = np.linalg.lstsq(UV_flat, b, rcond=None)
                W[w] = sol
            err = np.linalg.norm(_tensor_residual(U, V, W, T))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-15:
                break

    return best_U, best_V, best_W, best_err


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
    U_str02, V_str02, W_str02 = _get_strassen_block_np(n, m, p, 0, 2)
    U_str13, V_str13, W_str13 = _get_strassen_block_np(n, m, p, 1, 3)
    U_str14, V_str14, W_str14 = _get_strassen_block_np(n, m, p, 1, 4)
    U_str04, V_str04, W_str04 = _get_strassen_block_np(n, m, p, 0, 4)

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
    elif strategy == 3:
        if R >= 7:
            U[:, :7] = U_str02
            V[:, :7] = V_str02
            W[:, :7] = W_str02
        if R >= 14:
            U[:, 7:14] = U_str14
            V[:, 7:14] = V_str14
            W[:, 7:14] = W_str14
        if R > 14:
            U[:, 14:] = np.random.randn(n * m, R - 14) * 0.2
            V[:, 14:] = np.random.randn(m * p, R - 14) * 0.2
            W[:, 14:] = np.random.randn(p * n, R - 14) * 0.2
    elif strategy == 4:
        if R >= 7:
            U[:, :7] = U_str01
            V[:, :7] = V_str01
            W[:, :7] = W_str01
        if R >= 14:
            U[:, 7:14] = U_str13
            V[:, 7:14] = V_str13
            W[:, 7:14] = W_str13
        if R > 14:
            U[:, 14:] = np.random.randn(n * m, R - 14) * 0.2
            V[:, 14:] = np.random.randn(m * p, R - 14) * 0.2
            W[:, 14:] = np.random.randn(p * n, R - 14) * 0.2
    elif strategy == 5:
        U_str12, V_str12, W_str12 = _get_strassen_block_np(n, m, p, 1, 2)
        if R >= 7:
            U[:, :7] = U_str04
            V[:, :7] = V_str04
            W[:, :7] = W_str04
        if R >= 14:
            U[:, 7:14] = U_str12
            V[:, 7:14] = V_str12
            W[:, 7:14] = W_str12
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

        strategy = trial % 7
        smooth = (trial % 3 == 0)  # Use smooth surrogate on some trials
        U, V, W = _structured_init(n, m, p, R, strategy, trial)

        max_iter = 150 if trial < 20 else 100
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p,
                                    max_iter=max_iter, deadline=deadline,
                                    smooth=smooth, smooth_beta=0.05)

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


def _try_perturb_rank(n, m, p, target_rank, timeout=15, n_trials=80):
    start_time = time.time()
    deadline = start_time + timeout
    T = _build_tensor(n, m, p)

    base_U, base_V, base_W = _get_strassen_block_np(n, m, p, 0, 1)
    base_U2, base_V2, base_W2 = _get_strassen_block_np(n, m, p, 2, 3)

    full_U = np.zeros((n * m, 18))
    full_V = np.zeros((m * p, 18))
    full_W = np.zeros((p * n, 18))
    full_U[:, :7] = base_U
    full_V[:, :7] = base_V
    full_W[:, :7] = base_W
    full_U[:, 7:14] = base_U2
    full_V[:, 7:14] = base_V2
    full_W[:, 7:14] = base_W2
    full_U[0, 14] = 1; full_V[0 * p + 4, 14] = 1; full_W[4 * n + 0, 14] = 1
    full_U[1, 15] = 1; full_V[1 * p + 4, 15] = 1; full_W[4 * n + 0, 15] = 1
    full_U[2, 16] = 1; full_V[0 * p + 4, 16] = 1; full_W[4 * n + 1, 16] = 1
    full_U[3, 17] = 1; full_V[1 * p + 4, 17] = 1; full_W[4 * n + 1, 17] = 1

    for trial in range(n_trials):
        if time.time() > deadline:
            break

        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))

        if trial % 3 == 0:
            indices = np.random.choice(18, target_rank, replace=False)
            U[:, :] = full_U[:, indices]
            V[:, :] = full_V[:, indices]
            W[:, :] = full_W[:, indices]
        elif trial % 3 == 1:
            U[:, :] = full_U[:, :target_rank]
            V[:, :] = full_V[:, :target_rank]
            W[:, :] = full_W[:, :target_rank]
            U += np.random.randn(n * m, target_rank) * 0.05
            V += np.random.randn(m * p, target_rank) * 0.05
            W += np.random.randn(p * n, target_rank) * 0.05
        else:
            n_from_first = min(target_rank, 7)
            n_from_second = min(target_rank - n_from_first, 7)
            n_random = target_rank - n_from_first - n_from_second

            U[:, :n_from_first] = full_U[:, :n_from_first]
            V[:, :n_from_first] = full_V[:, :n_from_first]
            W[:, :n_from_first] = full_W[:, :n_from_first]

            if n_from_second > 0:
                U[:, n_from_first:n_from_first + n_from_second] = full_U[:, 7:7 + n_from_second]
                V[:, n_from_first:n_from_first + n_from_second] = full_V[:, 7:7 + n_from_second]
                W[:, n_from_first:n_from_first + n_from_second] = full_W[:, 7:7 + n_from_second]

            if n_random > 0:
                U[:, n_from_first + n_from_second:] = np.random.randn(n * m, n_random) * 0.3
                V[:, n_from_first + n_from_second:] = np.random.randn(m * p, n_random) * 0.3
                W[:, n_from_first + n_from_second:] = np.random.randn(p * n, n_random) * 0.3

        smooth = (trial % 2 == 0)
        U, V, W, err = _als_refine(U, V, W, T, target_rank, n, m, p,
                                    max_iter=120, deadline=deadline,
                                    smooth=smooth, smooth_beta=0.05)

        if err < 1e-8:
            result = _check_and_rationalize(U, V, W, T, target_rank, n, m, p)
            if result is not None:
                return result
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT

    # Try Rank 14
    remaining = deadline - time.time() - 30
    if remaining > 10:
        t14_budget = min(18, remaining)
        result = _search_rank(n, m, p, 14, timeout=t14_budget, n_trials=80)
        if result is not None:
            return result

    # Try Rank 15
    remaining = deadline - time.time() - 25
    if remaining > 10:
        t15_budget = min(16, remaining)
        result = _search_rank(n, m, p, 15, timeout=t15_budget, n_trials=100)
        if result is not None:
            return result

    # Try Rank 16
    remaining = deadline - time.time() - 20
    if remaining > 10:
        t16_budget = min(14, remaining)
        result = _search_rank(n, m, p, 16, timeout=t16_budget, n_trials=80)
        if result is not None:
            return result

    # Try Rank 17 via perturbation
    remaining = deadline - time.time() - 15
    if remaining > 8:
        t17_budget = min(12, remaining)
        result = _try_perturb_rank(n, m, p, 17, timeout=t17_budget, n_trials=80)
        if result is not None:
            return result

    # Try Rank 17 - standard search
    remaining = deadline - time.time() - 12
    if remaining > 6:
        t17_budget = min(10, remaining)
        result = _search_rank(n, m, p, 17, timeout=t17_budget, n_trials=60)
        if result is not None:
            return result

    # Fallback Rank 18 (Fast and Guaranteed)
    return _construct_rank18(n, m, p)
