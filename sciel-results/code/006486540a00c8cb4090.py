import numpy as np
import time

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140.0

    # Try to find rank 50 decomposition (5 blocks of 2x2x3 rank 10)
    result = _try_rank50(deadline)
    if result is not None:
        return result

    # Fallback to rank 53
    return _build_rank53()

def _build_rank53():
    n, m, p = 2, 2, 15
    rank = 53

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    U_sc = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_sc = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_sc = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_s[i*2 + j][r]
            for j in range(2):
                V[j*p + k0][c] = V_s[j*2 + 0][r]
                V[j*p + k1][c] = V_s[j*2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_s[k*2 + i][r]
        col += 7

    k_last = 14
    for r in range(4):
        c = col + r
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = U_sc[i*2 + j][r]
        for j in range(2):
            V[j*p + k_last][c] = V_sc[j][r]
        for i in range(2):
            W[k_last*n + i][c] = W_sc[i][r]

    return {"U": U, "V": V, "W": W}

def _try_rank50(deadline):
    n, m, p = 2, 2, 15
    rank_block = 10
    num_blocks = 5
    total_rank = rank_block * num_blocks

    # Find a rank 10 decomposition for 2x2x3
    U_b, V_b, W_b = _find_m223_rank10(deadline)
    if U_b is None:
        return None

    # Assemble 5 blocks
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for b in range(num_blocks):
        k0 = 3 * b
        col_offset = b * rank_block
        for r in range(rank_block):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_b[i*2 + j][r]
            for j in range(2):
                for k in range(3):
                    kk = k0 + k
                    V[j*p + kk][c] = V_b[j*3 + k][r]
            for k in range(3):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_b[k*2 + i][r]

    return {"U": U, "V": V, "W": W}

def _find_m223_rank10(deadline):
    n, m, p = 2, 2, 3
    rank = 10

    # Residual function for 2x2x3
    def residuals(params):
        U = params[:4*rank].reshape(4, rank)
        V = params[4*rank:4*rank+6*rank].reshape(6, rank)
        W = params[4*rank+6*rank:].reshape(6, rank)

        res = np.zeros(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    res[idx] = np.dot(U[i*2+j], V[j*3+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res

    # Try multiple seeds with improved initialization strategies
    for seed in range(100):
        if time.time() > deadline - 20:
            break

        np.random.seed(seed)

        # Different initialization strategies based on seed
        if seed % 4 == 0:
            # Small random values
            x0 = np.random.randn(16*rank) * 0.5
        elif seed % 4 == 1:
            # Larger random values
            x0 = np.random.randn(16*rank) * 1.0
        elif seed % 4 == 2:
            # Sparse initialization - mostly zeros with some ones
            x0 = np.zeros(16*rank)
            for _ in range(16*rank // 3):
                idx = np.random.randint(0, 16*rank)
                x0[idx] = np.random.choice([-1, 1])
        else:
            # Mixed initialization
            x0 = np.random.randn(16*rank) * 0.3
            for idx in range(0, 16*rank, 4):
                x0[idx] = np.random.choice([-1, 0, 1])

        try:
            # Use Levenberg-Marquardt with more iterations
            result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if np.linalg.norm(result.fun) < 1e-8:
                U = result.x[:4*rank].reshape(4, rank)
                V = result.x[4*rank:4*rank+6*rank].reshape(6, rank)
                W = result.x[4*rank+6*rank:].reshape(6, rank)

                # Try snapping to different grids
                for scale in [2, 4, 8]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale

                    # Verify
                    res_r = residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
                    if np.max(np.abs(res_r)) < 1e-6:
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()

                # If no grid snap works, try to polish with tighter tolerance
                result2 = least_squares(residuals, result.x, method='lm', max_nfev=10000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                if np.linalg.norm(result2.fun) < 1e-10:
                    U = result2.x[:4*rank].reshape(4, rank)
                    V = result2.x[4*rank:4*rank+6*rank].reshape(6, rank)
                    W = result2.x[4*rank+6*rank:].reshape(6, rank)

                    for scale in [2, 4, 8]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale

                        res_r = residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
                        if np.max(np.abs(res_r)) < 1e-6:
                            return U_r.tolist(), V_r.tolist(), W_r.tolist()

        except Exception:
            continue

    return None, None, None

from scipy.optimize import least_squares
