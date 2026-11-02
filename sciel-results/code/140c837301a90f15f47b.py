"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matmul tensor T[i,j,k]=1."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def residuals(params, R):
        """Compute residuals for a given parameter vector and rank R."""
        U = params[:6 * R].reshape(6, R)
        V = params[6 * R:6 * R + 21 * R].reshape(21, R)
        W = params[6 * R + 21 * R:].reshape(14, R)
        pred = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred[i, j, k] = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
        return (pred - 1.0).flatten()

    def try_rank(R, max_seeds=30, max_iter=15000):
        """Try to find a rank-R half-integer decomposition via numerical optimization."""
        if time.time() > deadline - 20:
            return None

        n_params = 6 * R + 21 * R + 14 * R  # U(6xR) + V(21xR) + W(14xR)

        try:
            from scipy.optimize import least_squares
        except ImportError:
            return None

        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                return None

            np.random.seed(seed)
            # Use various scales for initialization
            scale = 0.1 + 0.1 * (seed % 5)
            x0 = np.random.randn(n_params) * scale

            try:
                result = least_squares(residuals, x0, args=(R,), method='lm', max_nfev=max_iter)
            except Exception:
                continue

            if result.cost < 1e-12:
                U_sol = result.x[:6 * R].reshape(6, R)
                V_sol = result.x[6 * R:6 * R + 21 * R].reshape(21, R)
                W_sol = result.x[6 * R + 21 * R:].reshape(14, R)

                # Try rounding to nearest half-integer
                U_rounded = np.round(U_sol * 2) / 2.0
                V_rounded = np.round(V_sol * 2) / 2.0
                W_rounded = np.round(W_sol * 2) / 2.0

                # Verify exactness
                U_list = [[float(U_rounded[i, r]) for r in range(R)] for i in range(6)]
                V_list = [[float(V_rounded[i, r]) for r in range(R)] for i in range(21)]
                W_list = [[float(W_rounded[i, r]) for r in range(R)] for i in range(14)]

                if verify_decomposition(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}

                # If direct rounding failed, try nearby half-integer grid
                # Only for small R to keep it tractable
                if R <= 16:
                    # Try small perturbations
                    for du in [-0.5, 0.0, 0.5]:
                        for dv in [-0.5, 0.0, 0.5]:
                            for dw in [-0.5, 0.0, 0.5]:
                                U_try = np.round((U_sol + du) * 2) / 2.0
                                V_try = np.round((V_sol + dv) * 2) / 2.0
                                W_try = np.round((W_sol + dw) * 2) / 2.0
                                U_l = [[float(U_try[i, r]) for r in range(R)] for i in range(6)]
                                V_l = [[float(V_try[i, r]) for r in range(R)] for i in range(21)]
                                W_l = [[float(W_try[i, r]) for r in range(R)] for i in range(14)]
                                if verify_decomposition(U_l, V_l, W_l, R):
                                    return {"U": U_l, "V": V_l, "W": W_l}

        return None

    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(6, 21, 14) = 21
    # But we can try lower ranks too
    for R in range(14, 43):
        if time.time() > deadline - 10:
            break
        result = try_rank(R)
        if result is not None:
            return result

    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
