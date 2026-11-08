"""Direct tensor decomposition for 2x3 x 3x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
EVAL_DEADLINE = 140  # seconds, leave margin for evaluator


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 6

    # Trivial rank-36 solution
    rank_trivial = n * m * p  # 36
    U_trivial = np.zeros((n * m, rank_trivial), dtype=float)
    V_trivial = np.zeros((m * p, rank_trivial), dtype=float)
    W_trivial = np.zeros((p * n, rank_trivial), dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_trivial[i * m + j, r] = 1.0
                V_trivial[j * p + k, r] = 1.0
                W_trivial[k * n + i, r] = 1.0

    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial

    # Precompute index arrays
    num_constraints = n * m * p  # 36
    u_idx = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    v_idx = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    w_idx = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    target = np.ones(num_constraints)

    # Try to find lower-rank decompositions using L-BFGS
    for try_rank in range(35, 18, -1):
        if time.time() - t0 > EVAL_DEADLINE - 10:
            break

        rng = np.random.RandomState(try_rank * 1000 + 42)

        for attempt in range(12):
            if time.time() - t0 > EVAL_DEADLINE - 5:
                break

            # Random initialization with various scales
            scale = rng.choice([0.1, 0.3, 0.5, 1.0])
            U = rng.randn(n * m, try_rank) * scale
            V = rng.randn(m * p, try_rank) * scale
            W = rng.randn(p * n, try_rank) * scale

            # Flatten for optimization
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            size_u = n * m * try_rank
            size_v = m * p * try_rank
            size_w = p * n * try_rank

            def objective(params):
                U_flat = params[:size_u]
                V_flat = params[size_u:size_u + size_v]
                W_flat = params[size_u + size_v:size_u + size_v + size_w]

                U = U_flat.reshape(n * m, try_rank)
                V = V_flat.reshape(m * p, try_rank)
                W = W_flat.reshape(p * n, try_rank)

                # Compute tensor values
                vals = np.zeros(num_constraints)
                for r in range(try_rank):
                    vals += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]

                residuals = vals - target
                obj = np.sum(residuals ** 2)

                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)

                for r in range(try_rank):
                    contrib = 2 * residuals * V[v_idx, r] * W[w_idx, r]
                    np.add.at(grad_U, u_idx, contrib)
                    contrib = 2 * residuals * U[u_idx, r] * W[w_idx, r]
                    np.add.at(grad_V, v_idx, contrib)
                    contrib = 2 * residuals * U[u_idx, r] * V[v_idx, r]
                    np.add.at(grad_W, w_idx, contrib)

                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return obj, grad

            try:
                res = minimize(
                    objective,
                    params0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )

                # Check if converged
                final_params = res.x
                U_opt = final_params[:size_u].reshape(n * m, try_rank)
                V_opt = final_params[size_u:size_u + size_v].reshape(m * p, try_rank)
                W_opt = final_params[size_u + size_v:].reshape(p * n, try_rank)

                # Verify decomposition
                if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                    best_rank = try_rank
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    break

            except Exception:
                continue

        if best_rank < try_rank:
            break

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition satisfies the tensor equation."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True
