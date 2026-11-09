"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Surrogate-assisted optimizer using RBF interpolation + L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit, keep safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial exploration: sample points to build initial surrogate
        # Use a mix of center, corners, and random points
        points = []
        values = []

        # Center
        x0 = np.zeros(dim)
        points.append(x0)
        values.append(func(x0))

        # Random points for initial spread
        n_initial = min(20, budget - 1)
        for _ in range(n_initial):
            if len(values) >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            points.append(x)
            values.append(func(x))

        points = np.array(points)
        values = np.array(values)
        calls_made = len(values)

        best_x = points[np.argmin(values)]
        best_f = np.min(values)

        # RBF kernel function (Gaussian)
        def rbf_kernel(X, Y, length_scale=1.0):
            # X: (n, dim), Y: (m, dim)
            # Returns (n, m) kernel matrix
            dx2 = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :]
            dx2 = np.maximum(dx2 - 2 * X @ Y.T, 0)
            return np.exp(-dx2 / (2 * length_scale**2))

        def fit_rbf(X, y, length_scale=1.0):
            """Fit RBF interpolant. Returns lambda weights."""
            n = len(X)
            K = rbf_kernel(X, X, length_scale)
            # Add small regularization for numerical stability
            K += 1e-8 * np.eye(n)
            try:
                w = np.linalg.solve(K, y)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(K, y, rcond=None)[0]
            return w

        def rbf_predict(X_new, X_train, w, length_scale=1.0):
            """Predict using fitted RBF."""
            K_new = rbf_kernel(X_new, X_train, length_scale)
            return K_new @ w

        def rbf_gradient(x, X_train, w, length_scale=1.0):
            """Compute gradient of RBF at point x."""
            # d/dx [exp(-||x - X_j||^2 / (2*l^2))] = -exp(...) * (x - X_j) / l^2
            diffs = x[None, :] - X_train  # (n, dim)
            dx2 = np.sum(diffs**2, axis=1)  # (n,)
            kernel_vals = np.exp(-dx2 / (2 * length_scale**2))  # (n,)
            # Gradient: sum_j w_j * kernel_j * (-diffs_j / l^2)
            # = -(1/l^2) * sum_j w_j * kernel_j * diffs_j
            weights = w * kernel_vals  # (n,)
            grad = -np.sum(weights[:, None] * diffs, axis=0) / (length_scale**2)
            return grad

        # Adaptive length scale based on data spread
        def estimate_length_scale(X):
            if len(X) < 2:
                return 1.0
            # Use median of pairwise distances
            n = len(X)
            # Sample pairs for efficiency
            max_pairs = min(n * (n - 1) // 2, 1000)
            if n * (n - 1) // 2 <= max_pairs:
                idx_i, idx_j = np.triu_indices(n, k=1)
                dists = np.linalg.norm(X[idx_i] - X[idx_j], axis=1)
            else:
                idx_i = rng.integers(0, n, max_pairs)
                idx_j = rng.integers(0, n, max_pairs)
                mask = idx_i != idx_j
                idx_i, idx_j = idx_i[mask], idx_j[mask]
                dists = np.linalg.norm(X[idx_i] - X[idx_j], axis=1)
            return max(np.median(dists) * 0.5, 0.1)

        # Main optimization loop
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline - 2.0:
                break

            # Fit RBF surrogate
            ls = estimate_length_scale(points)
            try:
                w = fit_rbf(points, values, ls)
            except Exception:
                w = None

            if w is not None:
                # Define surrogate objective and gradient
                def surrogate_obj(x):
                    return float(rbf_predict(x.reshape(1, -1), points, w, ls)[0])

                def surrogate_grad(x):
                    return rbf_gradient(x, points, w, ls)

                # Try L-BFGS-B from best point and a few random starts
                candidates = [best_x.copy()]
                # Add a few random starts
                for _ in range(3):
                    candidates.append(rng.uniform(-5, 5, dim))

                for x_start in candidates:
                    if time.time() > deadline - 2.0:
                        break
                    if calls_made >= budget:
                        break

                    try:
                        res = minimize(
                            surrogate_obj,
                            x_start,
                            jac=surrogate_grad,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 50, 'ftol': 1e-8}
                        )
                        x_prop = np.clip(res.x, -5, 5)
                    except Exception:
                        continue

                    # Evaluate the true function at the proposed point
                    if calls_made >= budget:
                        break
                    f_prop = func(x_prop)
                    calls_made += 1
                    points = np.vstack([points, x_prop])
                    values = np.append(values, f_prop)

                    if f_prop < best_f:
                        best_f = f_prop
                        best_x = x_prop.copy()

            else:
                # Fallback: simple random step
                if calls_made >= budget:
                    break
                x_prop = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                f_prop = func(x_prop)
                calls_made += 1
                points = np.vstack([points, x_prop])
                values = np.append(values, f_prop)
                if f_prop < best_f:
                    best_f = f_prop
                    best_x = x_prop.copy()

            # If we've used a lot of budget, do some local search around best
            if calls_made > budget * 0.7:
                # Fine-grained local search
                for _ in range(min(5, budget - calls_made)):
                    if time.time() > deadline - 2.0:
                        break
                    if calls_made >= budget:
                        break
                    step_size = 0.1 * (1 - (calls_made - budget * 0.7) / (budget * 0.3))
                    x_prop = np.clip(best_x + step_size * rng.standard_normal(dim), -5, 5)
                    f_prop = func(x_prop)
                    calls_made += 1
                    points = np.vstack([points, x_prop])
                    values = np.append(values, f_prop)
                    if f_prop < best_f:
                        best_f = f_prop
                        best_x = x_prop.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
