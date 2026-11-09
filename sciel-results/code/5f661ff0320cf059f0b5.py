"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Surrogate-assisted optimizer with coordinate descent refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_used = 0

        def call_func(x):
            nonlocal calls_used
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            calls_used += 1
            return x, val

        # Phase 1: Initial sampling to build a rough picture
        # Use a Sobol-like or stratified sampling for initial points
        n_init = min(30, budget // 5)
        points = []
        values = []

        # Start with center
        x0 = np.zeros(dim)
        x0, f0 = call_func(x0)
        points.append(x0.copy())
        values.append(f0)

        # Add stratified samples
        for _ in range(n_init - 1):
            x = rng.uniform(-5, 5, dim)
            x, f = call_func(x)
            points.append(x.copy())
            values.append(f)

        points = np.array(points)
        values = np.array(values)
        best_idx = np.argmin(values)
        x_best = points[best_idx].copy()
        f_best = values[best_idx]

        # Phase 2: Surrogate-assisted search using RBF
        # We'll use a simple RBF model with Gaussian kernel
        # Fit the model and use its gradient to propose new points

        def rbf_predict(X, centers, weights, lengths):
            """Predict values using RBF model."""
            # X: (n, d), centers: (m, d), weights: (m,), lengths: (m,)
            diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist2 = np.sum(diffs ** 2, axis=2)
            # Gaussian kernel: exp(-dist2 / (2 * length^2))
            K = np.exp(-dist2 / (2.0 * lengths[np.newaxis, :] ** 2))
            return K @ weights

        def rbf_gradient(x, centers, weights, lengths):
            """Compute gradient of RBF model at point x."""
            # x: (d,), returns (d,)
            diffs = centers - x  # (m, d)
            dist2 = np.sum(diffs ** 2, axis=1)  # (m,)
            K = np.exp(-dist2 / (2.0 * lengths ** 2))  # (m,)
            # dK/dx = K * (diffs / length^2)
            grad_K = K[:, np.newaxis] * (diffs / lengths[:, np.newaxis] ** 2)
            return grad_K @ weights

        # Adaptively fit RBF and use L-BFGS-B on surrogate
        n_surrogate_steps = min(20, budget // 10)
        for step in range(n_surrogate_steps):
            if time.time() > deadline - 30:
                break
            if calls_used >= budget - 5:
                break

            # Fit RBF model to current data
            m = len(points)
            if m < dim + 2:
                # Not enough data, add more random points
                for _ in range(dim + 2 - m):
                    x = rng.uniform(-5, 5, dim)
                    x, f = call_func(x)
                    points = np.vstack([points, x])
                    values = np.append(values, f)
                continue

            # Use all points for fitting
            centers = points
            # Initialize lengths based on data spread
            lengths = np.full(m, 1.0)
            # Simple approach: use median distance
            if m > 1:
                dists = []
                for i in range(m):
                    d = np.linalg.norm(points - points[i], axis=1)
                    d = d[d > 1e-10]
                    if len(d) > 0:
                        dists.append(np.median(d))
                if dists:
                    lengths = np.full(m, max(0.5, np.median(dists)))

            # Solve for weights: K @ weights = values
            K = np.exp(-np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2) / (2.0 * lengths[:, np.newaxis] ** 2))
            # Add small regularization
            K_reg = K + 1e-6 * np.eye(m)
            try:
                weights = np.linalg.solve(K_reg, values)
            except np.linalg.LinAlgError:
                weights = np.linalg.lstsq(K_reg, values, rcond=None)[0]

            # Use L-BFGS-B on the surrogate to find a promising point
            def surrogate(x):
                return rbf_predict(x.reshape(1, -1), centers, weights, lengths)[0]

            def surrogate_grad(x):
                return rbf_gradient(x, centers, weights, lengths)

            # Start from current best
            x_start = x_best.copy()
            res = minimize(
                surrogate,
                x_start,
                jac=surrogate_grad,
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 50, 'gtol': 1e-6}
            )

            x_proposed = np.clip(res.x, -5, 5)
            x_proposed, f_proposed = call_func(x_proposed)

            # Add to data
            points = np.vstack([points, x_proposed])
            values = np.append(values, f_proposed)

            if f_proposed < f_best:
                x_best = x_proposed.copy()
                f_best = f_proposed

        # Phase 3: Coordinate descent refinement on the best point
        # This exploits the anisotropic nature of Bent Cigar
        if calls_used < budget - 10 and time.time() < deadline - 10:
            # Adaptive coordinate descent
            step_sizes = np.ones(dim) * 0.5
            x_current = x_best.copy()
            f_current = f_best

            cd_iterations = min(100, (budget - calls_used - 5) // 2)
            for it in range(cd_iterations):
                if time.time() > deadline - 5:
                    break
                if calls_used >= budget - 3:
                    break

                improved = False
                for d in range(dim):
                    if time.time() > deadline - 5:
                        break
                    if calls_used >= budget - 3:
                        break

                    # Try positive and negative steps
                    for direction in [1, -1]:
                        x_trial = x_current.copy()
                        x_trial[d] += direction * step_sizes[d]
                        x_trial = np.clip(x_trial, -5, 5)

                        if np.allclose(x_trial, x_current):
                            continue

                        x_trial, f_trial = call_func(x_trial)

                        if f_trial < f_current:
                            x_current = x_trial
                            f_current = f_trial
                            step_sizes[d] *= 1.2
                            improved = True
                        else:
                            step_sizes[d] *= 0.7

                    if f_current < f_best:
                        x_best = x_current.copy()
                        f_best = f_current

                if not improved:
                    # No improvement in this full pass, shrink all steps
                    step_sizes *= 0.8
                    if np.all(step_sizes < 1e-6):
                        break

        return x_best, f_best
# EVOLVE-BLOCK-END
