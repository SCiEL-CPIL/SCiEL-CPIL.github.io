"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


class Optimizer:
    """Two-phase surrogate-assisted strategy: GP exploration + L-BFGS-B exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill

        d = self.dim
        budget = self.budget

        # Phase 1: Global exploration with Gaussian Process surrogate
        # Use first ~60 evaluations to build a surrogate model
        n_explore = min(60, budget // 5)
        n_exploit = budget - n_explore

        # Initialize with a diverse set of points
        rng = np.random.default_rng(42)

        # Start with a stratified set of initial points
        points = []
        values = []

        # Add center and some structured points
        initial_candidates = [
            np.zeros(d),
            np.full(d, 2.0),
            np.full(d, -2.0),
            np.full(d, 4.0),
            np.full(d, -4.0),
        ]
        for p in initial_candidates:
            p = np.clip(p, -5, 5)
            points.append(p.copy())
            values.append(func(p))

        # Add random points
        while len(points) < n_explore:
            if time.time() > deadline:
                break
            p = rng.uniform(-5, 5, d)
            points.append(p)
            values.append(func(p))

        points = np.array(points)
        values = np.array(values)

        # Fit a simple GP-like surrogate using RBF kernel
        # We'll use a simplified approach: kernel regression with RBF
        def rbf_kernel(X1, X2, length_scale=1.0):
            # Compute squared distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            return np.exp(-sq_dist / (2 * length_scale ** 2))

        # Try a few length scales and pick the best
        best_length_scale = 1.0
        best_score = -np.inf

        for ls in [0.5, 1.0, 2.0, 3.0]:
            K = rbf_kernel(points, points, ls)
            # Add small regularization
            K_reg = K + 1e-6 * np.eye(len(points))
            try:
                alpha = np.linalg.solve(K_reg, values)
                # Score: how well does the model predict the min region
                # We want to find where the predicted minimum is
                # Sample candidate minima
                n_candidates = 200
                cand = rng.uniform(-5, 5, (n_candidates, d))
                K_pred = rbf_kernel(cand, points, ls)
                pred_vals = K_pred @ alpha
                best_idx = np.argmin(pred_vals)
                score = pred_vals[best_idx]
                if score > best_score:
                    best_score = score
                    best_length_scale = ls
            except np.linalg.LinAlgError:
                continue

        # Find the most promising basin center from the surrogate
        ls = best_length_scale
        K = rbf_kernel(points, points, ls)
        K_reg = K + 1e-6 * np.eye(len(points))
        try:
            alpha = np.linalg.solve(K_reg, values)
        except np.linalg.LinAlgError:
            alpha = np.zeros(len(points))

        # Sample many candidate points and find the one with lowest predicted value
        n_candidates = 500
        cand = rng.uniform(-5, 5, (n_candidates, d))
        K_pred = rbf_kernel(cand, points, ls)
        pred_vals = K_pred @ alpha
        best_cand_idx = np.argmin(pred_vals)
        best_start = cand[best_cand_idx]

        # Phase 2: Local exploitation with L-BFGS-B from multiple starting points
        # Use the best surrogate prediction and also the best observed point
        best_observed_idx = np.argmin(values)
        best_observed = points[best_observed_idx].copy()

        # Generate a set of starting points for local optimization
        start_points = [best_start, best_observed]

        # Add a few more diverse starting points based on surrogate
        top_k = 5
        top_indices = np.argsort(pred_vals)[:top_k]
        for idx in top_indices:
            if idx != best_cand_idx:
                start_points.append(cand[idx].copy())

        # Remove duplicates (approximately)
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for us in unique_starts:
                if np.linalg.norm(sp - us) < 0.1:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)

        # Run L-BFGS-B from each starting point, allocating remaining budget
        n_starts = len(unique_starts)
        if n_starts == 0:
            n_starts = 1
            unique_starts = [np.zeros(d)]

        calls_per_start = max(1, n_exploit // n_starts)

        best_x = best_observed
        best_f = best_observed_idx and values[best_observed_idx] or values[0]

        # Track total calls made
        calls_made = len(points)

        for sp in unique_starts:
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Define the objective function that counts calls
            def obj(x, _calls=[0]):
                nonlocal calls_made
                if calls_made >= budget:
                    return 1e10
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                return fx

            try:
                result = minimize(
                    obj,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={'maxiter': calls_per_start, 'maxfun': calls_per_start}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
