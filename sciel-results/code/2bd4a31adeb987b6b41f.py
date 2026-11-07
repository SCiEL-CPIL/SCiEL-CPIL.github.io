"""Sparse Grid Interpolation + Adaptive Refinement optimizer for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Sparse Grid + Adaptive Refinement optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        history = []  # List of (x, f) tuples

        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            history.append((x.copy(), f_val))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Initial Exploration with Sobol ---
        n_sobol = min(48, budget // 3)
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_points_unit = sampler.random(n_sobol)
            sobol_points = sobol_points_unit * 10.0 - 5.0

            for x in sobol_points:
                if eval_count >= budget or time.time() > deadline:
                    break
                wrapped_func(x)
        except Exception:
            for _ in range(n_sobol):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-5.0, 5.0, dim)
                wrapped_func(x)

        # --- Phase 2: Adaptive Refinement with IDW Uncertainty ---
        # We maintain a set of observed points.
        # At each step, we select the next point by minimizing an uncertainty measure.
        # We use a combination of:
        # 1. Inverse Distance Weighted (IDW) variance estimate
        # 2. Distance to best point (exploration)
        # 3. Local density (exploitation)

        # Precompute for efficiency
        X_obs = np.array([x for x, f in history]) if history else np.array([best_x])
        y_obs = np.array([f for x, f in history]) if history else np.array([best_f])

        # IDW parameters
        idw_power = 2.0
        idw_eps = 1e-6

        def idw_variance(X_new, X_ref, y_ref):
            """Compute IDW variance estimate at X_new given reference points X_ref, y_ref."""
            # X_new: (n_new, dim), X_ref: (n_ref, dim)
            # Returns: (n_new,) array of variance estimates
            
            if len(X_ref) == 0:
                return np.ones(len(X_new)) * 100.0
            
            if len(X_ref) == 1:
                # Only one point: variance is high everywhere except at that point
                dists = np.linalg.norm(X_new - X_ref[0], axis=1)
                return 1.0 / (dists + idw_eps)
            
            # Compute distances from each new point to each reference point
            # dists[i, j] = distance between X_new[i] and X_ref[j]
            dists = np.linalg.norm(X_new[:, np.newaxis, :] - X_ref[np.newaxis, :, :], axis=2)
            
            # IDW weights
            weights = 1.0 / (dists ** idw_power + idw_eps)
            
            # Normalized weights
            weights_sum = weights.sum(axis=1, keepdims=True)
            weights_sum[weights_sum < 1e-12] = 1.0
            norm_weights = weights / weights_sum
            
            # IDW prediction
            pred = norm_weights @ y_ref
            
            # Variance estimate: based on weighted variance of y_ref
            # Higher when points are far apart (low weight concentration)
            weighted_var = (norm_weights ** 2) @ (y_ref ** 2) - pred ** 2
            weighted_var = np.maximum(weighted_var, 0.0)
            
            # Also add a term based on distance to nearest point
            min_dists = dists.min(axis=1)
            dist_term = min_dists ** 2 / 10.0
            
            return weighted_var + dist_term

        def select_next_point(X_obs, y_obs, n_candidates=200):
            """Select the next point to evaluate using IDW uncertainty."""
            if eval_count >= budget or time.time() > deadline:
                return None
            
            # Generate candidate points
            # Mix of:
            # 1. Random points
            # 2. Points near best
            # 3. Points in gaps (based on current observations)
            
            candidates = []
            
            # 1. Random points
            n_rand = n_candidates // 3
            candidates.append(rng.uniform(-5.0, 5.0, (n_rand, dim)))
            
            # 2. Points near best
            n_near = n_candidates // 3
            sigma = max(0.5, 2.0 / (eval_count ** 0.25))  # Adaptive sigma
            near_points = best_x[None, :] + rng.normal(0, sigma, (n_near, dim))
            near_points = np.clip(near_points, -5.0, 5.0)
            candidates.append(near_points)
            
            # 3. Points in gaps: sample from regions far from existing points
            n_gap = n_candidates - n_rand - n_near
            if n_gap > 0:
                # Use rejection sampling to find points far from existing
                gap_points = []
                attempts = 0
                max_attempts = n_gap * 10
                while len(gap_points) < n_gap and attempts < max_attempts:
                    attempts += 1
                    x_new = rng.uniform(-5.0, 5.0, dim)
                    if len(X_obs) > 0:
                        min_dist = np.linalg.norm(X_obs - x_new, axis=1).min()
                    else:
                        min_dist = 10.0
                    # Accept with probability proportional to distance
                    if rng.random() < min_dist / 5.0:
                        gap_points.append(x_new)
                if gap_points:
                    candidates.append(np.array(gap_points))
            
            all_candidates = np.vstack(candidates)
            
            # Compute IDW variance for each candidate
            variances = idw_variance(all_candidates, X_obs, y_obs)
            
            # Also compute a simple "expected improvement" proxy:
            # Lower predicted value + higher uncertainty = better
            # IDW prediction
            if len(X_obs) > 0:
                dists = np.linalg.norm(all_candidates[:, np.newaxis, :] - X_obs[np.newaxis, :, :], axis=2)
                weights = 1.0 / (dists ** idw_power + idw_eps)
                weights_sum = weights.sum(axis=1, keepdims=True)
                weights_sum[weights_sum < 1e-12] = 1.0
                norm_weights = weights / weights_sum
                pred = norm_weights @ y_obs
            else:
                pred = np.zeros(len(all_candidates))
            
            # Score: lower is better
            # Combine predicted value and uncertainty
            score = pred + 0.5 * np.sqrt(variances + 1e-8)
            
            # Select the point with lowest score
            idx = np.argmin(score)
            return all_candidates[idx]

        # Adaptive refinement loop
        max_refine = budget - eval_count
        for i in range(max_refine):
            if eval_count >= budget or time.time() > deadline:
                break
            
            # Update observation arrays
            if history:
                X_obs = np.array([x for x, f in history])
                y_obs = np.array([f for x, f in history])
            else:
                X_obs = np.array([best_x])
                y_obs = np.array([best_f])
            
            # Select next point
            x_next = select_next_point(X_obs, y_obs, n_candidates=150)
            
            if x_next is None:
                break
            
            # Evaluate
            f_val = wrapped_func(x_next)
            
            # If we found a significantly better point, do local refinement
            if f_val < best_f - 1e-6 and eval_count < budget - 20:
                # Local L-BFGS-B refinement
                try:
                    res = minimize(
                        wrapped_func,
                        x_next,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
                    )
                except Exception:
                    pass

        # --- Phase 3: Final Polish ---
        if eval_count < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
