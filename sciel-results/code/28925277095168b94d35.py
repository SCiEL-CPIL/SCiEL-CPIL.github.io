"""Sparse Grid Interpolation + Adaptive Refinement optimizer for BBOB f22 (Gallagher 21 peaks) in 5D.

Strategy:
  Phase 1: Evaluate a Sobol space-filling sequence (first ~100 points) to
           build a coarse surrogate model of the landscape.
  Phase 2: Iteratively select the next evaluation point by minimizing an
           Inverse-Distance-Weighted (IDW) acquisition function that balances
           exploitation (predicted low f) and exploration (high uncertainty
           from sparse coverage).
  Phase 3: When budget remains or stagnation is detected, run L-BFGS-B
           local refinement from the best points found so far.
  A wall-clock deadline guard (325 s) returns the best result found safely early.
"""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Sparse Grid Interpolation + Adaptive Refinement optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    @staticmethod
    def _sobol_sequence(n, count, seed=12345):
        """Generate a Sobol-like quasi-random sequence in [0,1]^n using
        the inverse CDF of a uniform distribution applied to a Halton-like
        construction (deterministic, no scipy dependency)."""
        points = np.zeros((count, n))
        for d in range(n):
            # Use a different base for each dimension (Halton)
            base = d + 2  # bases: 2, 3, 4, 5, 6, ...
            for i in range(count):
                # Van der Corput sequence in the given base
                x = 0.0
                denom = 1.0
                idx = i + 1
                while idx > 0:
                    denom *= base
                    x += (idx % base) / denom
                    idx //= base
                points[i, d] = x
        return points

    def __call__(self, func):
        deadline = time.time() + 325.0  # safety margin before 330 s kill
        n = self.dim
        budget = self.budget
        calls = {"count": 0}

        def fwrap(x):
            if time.time() >= deadline:
                raise TimeoutError
            calls["count"] += 1
            if calls["count"] > budget:
                raise ValueError("Budget exceeded")
            return func(x)

        best = {"x": None, "f": np.inf}

        def update_best(x, f):
            if f < best["f"]:
                best["f"] = f
                best["x"] = np.asarray(x, dtype=float).copy()

        # Data structures for the surrogate
        X_data = []  # list of evaluated points (n-dim)
        Y_data = []  # list of function values

        def record_point(x, f):
            X_data.append(np.asarray(x, dtype=float).copy())
            Y_data.append(float(f))
            update_best(x, f)

        # ---------------- Phase 1: Sobol initialization ----------------
        n_init = min(80, budget - 20)
        sobol_unit = self._sobol_sequence(n, n_init, seed=42)
        # Map from [0,1]^n to [-5,5]^n
        X_init = sobol_unit * 10.0 - 5.0

        for i in range(n_init):
            if time.time() >= deadline:
                break
            if calls["count"] >= budget - 10:
                break
            x_pt = X_init[i]
            try:
                f_val = fwrap(x_pt)
                record_point(x_pt, f_val)
            except (TimeoutError, ValueError):
                break

        if best["x"] is None:
            x0 = np.zeros(n)
            try:
                best["f"] = fwrap(x0)
                record_point(x0, best["f"])
            except Exception:
                return (None, np.inf)

        # ---------------- Phase 2: IDW-guided adaptive refinement ----------------
        # The acquisition function: for a candidate point x, compute
        #   - IDW prediction: weighted average of nearby Y values
        #   - IDW uncertainty: weighted variance (or inverse of sum of weights)
        # We minimize: prediction - beta * uncertainty  (exploitation - exploration)
        # This is equivalent to maximizing: -prediction + beta * uncertainty

        def idw_acquire(x, beta=1.5, power=2.0, min_weight=1e-10):
            """Compute IDW prediction and uncertainty at point x.
            Returns (predicted_value, uncertainty)."""
            if len(X_data) == 0:
                return 0.0, 100.0
            X_arr = np.array(X_data)
            Y_arr = np.array(Y_data)
            dists = np.linalg.norm(X_arr - x, axis=1)
            # Avoid zero distance (exact point already evaluated)
            dists = np.maximum(dists, 1e-12)
            weights = 1.0 / (dists ** power)
            total_w = weights.sum()
            if total_w < min_weight:
                # Very far from all points: high uncertainty
                return float(np.mean(Y_arr)), 100.0
            pred = np.sum(weights * Y_arr) / total_w
            # Variance-like uncertainty: weighted variance
            var = np.sum(weights * (Y_arr - pred) ** 2) / total_w
            # Also add a distance-based uncertainty component
            nearest_d = float(np.min(dists))
            dist_unc = nearest_d * 0.5  # scale
            uncertainty = float(np.sqrt(var) + dist_unc)
            return float(pred), uncertainty

        def acquisition(x, beta=1.5):
            """Lower is better: predict - beta * uncertainty"""
            pred, unc = idw_acquire(x, beta=beta)
            return pred - beta * unc

        # Adaptive phase: use remaining budget for guided search
        # We'll do batches: evaluate a candidate, update model, pick next.
        # To speed up candidate selection, use a small local grid around
        # promising regions.

        beta = 1.5  # exploration-exploitation balance
        stagnation_count = 0
        best_f_at_stagnation = best["f"]
        stagnation_window = 30  # no improvement in this many calls

        remaining_budget = budget - calls["count"]
        # Reserve some budget for final L-BFGS-B polish
        polish_budget = min(50, remaining_budget // 5)
        adaptive_budget = remaining_budget - polish_budget

        # For candidate generation, we'll use a combination of:
        # 1. Random points (for exploration)
        # 2. Points near the current best (for exploitation)
        # 3. Points in high-uncertainty regions (via coarse grid sampling)

        rng = np.random.default_rng(777)
        n_candidates = 20  # candidates per selection step

        for step in range(adaptive_budget):
            if time.time() >= deadline:
                break
            if calls["count"] >= budget - polish_budget - 5:
                break

            # Generate candidate points
            candidates = []

            # Exploitation: near current best
            if best["x"] is not None:
                for scale in [0.3, 0.5, 0.8, 1.5]:
                    c = best["x"] + rng.normal(0, scale, size=n)
                    c = np.clip(c, -5.0, 5.0)
                    candidates.append(c)

            # Exploration: random points
            for _ in range(n_candidates // 2):
                c = rng.uniform(-5.0, 5.0, size=n)
                candidates.append(c)

            # High-uncertainty regions: sample a coarse grid and pick
            # points far from existing data
            coarse_grid = np.linspace(-5.0, 5.0, 4)  # 4^5 = 1024 is too many
            # Use 3^5 = 243
            coarse_grid = np.linspace(-5.0, 5.0, 3)
            grid_pts = np.array(list(np.meshgrid(*([coarse_grid] * n), indexing="ij")))
            grid_pts = grid_pts.reshape(-1, n)
            # Score by distance to nearest data point
            if len(X_data) > 0:
                X_arr = np.array(X_data)
                # For each grid point, find min distance to data
                # Vectorized: compute all pairwise distances (may be large)
                # For 243 grid points and up to ~450 data points, this is fine
                dists_grid = np.linalg.norm(grid_pts[:, None, :] - X_arr[None, :, :], axis=2)
                min_dists = dists_grid.min(axis=1)
                # Pick the top 5 farthest grid points
                top_far = np.argsort(min_dists)[-5:]
                for idx in top_far:
                    c = grid_pts[idx] + rng.normal(0, 0.3, size=n)
                    c = np.clip(c, -5.0, 5.0)
                    candidates.append(c)

            # Evaluate acquisition for all candidates and pick the best
            best_cand = None
            best_acq = np.inf
            for c in candidates:
                acq_val = acquisition(c, beta=beta)
                if acq_val < best_acq:
                    best_acq = acq_val
                    best_cand = c

            if best_cand is None:
                best_cand = rng.uniform(-5.0, 5.0, size=n)

            try:
                f_val = fwrap(best_cand)
                record_point(best_cand, f_val)

                # Stagnation check
                if best["f"] < best_f_at_stagnation - 1e-6:
                    best_f_at_stagnation = best["f"]
                    stagnation_count = 0
                    # Reduce exploration when making progress
                    beta = max(0.5, beta * 0.9)
                else:
                    stagnation_count += 1
                    if stagnation_count >= stagnation_window:
                        # Increase exploration
                        beta = min(3.0, beta * 1.2)
                        stagnation_count = 0

            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # ---------------- Phase 3: L-BFGS-B polish ----------------
        if best["x"] is not None and calls["count"] < budget - 5 and time.time() < deadline:
            bounds = [(-5.0, 5.0) for _ in range(n)]
            # Polish from the best point and a few nearby points
            polish_starts = [best["x"]]
            # Add a few diverse starting points from the data
            if len(X_data) > 1:
                X_arr = np.array(X_data)
                Y_arr = np.array(Y_data)
                # Top 3 by function value
                top_idx = np.argsort(Y_arr)[:3]
                for idx in top_idx:
                    if idx != 0 or X_arr[idx].tolist() != best["x"].tolist():
                        polish_starts.append(X_arr[idx])

            for start_x in polish_starts[:4]:
                if time.time() >= deadline:
                    break
                if calls["count"] >= budget - 3:
                    break
                try:
                    res = minimize(fwrap, start_x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 80, 'maxfun': 80})
                    if res.fun < best["f"]:
                        update_best(res.x, res.fun)
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    continue

            # A few jittered restarts if budget allows
            if calls["count"] < budget - 10 and time.time() < deadline:
                for _ in range(3):
                    if time.time() >= deadline:
                        break
                    if calls["count"] >= budget - 3:
                        break
                    jitter = rng.normal(0, 0.2, size=n)
                    x_start = best["x"] + jitter
                    x_start = np.clip(x_start, -5.0, 5.0)
                    try:
                        res = minimize(fwrap, x_start, method='L-BFGS-B', bounds=bounds,
                                       options={'maxiter': 50, 'maxfun': 50})
                        if res.fun < best["f"]:
                            update_best(res.x, res.fun)
                    except (TimeoutError, ValueError):
                        break
                    except Exception:
                        continue

        # Final safety
        if best["x"] is None:
            x0 = np.zeros(n)
            best["f"] = fwrap(x0)
            best["x"] = x0

        return best["x"], best["f"]
