"""Sparse-grid surrogate-guided search with adaptive L-BFGS-B refinement for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Surrogate-guided multi-start L-BFGS-B for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Evaluation counter
        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=float).copy()

        rng = np.random.default_rng(42)

        # Phase 1: Build a coarse space-filling sample with Sobol points
        n_sobol = 80
        sobol_engine = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sobol_engine.random(n_sobol)
        sobol_samples = sobol_samples * 10.0 - 5.0

        # Evaluate initial Sobol points
        eval_points = []
        eval_values = []
        for pt in sobol_samples:
            if eval_count >= budget - 150:
                break
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)
            eval_points.append(pt.copy())
            eval_values.append(f_val)

        if not eval_points:
            # Fallback: evaluate at origin
            f_val = wrapped_func(np.zeros(dim))
            update_best(np.zeros(dim), f_val)
            eval_points = [np.zeros(dim)]
            eval_values = [f_val]

        # Phase 2: Surrogate-guided adaptive refinement
        # Use inverse-distance weighting (IDW) to build a simple surrogate
        # and select next points that balance exploitation and exploration.
        
        def idw_predict(x, points, values, power=2.0, epsilon=1e-6):
            """Inverse-distance weighted interpolation with uncertainty estimate."""
            d = np.array(points)
            dists = np.linalg.norm(d - x, axis=1)
            dists = np.maximum(dists, epsilon)
            weights = 1.0 / (dists ** power)
            if np.sum(weights) < 1e-12:
                return np.mean(values), 1.0
            pred = np.sum(weights * np.array(values)) / np.sum(weights)
            # Uncertainty: spread of values weighted by proximity
            var = np.sum(weights * (np.array(values) - pred) ** 2) / np.sum(weights)
            uncertainty = np.sqrt(var + 1e-8)
            return pred, uncertainty

        def select_next_point(points, values, n_candidates=200):
            """Select next evaluation point using expected improvement + exploration."""
            # Generate candidate points: mix of Sobol, random, and perturbations around best
            candidates = []
            
            # New Sobol points (different seed for diversity)
            sobol_engine2 = qmc.Sobol(d=dim, scramble=True, seed=eval_count + 100)
            sobol_new = sobol_engine2.random(n_candidates // 2)
            sobol_new = sobol_new * 10.0 - 5.0
            candidates.extend(sobol_new)
            
            # Random points
            rand_pts = rng.uniform(-4.5, 4.5, size=(n_candidates // 4, dim))
            candidates.extend(rand_pts)
            
            # Perturbations around current best
            for scale in [1.0, 0.5, 0.2, 0.1]:
                pert = best_x + rng.normal(0, scale, size=(n_candidates // 8, dim))
                pert = np.clip(pert, -5.0, 5.0)
                candidates.extend(pert)
            
            candidates = np.array(candidates)
            candidates = np.clip(candidates, -5.0, 5.0)
            
            # Score each candidate: low predicted value + high uncertainty (exploration)
            scores = []
            for c in candidates:
                pred, unc = idw_predict(c, eval_points, eval_values)
                # Balance: exploit low predictions, explore high uncertainty
                score = pred + 0.5 * unc
                scores.append(score)
            
            scores = np.array(scores)
            # Select the candidate with the best (lowest) score
            best_idx = np.argmin(scores)
            return candidates[best_idx]

        # Adaptive refinement loop: alternate between surrogate-guided sampling and local optimization
        n_refine_rounds = 10
        for round_i in range(n_refine_rounds):
            if eval_count >= budget - 50:
                break
            if time.time() > deadline:
                break
            
            # Select and evaluate 5 new points guided by surrogate
            for _ in range(5):
                if eval_count >= budget - 50:
                    break
                if time.time() > deadline:
                    break
                
                next_pt = select_next_point(eval_points, eval_values)
                f_val = wrapped_func(next_pt)
                update_best(next_pt, f_val)
                eval_points.append(next_pt.copy())
                eval_values.append(f_val)
            
            # Run L-BFGS-B from the current best point to refine
            if eval_count < budget - 30 and time.time() < deadline:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 30, 'maxls': 15, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        # Add the refined point to our dataset
                        eval_points.append(best_x.copy())
                        eval_values.append(best_f)
                except Exception:
                    pass

        # Phase 3: Multi-start L-BFGS-B from top surrogate-predicted points
        # Sort evaluated points by actual fitness
        sorted_indices = np.argsort(eval_values)
        top_k = min(8, len(eval_points))
        
        start_points = []
        for idx in sorted_indices[:top_k]:
            start_points.append(eval_points[idx])
        
        # Add perturbations around the best point at multiple scales
        for scale in [0.5, 0.25, 0.1]:
            for _ in range(2):
                sp = best_x + rng.normal(0, scale, dim)
                sp = np.clip(sp, -5.0, 5.0)
                start_points.append(sp)

        # Remove duplicates
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for us in unique_starts:
                if np.linalg.norm(sp - us) < 0.05:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)

        # Run L-BFGS-B from each unique starting point
        for sp in unique_starts:
            if eval_count >= budget - 20:
                break
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 40, 'maxls': 20, 'ftol': 1e-14, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: Use remaining budget for fine local random search around best
        step_size = 0.05
        no_improve_count = 0
        max_no_improve = 25

        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, step_size, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    # Shrink step size
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
                    no_improve_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
