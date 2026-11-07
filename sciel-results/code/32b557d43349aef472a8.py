"""Rewritten program for WFG7 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using objective-space targeting with L-BFGS-B refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for scoring
        ref = np.array([3.0, 5.0, 7.0])

        # For WFG7 with 3 objectives, the Pareto front is a 2D surface.
        # We parameterize points on the front using two parameters (u1, u2) in [0,1].
        # The objective values on the front follow specific patterns.
        # 
        # Strategy: 
        # 1. Generate target objective vectors that are likely on or near the Pareto front
        # 2. For each target, use L-BFGS-B to find decision variables that minimize
        #    the distance to the target in objective space
        # 3. This directly populates the Pareto front region
        #
        # WFG7 front characteristics (approximate):
        # - f1 ranges roughly [1, 20]
        # - f2 ranges roughly [1, 20]  
        # - f3 ranges roughly [1, 20]
        # - The front is convex/concave depending on parameterization
        #
        # We'll use a stratified grid of target points in objective space,
        # then refine each with local optimization.

        # Phase 1: Generate diverse initial targets in objective space
        # Use a stratified sampling approach to cover the front region
        n_targets = 15  # number of target points to optimize
        n_refine = 5    # refinement steps per target (each uses ~10-15 evals)

        # Create target grid - spread across the expected front region
        # For WFG7, the front typically has objectives in [1, 15] range
        # We'll create targets that are strictly better than ref in all objectives
        u_vals = np.linspace(0.05, 0.95, 4)  # 4x4 grid = 16, use 15
        targets = []
        for u1 in u_vals:
            for u2 in u_vals:
                if len(targets) >= n_targets:
                    break
                # Parameterize front: objectives decrease as parameters increase
                # This is a rough approximation of WFG7 front shape
                f1 = 1.0 + 14.0 * (1.0 - u1) * (1.0 - 0.3 * u2)
                f2 = 1.0 + 14.0 * (1.0 - u2) * (1.0 - 0.3 * u1)
                f3 = 1.0 + 14.0 * (1.0 - 0.7 * u1) * (1.0 - 0.7 * u2)
                # Ensure strictly below reference
                f1 = min(f1, ref[0] - 0.5)
                f2 = min(f2, ref[1] - 0.5)
                f3 = min(f3, ref[2] - 0.5)
                targets.append([f1, f2, f3])

        # Also add some random targets for diversity
        rng = np.random.default_rng(0)
        for _ in range(5):
            f1 = rng.uniform(1.0, ref[0] - 0.5)
            f2 = rng.uniform(1.0, ref[1] - 0.5)
            f3 = rng.uniform(1.0, ref[2] - 0.5)
            targets.append([f1, f2, f3])

        targets = np.array(targets[:n_targets])

        # Evaluate budget tracking
        eval_count = [0]

        def count_eval():
            eval_count[0] += 1
            return eval_count[0]

        # Initial random evaluations to get a sense of the landscape
        n_initial = min(10, budget)
        X_init = lo + span * rng.random((n_initial, dim))
        F_init = np.array([problem(x) for x in X_init])
        eval_count[0] += n_initial

        # Find the best initial point (lowest sum of objectives)
        sums = np.sum(F_init, axis=1)
        best_init_idx = np.argmin(sums)
        x_best_init = X_init[best_init_idx].copy()
        f_best_init = F_init[best_init_idx].copy()

        # Phase 2: For each target, use L-BFGS-B to find decision variables
        # that achieve objectives close to the target
        all_x = [x_best_init]
        all_f = [f_best_init]

        for target in targets:
            if time.time() >= deadline or eval_count[0] >= budget - 10:
                break

            # Define objective function for L-BFGS-B:
            # minimize sum of squared differences between actual and target objectives
            # Plus a penalty for being above the reference point
            def target_objective(x, target=target):
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                count_eval()
                
                # Distance to target
                dist_sq = np.sum((np.array(f) - target) ** 2)
                
                # Penalty for exceeding reference (shouldn't happen if clipped, but safe)
                penalty = 0.0
                for i in range(n_obj):
                    if f[i] > ref[i]:
                        penalty += (f[i] - ref[i]) ** 2 * 100.0
                
                # Also penalize points that are dominated by known best
                # (to encourage Pareto diversity)
                return dist_sq + penalty

            # Try multiple starting points for each target
            n_starts = 3
            best_x_for_target = None
            best_f_for_target = None
            best_obj_val = np.inf

            for s in range(n_starts):
                if time.time() >= deadline or eval_count[0] >= budget - 5:
                    break

                # Different starting points: best initial, random, or perturbed
                if s == 0:
                    x0 = x_best_init.copy()
                elif s == 1:
                    x0 = lo + span * rng.random(dim)
                else:
                    x0 = x_best_init + span * rng.standard_normal(dim) * 0.1
                    x0 = np.clip(x0, lo, hi)

                # L-BFGS-B optimization
                try:
                    result = minimize(
                        target_objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 50, 'maxfun': 30, 'ftol': 1e-6}
                    )
                except Exception:
                    continue

                if eval_count[0] >= budget:
                    break

                x_opt = np.clip(result.x, lo, hi)
                f_opt = np.array(problem(x_opt))
                count_eval()

                # Check if this is valid (below reference)
                if np.all(f_opt < ref):
                    obj_val = np.sum((f_opt - target) ** 2)
                    if obj_val < best_obj_val:
                        best_obj_val = obj_val
                        best_x_for_target = x_opt.copy()
                        best_f_for_target = f_opt.copy()

            if best_x_for_target is not None:
                all_x.append(best_x_for_target)
                all_f.append(best_f_for_target)

                # Update best initial point if we found something better
                if np.sum(best_f_for_target) < np.sum(f_best_init):
                    x_best_init = best_x_for_target.copy()
                    f_best_init = best_f_for_target.copy()

        # Phase 3: Use remaining budget for local refinement of the best points
        if eval_count[0] < budget and len(all_x) > 0:
            all_f_arr = np.array(all_f)
            all_x_arr = np.array(all_x)

            # Sort by quality (sum of objectives)
            quality_order = np.argsort(np.sum(all_f_arr, axis=1))

            # Refine the top points with local search
            step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
            step_idx = 0

            while eval_count[0] < budget - 5 and time.time() < deadline:
                # Pick a point to refine (cycle through good ones)
                idx = quality_order[eval_count[0] % len(quality_order)]
                if idx >= len(all_x_arr):
                    idx = 0

                x_cur = all_x_arr[idx]
                f_cur = all_f_arr[idx]

                step = step_sizes[min(step_idx, len(step_sizes) - 1)]
                improved = False

                # Try several perturbations
                n_perturb = min(3, budget - eval_count[0])
                for _ in range(n_perturb):
                    if eval_count[0] >= budget or time.time() >= deadline:
                        break

                    direction = rng.standard_normal(dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    x_child = x_cur + step * span * direction
                    x_child = np.clip(x_child, lo, hi)

                    f_child = np.array(problem(x_child))
                    eval_count[0] += 1

                    # Accept if it's better in at least one objective and not worse in others
                    # (weak dominance check for diversity)
                    if np.all(f_child <= f_cur + 1e-10) and np.any(f_child < f_cur - 1e-10):
                        all_x_arr[idx] = x_child
                        all_f_arr[idx] = f_child
                        improved = True
                        break

                if not improved:
                    step_idx = min(step_idx + 1, len(step_sizes) - 1)

            all_x = list(all_x_arr)
            all_f = list(all_f_arr)

        # Final: return the archive
        return np.array(all_x)
# EVOLVE-BLOCK-END
