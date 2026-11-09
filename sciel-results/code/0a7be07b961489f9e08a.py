"""Multi-objective optimizer for RE21 using Augmented Epsilon-Constraint decomposition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using Augmented Epsilon-Constraint Method."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Track all evaluated points
        all_x = []
        all_f = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            all_x.append(x.copy())
            all_f.append(f.copy())
            eval_count += 1
            return x, f

        # Phase 1: Sample to get initial bounds on objectives
        n_init = min(20, self.budget // 4)
        rng = np.random.default_rng(0)
        for _ in range(n_init):
            x = lo + span * rng.random(dim)
            x, f = evaluate(x)

        # Get current objective ranges
        f_arr = np.array(all_f)
        f_min = f_arr.min(axis=0)
        f_max = f_arr.max(axis=0)

        # Phase 2: Augmented Epsilon-Constraint Method
        # We optimize f1 subject to f2 <= epsilon, with epsilon varying
        # Use augmented form: minimize f1 + rho * max(0, f2 - epsilon)

        # Determine number of epsilon values based on remaining budget
        remaining = self.budget - eval_count
        # Each scipy.optimize call may use multiple function evaluations internally
        # We'll do a moderate number of epsilon values
        n_eps = min(12, max(4, remaining // 8))

        # Create epsilon values: from f_max[1] down to f_min[1]
        # Use geometric spacing for better coverage
        eps_values = np.linspace(f_max[1] * 1.1, f_min[1] * 0.9, n_eps)
        # Ensure epsilon is decreasing
        eps_values = np.sort(eps_values)[::-1]

        rho = 0.1  # augmentation parameter

        def objective_augmented(x, eps_val):
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            # Augmented epsilon constraint
            penalty = rho * max(0.0, f[1] - eps_val)
            return f[0] + penalty

        # For each epsilon, run L-BFGS-B from multiple starting points
        n_starts = max(1, min(3, remaining // (n_eps * 5)))

        for eps_val in eps_values:
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break

            # Try multiple starting points
            for start_i in range(n_starts):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break

                if start_i == 0:
                    # Start from the point that best satisfies f2 <= eps
                    valid = [i for i, f in enumerate(all_f) if f[1] <= eps_val]
                    if valid:
                        # Pick the one with smallest f1
                        start_x = all_x[min(valid, key=lambda i: all_f[i][0])]
                    else:
                        start_x = (lo + hi) / 2
                else:
                    # Random perturbation
                    start_x = lo + span * rng.random(dim)

                # Define the objective for scipy
                def obj_func(x):
                    return objective_augmented(x, eps_val)

                # Bounds for scipy
                bounds_scipy = list(zip(lo, hi))

                try:
                    result = minimize(
                        obj_func,
                        start_x,
                        method='L-BFGS-B',
                        bounds=bounds_scipy,
                        options={'maxiter': 100, 'maxfun': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                except Exception:
                    continue

                # Evaluate the result
                if result.x is not None:
                    x_res, f_res = evaluate(result.x)

        # Phase 3: Local refinement using Nelder-Mead on promising points
        if eval_count < self.budget and time.time() < deadline - 10:
            # Get non-dominated points
            f_arr = np.array(all_f)
            n = len(f_arr)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(f_arr[j] <= f_arr[i]) and np.any(f_arr[j] < f_arr[i]):
                        nd_mask[i] = False
                        break

            nd_indices = np.where(nd_mask)[0]
            # Sort by f1 to get diverse points
            nd_indices = nd_indices[np.argsort(f_arr[nd_indices, 0])]

            # Refine a subset of non-dominated points
            n_refine = min(len(nd_indices), max(3, (self.budget - eval_count) // 10))
            # Pick evenly spaced from the sorted list
            if n_refine > 0:
                step_idx = max(1, len(nd_indices) // n_refine)
                refine_indices = nd_indices[::step_idx][:n_refine]

                for idx in refine_indices:
                    if eval_count >= self.budget:
                        break
                    if time.time() > deadline - 5:
                        break

                    start_x = all_x[idx].copy()

                    def obj_combined(x):
                        x_clipped = np.clip(x, lo, hi)
                        f = problem(x_clipped)
                        # Combined objective: weighted sum
                        return f[0] + 0.5 * f[1]

                    try:
                        result = minimize(
                            obj_combined,
                            start_x,
                            method='Nelder-Mead',
                            options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                        )
                    except Exception:
                        continue

                    if result.x is not None:
                        x_res, f_res = evaluate(result.x)

        # Phase 4: Final scatter to fill gaps in the front
        if eval_count < self.budget and time.time() < deadline - 5:
            f_arr = np.array(all_f)
            # Find gaps in f1 values
            f1_vals = np.sort(f_arr[:, 0])
            if len(f1_vals) > 1:
                # Add points in gaps
                for _ in range(min(5, self.budget - eval_count)):
                    if time.time() > deadline - 2:
                        break
                    # Random point biased toward gaps
                    x = lo + span * rng.random(dim)
                    x, f = evaluate(x)

        return all_x
# EVOLVE-BLOCK-END
