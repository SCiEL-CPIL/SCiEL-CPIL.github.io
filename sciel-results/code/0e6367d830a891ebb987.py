"""Augmented Epsilon-Constraint Method with LHS initialization for RE21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE21 (4 vars, 2 objectives).
    Strategy:
      1. LHS (Latin Hypercube) initialization for diverse starting points.
      2. Augmented Epsilon-Constraint Method: minimize f1 + mu*(max(0, f2-eps))^2
         for a sequence of epsilon values, using L-BFGS-B.
      3. Local refinement around Pareto-optimal points.
      4. Deadline guard to ensure we finish within wall-clock limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        all_points = []
        all_f = []
        eval_count = [0]

        def eval_and_store(x):
            nonlocal all_points, all_f
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_points.append(x.copy())
            all_f.append(f.copy())
            eval_count[0] += 1
            return f

        def time_ok():
            return time.time() < deadline and eval_count[0] < self.budget

        def remaining_budget():
            return max(0, self.budget - eval_count[0])

        # Phase 1: LHS initialization
        n_init = min(30, self.budget // 4)
        for i in range(n_init):
            if not time_ok():
                break
            # LHS: divide each dimension into n_init equal intervals
            x = np.zeros(self.dim)
            for d in range(self.dim):
                # Place point i in the i-th interval of dimension d
                lower_frac = i / n_init
                upper_frac = (i + 1) / n_init
                frac = lower_frac + rng.random() * (upper_frac - lower_frac)
                x[d] = lo[d] + frac * (hi[d] - lo[d])
            eval_and_store(x)

        # Estimate objective ranges from initial samples
        F_arr = np.array(all_f)
        f1_min, f1_max = F_arr[:, 0].min(), F_arr[:, 0].max()
        f2_min, f2_max = F_arr[:, 1].min(), F_arr[:, 1].max()

        # Phase 2: Augmented Epsilon-Constraint Method
        # Minimize f1 + mu * max(0, f2 - eps)^2
        # Use a sequence of epsilon values from high to low f2
        n_eps = min(12, max(6, remaining_budget() // 4))
        epsilons = np.linspace(f2_max, f2_min, n_eps + 2)[1:-1]

        mu = 10.0  # penalty parameter

        for eps in epsilons:
            if not time_ok():
                break
            if remaining_budget() < 5:
                break

            # Objective function with augmented penalty
            def aug_objective(x):
                x_clipped = np.clip(x, lo, hi)
                f = np.asarray(problem(x_clipped), dtype=float)
                all_points.append(x_clipped.copy())
                all_f.append(f.copy())
                eval_count[0] += 1
                violation = max(0.0, f[1] - eps)
                return f[0] + mu * violation ** 2

            # Start from a good point: use the point with best f1 among those
            # that satisfy f2 <= eps, or the overall best f1 if none
            F_cur = np.array(all_f)
            valid_mask = F_cur[:, 1] <= eps + 1e-6
            if np.any(valid_mask):
                start_idx = int(np.argmin(F_cur[valid_mask, 0]))
                # Map back to original index
                valid_indices = np.where(valid_mask)[0]
                x0 = all_points[valid_indices[start_idx]].copy()
            else:
                start_idx = int(np.argmin(F_cur[:, 0]))
                x0 = all_points[start_idx].copy()

            try:
                res = minimize(
                    aug_objective, x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 20, 'maxfun': 8, 'gtol': 1e-4}
                )
                if res.success or res.fun < 1e10:
                    x_best = np.clip(res.x, lo, hi)
                    f_best = np.asarray(problem(x_best), dtype=float)
                    all_points.append(x_best.copy())
                    all_f.append(f_best.copy())
                    eval_count[0] += 1
            except Exception:
                pass

        # Phase 3: Local refinement around non-dominated points
        if time_ok() and remaining_budget() > 3:
            F_arr = np.array(all_f)
            n = len(F_arr)
            # Find non-dominated points
            is_dom = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_dom[i] = True
                        break
            nd_idx = np.where(~is_dom)[0]

            # Refine top non-dominated points with small perturbations
            for idx in nd_idx[:min(5, len(nd_idx))]:
                if not time_ok() or remaining_budget() < 2:
                    break
                x0 = all_points[idx].copy()
                for _ in range(2):
                    if not time_ok() or remaining_budget() < 1:
                        break
                    dx = 0.03 * (hi - lo) * rng.standard_normal(self.dim)
                    x_new = np.clip(x0 + dx, lo, hi)
                    eval_and_store(x_new)

        return all_points
# EVOLVE-BLOCK-END
