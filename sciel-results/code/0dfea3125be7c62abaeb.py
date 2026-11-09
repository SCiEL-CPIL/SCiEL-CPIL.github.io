"""Rewritten multi-objective optimizer using Epsilon-Constraint decomposition with gradient-based refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE21 (4 vars, 2 objectives).
    Strategy:
      1. Random sampling to get an initial front and estimate objective ranges.
      2. Epsilon-Constraint Method: for a set of epsilon values on objective 2,
         minimize objective 1 subject to objective 2 <= epsilon using L-BFGS-B.
      3. Local refinement around the best points found.
      4. Deadline guard to ensure we finish within wall-clock limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        all_points = []  # list of x vectors
        all_f = []       # list of objective vectors

        def eval_and_store(x):
            nonlocal all_points, all_f
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_points.append(x.copy())
            all_f.append(f.copy())
            return f

        def time_ok():
            return time.time() < deadline

        # Phase 1: Random sampling
        n_random = min(40, self.budget // 4)
        for _ in range(n_random):
            if not time_ok():
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            eval_and_store(x)

        # Estimate objective ranges from random samples
        F_rand = np.array(all_f)
        f1_min, f1_max = F_rand[:, 0].min(), F_rand[:, 0].max()
        f2_min, f2_max = F_rand[:, 1].min(), F_rand[:, 1].max()

        # Phase 2: Epsilon-Constraint Method
        # We minimize f1 subject to f2 <= epsilon
        # Choose epsilon values from f2_max down to f2_min
        n_eps = min(15, max(5, (self.budget - len(all_points)) // 3))
        epsilons = np.linspace(f2_max, f2_min, n_eps + 2)[1:-1]  # avoid extremes

        for eps in epsilons:
            if not time_ok():
                break
            if len(all_points) >= self.budget - 10:
                break

            # Objective: f1 + penalty for violating f2 <= eps
            def objective(x):
                f = np.asarray(problem(x), dtype=float)
                # We need to count this evaluation
                x_clipped = np.clip(x, lo, hi)
                all_points.append(x_clipped.copy())
                all_f.append(f.copy())
                penalty = max(0.0, f[1] - eps) ** 2 * 100.0
                return f[0] + penalty

            # Start from the best point so far in terms of f1
            F_arr = np.array(all_f)
            start_idx = int(np.argmin(F_arr[:, 0]))
            x0 = all_points[start_idx].copy()

            try:
                res = minimize(
                    objective, x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 30, 'maxfun': 10, 'gtol': 1e-4}
                )
                if res.success or res.fun < 1e10:
                    x_best = np.clip(res.x, lo, hi)
                    f_best = np.asarray(problem(x_best), dtype=float)
                    all_points.append(x_best.copy())
                    all_f.append(f_best.copy())
            except Exception:
                pass

        # Phase 3: Local refinement around non-dominated points
        if time_ok() and len(all_points) < self.budget - 5:
            F_arr = np.array(all_f)
            # Find non-dominated points
            n = len(F_arr)
            is_dom = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_dom[i] = True
                        break
            nd_idx = np.where(~is_dom)[0]
            
            # Refine top non-dominated points
            for idx in nd_idx[:5]:
                if not time_ok() or len(all_points) >= self.budget - 2:
                    break
                x0 = all_points[idx].copy()
                # Try small perturbations
                for _ in range(3):
                    if not time_ok() or len(all_points) >= self.budget - 1:
                        break
                    dx = 0.05 * (hi - lo) * rng.standard_normal(self.dim)
                    x_new = np.clip(x0 + dx, lo, hi)
                    eval_and_store(x_new)

        # Ensure we don't exceed budget
        if len(all_points) > self.budget:
            all_points = all_points[:self.budget]
            all_f = all_f[:self.budget]

        return all_points
# EVOLVE-BLOCK-END
