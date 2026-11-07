"""Multi-objective optimizer for RE37 using weighted-sum scalarization with local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Count evaluations
        eval_count = 0
        max_evals = self.budget

        # Safe evaluation wrapper
        def safe_eval(x):
            nonlocal eval_count
            if eval_count >= max_evals:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = np.asarray(problem(x), dtype=float)
                if f.shape != (n_obj,) or not np.all(np.isfinite(f)):
                    return None
                eval_count += 1
                return f
            except Exception:
                return None

        # Reference point
        ref = np.array([1.1, 1.3, 1.1])

        # Store all evaluated points
        X_all = []
        F_all = []

        def record(x, f):
            if f is not None and np.all(f < ref):
                X_all.append(x.copy())
                F_all.append(f.copy())

        # Phase 1: Weighted-sum scalarization with multiple weight vectors
        # Generate weight vectors that cover the objective space
        n_weights = 12
        weights = []
        # Uniformly spaced weights in the simplex
        for i in range(n_weights):
            # Use a grid of weight vectors
            if i == 0:
                w = np.ones(n_obj) / n_obj
            elif i == 1:
                w = np.zeros(n_obj)
                w[0] = 1.0
            elif i == 2:
                w = np.zeros(n_obj)
                w[1] = 1.0
            elif i == 3:
                w = np.zeros(n_obj)
                w[2] = 1.0
            elif i == 4:
                w = np.zeros(n_obj)
                w[0] = 0.7
                w[1] = 0.15
                w[2] = 0.15
            elif i == 5:
                w = np.zeros(n_obj)
                w[0] = 0.15
                w[1] = 0.7
                w[2] = 0.15
            elif i == 6:
                w = np.zeros(n_obj)
                w[0] = 0.15
                w[1] = 0.15
                w[2] = 0.7
            elif i == 7:
                w = np.zeros(n_obj)
                w[0] = 0.5
                w[1] = 0.3
                w[2] = 0.2
            elif i == 8:
                w = np.zeros(n_obj)
                w[0] = 0.2
                w[1] = 0.5
                w[2] = 0.3
            elif i == 9:
                w = np.zeros(n_obj)
                w[0] = 0.3
                w[1] = 0.2
                w[2] = 0.5
            elif i == 10:
                w = np.zeros(n_obj)
                w[0] = 0.6
                w[1] = 0.2
                w[2] = 0.2
            else:
                w = np.zeros(n_obj)
                w[0] = 0.2
                w[1] = 0.6
                w[2] = 0.2
            weights.append(w)

        # Multi-start local search for each weight vector
        n_starts_per_weight = 5
        rng = np.random.default_rng(42)

        for w in weights:
            if eval_count >= max_evals or time.time() > deadline:
                break

            for s in range(n_starts_per_weight):
                if eval_count >= max_evals or time.time() > deadline:
                    break

                # Random initial point
                x0 = lo + (hi - lo) * rng.random(dim)

                # Scalarized objective
                def scalar_obj(x, w=w):
                    f = safe_eval(x)
                    if f is None:
                        return 1e10
                    return float(np.dot(w, f))

                # Try L-BFGS-B
                try:
                    res = minimize(
                        scalar_obj, x0, method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 50, 'ftol': 1e-8}
                    )
                    if res.success or res.fun < 1e10:
                        x_opt = np.clip(res.x, lo, hi)
                        f_opt = safe_eval(x_opt)
                        if f_opt is not None:
                            record(x_opt, f_opt)
                except Exception:
                    pass

        # Phase 2: Fill remaining budget with random sampling + local polish
        remaining = max_evals - eval_count
        if remaining > 0 and time.time() < deadline:
            n_random = min(remaining, max(20, remaining // 3))
            for _ in range(n_random):
                if eval_count >= max_evals or time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                f = safe_eval(x)
                if f is not None:
                    record(x, f)

        # Phase 3: Local polish on best non-dominated points
        if X_all and F_all and eval_count < max_evals and time.time() < deadline:
            F_arr = np.array(F_all)
            X_arr = np.array(X_all)

            # Find non-dominated points
            n_pts = len(F_arr)
            non_dom_idx = []
            for i in range(n_pts):
                dominated = False
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_idx.append(i)

            # Polish top non-dominated points
            max_polish = min(len(non_dom_idx), 10)
            for idx in non_dom_idx[:max_polish]:
                if eval_count >= max_evals or time.time() > deadline:
                    break

                x0 = X_arr[idx]
                # Try small perturbations with local optimization
                for perturb_scale in [0.05, 0.02, 0.01]:
                    if eval_count >= max_evals or time.time() > deadline:
                        break

                    def obj(x, x0=x0):
                        f = safe_eval(x)
                        if f is None:
                            return 1e10
                        # Use a simple norm-based objective to polish
                        return float(np.sum(np.maximum(f - F_arr[idx], 0)**2) + 0.1 * np.sum((x - x0)**2))

                    try:
                        res = minimize(
                            obj, x0, method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 30, 'ftol': 1e-9}
                        )
                        if res.success or res.fun < 1e10:
                            x_opt = np.clip(res.x, lo, hi)
                            f_opt = safe_eval(x_opt)
                            if f_opt is not None:
                                record(x_opt, f_opt)
                    except Exception:
                        pass

        # Return the best set (interface requires returning something; evaluator ignores it)
        if X_all:
            return np.array(X_all)
        return np.array([])
# EVOLVE-BLOCK-END
