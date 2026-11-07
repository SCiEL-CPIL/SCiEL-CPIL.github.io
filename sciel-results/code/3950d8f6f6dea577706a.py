# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using weighted-sum decomposition with diverse weights."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count += 1
            return x, f

        # Phase 1: Generate diverse weight vectors for weighted-sum decomposition
        # Use a mix of uniform and non-uniform weights to cover the Pareto front
        n_weights = 40
        weights = []

        # Uniform weights
        n_uniform = 20
        for i in range(n_uniform):
            w = np.full(n_obj, 1.0 / n_obj)
            # Perturb slightly
            for d in range(n_obj):
                w[d] = (i + 1) / (n_uniform + 1)
            w = w / np.sum(w)
            weights.append(w)

        # Non-uniform weights focusing on extremes
        n_extreme = 10
        for i in range(n_extreme):
            t = (i + 1) / (n_extreme + 1)
            w = np.zeros(n_obj)
            w[0] = t
            w[1] = 1 - t
            weights.append(w)

        # Some random weights
        np.random.seed(123)
        for i in range(n_weights - n_uniform - n_extreme):
            w = np.random.dirichlet(np.ones(n_obj))
            weights.append(w)

        # Phase 2: Initial exploration with a coarse grid
        # Evaluate a few points to get initial sense of the problem
        n_initial = 10
        for i in range(n_initial):
            if eval_count >= budget or time.time() > deadline:
                break
            # Spread points across the domain
            x = np.array([(lo[d] + hi[d]) * (i + 1) / (n_initial + 1) for d in range(dim)])
            evaluate(x)

        # Phase 3: Weighted-sum optimization with diverse weights
        # For each weight vector, run a local optimization
        for w in weights:
            if eval_count >= budget or time.time() > deadline:
                break

            # Choose a starting point based on the weight
            # Points with higher weight on objective 0 should start in regions good for obj 0
            x0 = np.array([(lo[d] + hi[d]) * 0.5 for d in range(dim)])

            # Add some diversity in starting points
            if len(all_X) > 0:
                # Pick a random existing point as starting point
                idx = np.random.randint(0, len(all_X))
                x0 = all_X[idx].copy()

            def objective(x):
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                return np.dot(w, f)

            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if eval_count < budget and time.time() < deadline:
                    cx = np.clip(res.x, lo, hi)
                    cf = np.array(problem(cx))
                    all_X.append(cx.copy())
                    all_F.append(cf.copy())
                    eval_count += 1
            except Exception:
                pass

        # Phase 4: Polish the best non-dominated points
        if eval_count < budget and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Find non-dominated points
            mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            nd_idx = np.where(mask)[0]

            # Pick top points by sum of objectives
            if len(nd_idx) > 0:
                order = np.argsort(np.sum(F_arr[nd_idx], axis=1))
                top_n = min(10, len(nd_idx), (budget - eval_count) // 3)

                for idx in order[:top_n]:
                    if eval_count >= budget or time.time() > deadline:
                        break

                    x0 = X_arr[idx].copy()

                    # Use Nelder-Mead for local polish
                    def objective_nm(x):
                        x = np.clip(x, lo, hi)
                        f = np.array(problem(x))
                        return np.sum(f)

                    try:
                        res = minimize(objective_nm, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                        if eval_count < budget and time.time() < deadline:
                            cx = np.clip(res.x, lo, hi)
                            cf = np.array(problem(cx))
                            all_X.append(cx.copy())
                            all_F.append(cf.copy())
                            eval_count += 1
                    except Exception:
                        pass

        return all_X
# EVOLVE-BLOCK-END
