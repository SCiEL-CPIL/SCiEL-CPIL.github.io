# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-objective optimizer for RE37: Sobol init + weighted-sum local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        # Phase 1: Sobol quasi-random initialization (40 points)
        n_init = 40
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        X_init = sampler.random(n_init)
        X_init = lo + (hi - lo) * X_init

        F_list = []
        for x in X_init:
            F_list.append(np.array(problem(x), dtype=float))
        F = np.array(F_list)
        used = n_init

        # Helper: check if we're within budget and time
        def can_evaluate():
            return used < self.budget and (time.time() - t_start) < deadline

        # Helper: local refinement via Nelder-Mead on weighted sum
        def local_refine(x0, weights, max_iter=50):
            nonlocal used
            best_x = x0.copy()
            best_f = np.array(problem(x0), dtype=float)
            used += 1

            def objective(x):
                nonlocal used
                if not can_evaluate():
                    return float('inf')
                f = np.array(problem(x), dtype=float)
                used += 1
                return float(np.dot(weights, f))

            # Initial Nelder-Mead setup
            x_init = x0.copy()
            res = minimize(objective, x_init, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'initial_simplex': 0.1 * (hi - lo)})

            if res.fun < float('inf') and res.fun < np.dot(weights, best_f):
                best_x = res.x
                best_f = np.array(problem(res.x), dtype=float)
                used += 1

            return best_x, best_f

        # Phase 2: Weighted-sum local refinement
        # Use various weight vectors to explore different parts of Pareto front
        weight_sets = [
            np.ones(n_obj) / n_obj,
            np.array([0.5, 0.25, 0.25]),
            np.array([0.25, 0.5, 0.25]),
            np.array([0.25, 0.25, 0.5]),
            np.array([0.6, 0.2, 0.2]),
            np.array([0.2, 0.6, 0.2]),
            np.array([0.2, 0.2, 0.6]),
            np.array([0.4, 0.3, 0.3]),
            np.array([0.3, 0.4, 0.3]),
            np.array([0.3, 0.3, 0.4]),
        ]

        # Sort by aggregate to find diverse starting points
        F_sorted_idx = np.lexsort((F[:, -1], F[:, -2], F[:, 0]))
        F_sorted = F[F_sorted_idx]
        X_sorted = X_init[F_sorted_idx]

        for i, w in enumerate(weight_sets):
            if not can_evaluate():
                break

            # Pick starting point: prefer points that are good in the weighted direction
            weighted_scores = F @ w
            # Pick the best 3 starting points for this weight direction
            top_idx = np.argsort(weighted_scores)[:3]

            for start_idx in top_idx:
                if not can_evaluate():
                    break
                x0 = X_sorted[start_idx].copy()
                try:
                    bx, bf = local_refine(x0, w, max_iter=30)
                    if np.all(bf < np.array([1.1, 1.3, 1.1])):  # better than reference
                        F = np.vstack([F, bf])
                        X_init = np.vstack([X_init, bx])
                except Exception:
                    pass

        # Phase 3: Additional random + local refinement if budget allows
        rng = np.random.default_rng(123)
        n_extra = 10
        for _ in range(n_extra):
            if not can_evaluate():
                break
            x_new = lo + (hi - lo) * rng.random(dim)
            f_new = np.array(problem(x_new), dtype=float)
            used += 1
            if np.all(f_new < np.array([1.1, 1.3, 1.1])):
                F = np.vstack([F, f_new])
                X_init = np.vstack([X_init, x_new])

        # Phase 4: Final local refinement on best non-dominated points
        if can_evaluate():
            # Find non-dominated points
            non_dom_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dom_mask[i] = False
                        break

            if np.any(non_dom_mask):
                nd_idx = np.where(non_dom_mask)[0]
                # Refine top 5 non-dominated points
                for idx in nd_idx[:5]:
                    if not can_evaluate():
                        break
                    w = np.ones(n_obj) / n_obj
                    try:
                        bx, bf = local_refine(X_init[idx], w, max_iter=20)
                        if np.all(bf < np.array([1.1, 1.3, 1.1])):
                            F = np.vstack([F, bf])
                            X_init = np.vstack([X_init, bx])
                    except Exception:
                        pass

        return X_init
# EVOLVE-BLOCK-END
