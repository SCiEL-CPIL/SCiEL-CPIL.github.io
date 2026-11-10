"""Multi-start local search with Augmented Epsilon-Constraint for RE21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start local search using Augmented Epsilon-Constraint method."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate quasi-random starting points using Sobol sequence
        sampler = qmc.Sobol(d=dim, scramble=True)
        n_starts = 50
        # Draw more than needed, we'll use a subset based on budget
        n_sobol = min(budget, 200)
        sobol_points = sampler.random(n_sobol)  # in [0,1]^dim
        X_starts = lo + span * sobol_points

        # We need to decide how to allocate budget:
        # - n_starts local searches, each with a limited budget
        # - Also keep some evaluations for initial sampling
        #
        # Strategy: Use epsilon-constraint method.
        # For 2 objectives, we fix the second objective to be <= epsilon
        # and minimize the first. We sweep epsilon to trace the Pareto front.
        #
        # But we need to know the range of f2. Let's first do a small random
        # sample to estimate bounds of objectives, then do local searches.
        #
        # Actually, with budget=200 and 50 starts, each local search gets ~4 evals.
        # That's too few for L-BFGS-B. Let's reconsider.
        #
        # Better approach: Use fewer, higher-quality local searches.
        # Let's use 10-15 starts with more budget each.
        #
        # Revised strategy:
        # 1. Initial random sample of ~30 points to get objective ranges
        # 2. For each of ~10 epsilon values, do a local search minimizing f1
        #    subject to f2 <= epsilon
        # 3. Also do some unconstrained local searches from good initial points
        #
        # Total budget: 200
        # - 30 for initial sampling
        # - 170 for local searches (17 searches of ~10 evals each, or 10 of ~17)

        n_initial = 30
        n_starts = 10
        budget_per_search = (budget - n_initial) // n_starts

        # Phase 1: Initial sampling to estimate objective ranges
        X_init = X_starts[:n_initial]
        F_init = []
        for x in X_init:
            f = np.asarray(problem(x), dtype=float)
            F_init.append(f)
        F_init = np.array(F_init)

        # Estimate objective ranges
        f1_min = np.min(F_init[:, 0])
        f1_max = np.max(F_init[:, 0])
        f2_min = np.min(F_init[:, 1])
        f2_max = np.max(F_init[:, 1])

        # Create epsilon values spanning the range of f2
        # Use log-spaced or linear-spaced epsilons
        n_eps = 8
        epsilons = np.linspace(f2_min, f2_max, n_eps)

        # Phase 2: Local searches with epsilon constraints
        # For each epsilon, minimize f1 subject to f2 <= epsilon
        # Use augmented Lagrangian or penalty method

        all_X = [X_init]
        all_F = [F_init]

        for i_eps in range(n_eps):
            if time.time() > deadline:
                break
            if budget - (n_initial + i_eps * budget_per_search) < 5:
                break

            epsilon = epsilons[i_eps]
            # Choose starting point: pick the one closest to having f2 near epsilon
            # Or just use a random start from the initial set
            start_idx = i_eps % n_initial
            x0 = X_init[start_idx].copy()

            # Define objective with penalty for constraint violation
            def obj(x):
                fx = np.asarray(problem(x), dtype=float)
                f1_val = fx[0]
                f2_val = fx[1]
                # Penalty for violating f2 <= epsilon
                penalty = 100.0 * max(0, f2_val - epsilon)
                return f1_val + penalty

            # Use L-BFGS-B
            try:
                res = minimize(obj, x0, method='L-BFGS-B',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': 50, 'maxfun': budget_per_search})
                x_opt = np.clip(res.x, lo, hi)
                f_opt = np.asarray(problem(x_opt), dtype=float)
                all_X.append(x_opt.reshape(1, -1))
                all_F.append(f_opt.reshape(1, -1))
            except Exception:
                pass

        # Phase 3: Also do some unconstrained local searches from best points
        # Find the point with best f1 and best f2, do local searches from there
        X_all = np.vstack(all_X)
        F_all = np.vstack(all_F)

        # Get non-dominated points
        nd_mask = self._non_dominated_mask(F_all)
        if np.any(nd_mask):
            X_nd = X_all[nd_mask]
            F_nd = F_all[nd_mask]
        else:
            X_nd = X_all
            F_nd = F_all

        # Do a few more local searches from non-dominated points
        n_extra = min(5, (budget - n_initial - n_eps * budget_per_search) // 10)
        for i in range(n_extra):
            if time.time() > deadline:
                break
            if i >= len(X_nd):
                break
            x0 = X_nd[i].copy()

            def obj_unconstrained(x):
                fx = np.asarray(problem(x), dtype=float)
                return fx[0]  # Minimize f1

            try:
                res = minimize(obj_unconstrained, x0, method='L-BFGS-B',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': 30, 'maxfun': 10})
                x_opt = np.clip(res.x, lo, hi)
                f_opt = np.asarray(problem(x_opt), dtype=float)
                all_X.append(x_opt.reshape(1, -1))
                all_F.append(f_opt.reshape(1, -1))
            except Exception:
                pass

        # Return all evaluated points
        return np.vstack(all_X)

    def _non_dominated_mask(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask
# EVOLVE-BLOCK-END
