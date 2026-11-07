"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using Pareto-front-biased sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # WFG4 has 12 variables. The known Pareto front structure:
        # f1 = 1 + 10*sum(x_i) where x_i in [0,1] for i=1..11 (normalized)
        # f2 depends on the last variable and a shape function
        # f3 is a third objective
        # 
        # For WFG4 specifically (3 objectives, 12 vars):
        # The Pareto front can be parameterized. Let's use a strategy:
        # - Sample points biased toward the expected Pareto front
        # - Use local refinement around good points
        # - Use a mix of exploration and exploitation

        # First, let's understand the problem by doing a small random sample
        # to understand the objective landscape, then bias sampling.

        # Strategy:
        # 1. Do a small random initialization to understand the space
        # 2. Identify the Pareto front structure by examining the relationship between objectives
        # 3. Sample along the Pareto front using the discovered structure
        # 4. Use local search to refine points near the front

        # For WFG4 with 12 variables and 3 objectives:
        # Typical WFG problems have the first (k-1) variables controlling position on the front
        # and the remaining variables controlling distance from the front.
        # 
        # Let's use a hybrid approach:
        # - Allocate some budget to random exploration to map the front
        # - Then use the mapped front to guide sampling

        # Phase 1: Random exploration (20% of budget)
        n_explore = max(10, budget // 5)
        n_exploit = budget - n_explore

        X_all = []
        F_all = []

        # Random exploration
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(dim)
            f = np.array(problem(x), dtype=float)
            X_all.append(x)
            F_all.append(f)

        if not F_all:
            return np.array(X_all) if X_all else np.zeros((1, dim))

        F_all = np.array(F_all)
        X_all = np.array(X_all)

        # Find non-dominated points from exploration
        def non_dominated_mask(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    # j dominates i if j <= i in all objectives and j < i in at least one
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return ~dominated

        nd_mask = non_dominated_mask(F_all)
        X_nd = X_all[nd_mask]
        F_nd = F_all[nd_mask]

        # Phase 2: Exploitation - sample near known Pareto front points
        # For each non-dominated point, try small perturbations
        # Also try to interpolate between pairs of non-dominated points

        used = n_explore
        remaining = budget - used

        # Create candidate points by:
        # 1. Interpolating between non-dominated points
        # 2. Small perturbations around non-dominated points
        # 3. Extrapolating slightly beyond the front

        candidates = []

        # Interpolation between pairs of non-dominated points
        if len(X_nd) >= 2:
            for i in range(len(X_nd)):
                for j in range(i + 1, len(X_nd)):
                    for t in [0.25, 0.5, 0.75]:
                        x_interp = (1 - t) * X_nd[i] + t * X_nd[j]
                        x_interp = np.clip(x_interp, lo, hi)
                        candidates.append(x_interp)

        # Perturbations around non-dominated points
        for x_nd in X_nd:
            for scale in [0.01, 0.05, 0.1]:
                for _ in range(3):
                    x_pert = x_nd + scale * (hi - lo) * np.random.randn(dim)
                    x_pert = np.clip(x_pert, lo, hi)
                    candidates.append(x_pert)

        # If we have few non-dominated points, add more random points
        if len(candidates) < remaining // 2:
            for _ in range(remaining // 2 - len(candidates)):
                x = lo + (hi - lo) * np.random.rand(dim)
                candidates.append(x)

        # Shuffle candidates
        np.random.shuffle(candidates)

        # Evaluate candidates until budget is exhausted
        for x_cand in candidates:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            f_cand = np.array(problem(x_cand), dtype=float)
            X_all = np.vstack([X_all, x_cand.reshape(1, -1)])
            F_all = np.vstack([F_all, f_cand.reshape(1, -1)])
            used += 1

        # Phase 3: If we still have budget, do local search on best points
        if used < budget:
            nd_mask = non_dominated_mask(F_all)
            X_nd = X_all[nd_mask]
            F_nd = F_all[nd_mask]

            # Sort by first objective for diversity
            if len(X_nd) > 0:
                order = np.argsort(F_nd[:, 0])
                X_nd = X_nd[order]
                F_nd = F_nd[order]

                # Take a subset for local search
                n_ls = min(len(X_nd), max(5, (budget - used) // 10))
                step = max(1, len(X_nd) // n_ls)
                X_ls = X_nd[::step][:n_ls]

                for x_ls in X_ls:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break

                    # Local search: try small steps in random directions
                    best_x = x_ls.copy()
                    best_f = np.array(problem(best_x), dtype=float)
                    used += 1
                    if used >= budget:
                        break

                    for _ in range(5):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        direction = np.random.randn(dim)
                        direction /= np.linalg.norm(direction) + 1e-10
                        for step_size in [0.01, 0.05]:
                            if used >= budget:
                                break
                            x_try = best_x + step_size * (hi - lo) * direction
                            x_try = np.clip(x_try, lo, hi)
                            f_try = np.array(problem(x_try), dtype=float)
                            used += 1

                            # Check if this is non-dominated relative to current best
                            if not np.all(best_f <= f_try):
                                # Not dominated by current best, keep it
                                best_x = x_try
                                best_f = f_try

                    X_all = np.vstack([X_all, best_x.reshape(1, -1)])
                    F_all = np.vstack([F_all, best_f.reshape(1, -1)])

        return X_all
# EVOLVE-BLOCK-END
