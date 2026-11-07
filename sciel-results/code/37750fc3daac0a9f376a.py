# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using a deterministic front-refinement
    strategy: sample a dense set of points on the Pareto front, then use
    gradient-free local searches to refine hypervolume contribution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # WFG7 specific: the Pareto front lies on a curve in objective space.
        # For WFG7 with 12 variables and 3 objectives, we can parameterize
        # the front and sample it densely, then refine.
        #
        # Strategy:
        # 1. Generate a diverse initial population using a structured grid
        #    in the parameter space of the Pareto front.
        # 2. Evaluate them.
        # 3. Use remaining budget for local refinements (Nelder-Mead style)
        #    around the best hypervolume contributors.

        # WFG7 front parameterization:
        # For WFG7, the Pareto front can be parameterized. With 3 objectives
        # and 12 variables, we use a simplex-based parameterization.
        #
        # Actually, for WFG7 the front is complex. Let's use a different approach:
        # We'll use a hybrid: start with a well-spread population, then do
        # adaptive local searches.

        # Phase 1: Initial structured population
        # Use a Latin hypercube or grid to ensure good spread
        n_initial = min(40, self.budget)
        rng = np.random.default_rng(0)

        # Create a structured initial population
        # Use a combination of random and structured points
        X_init = np.zeros((n_initial, self.dim))
        F_init = np.zeros((n_initial, self.n_obj))

        # First 10 points: vertices and midpoints of the bounding box
        for i in range(min(10, n_initial)):
            if i == 0:
                X_init[i] = lo
            elif i == 1:
                X_init[i] = hi
            else:
                # Random but structured
                X_init[i] = lo + span * rng.random(self.dim)

        # Remaining: Latin hypercube-like
        if n_initial > 10:
            n_lh = n_initial - 10
            for i in range(n_lh):
                # Each dimension gets a unique random value in [0,1]
                X_init[10 + i] = lo + span * rng.random(self.dim)

        # Evaluate initial population
        used = 0
        for i in range(n_initial):
            F_init[i] = problem(X_init[i])
            used += 1

        # Phase 2: Local refinement using Nelder-Mead style search
        # Focus on points that contribute to hypervolume (non-dominated points)

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(F, idx):
            for j in range(len(F)):
                if j != idx and dominates(F[j], F[idx]):
                    return False
            return True

        # Find non-dominated points
        nd_indices = [i for i in range(len(F_init)) if is_non_dominated(F_init, i)]

        # For each non-dominated point, do a local search
        # Use Nelder-Mead to minimize a weighted sum (Tchebycheff or scalarized)
        # This helps refine the front

        remaining_budget = self.budget - used

        # Simple local search: for each ND point, try small perturbations
        # and keep improvements
        all_X = [X_init[i].copy() for i in range(n_initial)]
        all_F = [F_init[i].copy() for i in range(n_initial)]

        # Sort ND points by their objective values (spread)
        # Do local search on a subset to conserve budget
        n_local = min(len(nd_indices), remaining_budget // 15)  # ~15 evals per local search

        # For local search, use a simple coordinate descent or Nelder-Mead
        # on a scalarized objective
        for ni in range(n_local):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            idx = nd_indices[ni % len(nd_indices)]
            x0 = all_X[idx].copy()
            f0 = all_F[idx].copy()

            # Use Nelder-Mead on a Tchebycheff scalarization
            # Weighted toward the "middle" of the front
            weights = np.ones(self.n_obj) / self.n_obj

            def scalarized(x):
                fx = problem(x)
                # Tchebycheff: max of weighted deviations
                # Use a reference point
                ref = np.array([3.0, 5.0, 7.0])
                return np.max(weights * (fx - ref))

            # Simple Nelder-Mead implementation
            x_best = x0.copy()
            f_best = scalarized(x0)
            used += 1  # count the initial evaluation (already done, but we re-evaluate)
            # Actually, we already have f0, so let's be more careful

            # Let's do a simpler local search: try perturbations in each direction
            step_sizes = [0.1 * span, 0.05 * span, 0.01 * span]
            for step in step_sizes:
                if used >= self.budget:
                    break
                for d in range(self.dim):
                    if used >= self.budget:
                        break
                    for direction in [-1, 1]:
                        if used >= self.budget:
                            break
                        x_try = x_best.copy()
                        x_try[d] += direction * step[d]
                        x_try = np.clip(x_try, lo, hi)
                        f_try = problem(x_try)
                        used += 1

                        # Check if this improves hypervolume contribution
                        # Simple check: is it non-dominated and closer to ideal?
                        if all(np.all(f_try <= f) or np.any(f_try < f) for f in all_F):
                            # It's non-dominated w.r.t. current set
                            # Accept if it's better in at least one objective
                            if np.any(f_try < f_best):
                                x_best = x_try
                                f_best = f_try
                                all_X[idx] = x_best
                                all_F[idx] = f_best

        # Phase 3: Final random sampling to fill remaining budget
        while used < self.budget:
            if time.time() > deadline:
                break
            x_rand = lo + span * rng.random(self.dim)
            f_rand = problem(x_rand)
            used += 1
            all_X.append(x_rand)
            all_F.append(f_rand)

        # Return the population
        return np.array(all_X)
# EVOLVE-BLOCK-END
