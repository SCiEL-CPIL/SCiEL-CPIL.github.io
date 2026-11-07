"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using a population-based evolutionary strategy
    with Pareto dominance and adaptive step sizes."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        width = hi - lo

        X_archive = []
        F_archive = []
        eval_count = 0

        def safe_problem(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return [1e10] * self.n_obj
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            if not np.all(np.isfinite(f)):
                f = np.full(self.n_obj, 1e10)
            eval_count += 1
            X_archive.append(x.copy())
            F_archive.append(f.copy())
            return f

        def is_dominated(f_new, F_list):
            """Check if f_new is dominated by any in F_list."""
            for f_old in F_list:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def non_dominated_front(F_list):
            """Return indices of non-dominated points."""
            n = len(F_list)
            if n == 0:
                return []
            F = np.array(F_list)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            return [i for i in range(n) if not dominated[i]]

        # 1. Initial population: stratified sampling for coverage
        n_init = min(20, self.budget)
        pop_X = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            pop_X[:, d] = (perm + np.random.rand(n_init)) / n_init
        pop_X = lo + pop_X * width

        for x in pop_X:
            if eval_count >= self.budget:
                break
            safe_problem(x)

        if eval_count >= self.budget:
            return X_archive

        # 2. Evolutionary loop
        # Maintain a population of size n_pop
        n_pop = min(15, self.budget - eval_count)
        if n_pop < 3:
            while eval_count < self.budget:
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.rand(self.dim)
                safe_problem(x)
            return X_archive

        # Select initial population from archive (non-dominated if possible)
        # We'll work with the archive directly and generate offspring
        # from the current best (non-dominated) points

        # Adaptive step size
        step_scale = 0.15  # fraction of range
        min_step_scale = 0.01
        stagnation_count = 0
        max_stagnation = 10

        # We'll iterate, generating offspring from current non-dominated points
        # until budget is exhausted or time limit reached
        while eval_count < self.budget - 5:
            if time.time() > deadline:
                break

            # Get current non-dominated front
            nd_idx = non_dominated_front(F_archive)
            if not nd_idx:
                # Fallback: use all points
                nd_idx = list(range(len(F_archive)))

            # Limit number of parents to keep computation manageable
            max_parents = min(len(nd_idx), 10)
            # Randomly select parents from non-dominated set
            if len(nd_idx) > max_parents:
                parent_indices = np.random.choice(nd_idx, size=max_parents, replace=False)
            else:
                parent_indices = nd_idx

            # Generate one offspring per parent (or fewer if budget is low)
            n_offspring = min(max_parents, self.budget - eval_count - 5)
            if n_offspring <= 0:
                break

            new_X = []
            new_F = []

            for _ in range(n_offspring):
                # Select a random parent
                pi = parent_indices[np.random.randint(len(parent_indices))]
                parent_x = X_archive[pi]

                # Gaussian perturbation
                step = step_scale * width
                child_x = parent_x + np.random.randn(self.dim) * step
                child_x = np.clip(child_x, lo, hi)

                # Evaluate
                f_child = safe_problem(child_x)
                if eval_count >= self.budget:
                    break

                new_X.append(child_x)
                new_F.append(f_child)

            # Check for improvement: did any new point enter the non-dominated front?
            if new_F:
                all_F = F_archive + new_F
                nd_idx_new = non_dominated_front(all_F)
                # If any new point (index >= len(F_archive)) is in the front, we improved
                improved = any(idx >= len(F_archive) for idx in nd_idx_new)
                if improved:
                    stagnation_count = 0
                    # Slightly increase step scale for exploration
                    step_scale = min(step_scale * 1.1, 0.3)
                else:
                    stagnation_count += 1
                    if stagnation_count >= max_stagnation:
                        # Reset and reduce step scale
                        step_scale = max(step_scale * 0.7, min_step_scale)
                        stagnation_count = 0

        # 3. Final random fill if budget remains
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(self.dim)
            safe_problem(x)

        return X_archive
# EVOLVE-BLOCK-END
