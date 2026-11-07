"""Multi-objective optimizer for RE34 using MOEA/D with L-BFGS-B subproblem solving."""
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """MOEA/D-style decomposition-based multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        width = hi - lo
        mid = (lo + hi) / 2.0

        ref = np.array([1705.0, 12.0, 0.3])

        X_archive = []
        F_archive = []
        eval_count = 0

        def safe_problem(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return np.full(self.n_obj, 1e10)
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            if not np.all(np.isfinite(f)):
                f = np.full(self.n_obj, 1e10)
            eval_count += 1
            X_archive.append(x.copy())
            F_archive.append(f.copy())
            return f

        def non_dominated_front(F_list):
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

        # Phase 1: Initial diverse sampling
        n_init = min(20, self.budget)
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            lhs_points[:, d] = (perm + np.random.rand(n_init)) / n_init
        X_init = lo + lhs_points * width
        for x in X_init:
            if eval_count >= self.budget:
                break
            safe_problem(x)

        if eval_count >= self.budget:
            return X_archive

        # Phase 2: Generate weight vectors for decomposition
        # Use a structured grid on the simplex for 3 objectives
        n_grid = 4  # 4 points per edge -> reasonable number of subproblems
        weights = []
        for w1 in np.linspace(0, 1, n_grid):
            for w2 in np.linspace(0, 1 - w1, max(1, int(round((1 - w1) * n_grid)) + 1)):
                w3 = 1.0 - w1 - w2
                if w3 >= -1e-10:
                    w3 = max(0.0, w3)
                    w = np.array([w1, w2, w3])
                    w_sum = np.sum(w)
                    if w_sum > 0:
                        w = w / w_sum
                        weights.append(w)

        # Add extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)

        # Add center
        weights.append(np.ones(self.n_obj) / self.n_obj)

        # Remove near-duplicates
        unique_weights = []
        for w in weights:
            is_dup = False
            for uw in unique_weights:
                if np.allclose(w, uw, atol=0.05):
                    is_dup = True
                    break
            if not is_dup:
                unique_weights.append(w)
        weights = unique_weights
        np.random.shuffle(weights)

        n_weights = len(weights)
        remaining = self.budget - eval_count

        if remaining < 10:
            while eval_count < self.budget:
                if time.time() > deadline:
                    break
                x = lo + width * np.random.rand(self.dim)
                safe_problem(x)
            return X_archive

        # Phase 3: Solve each subproblem with L-BFGS-B
        # Budget allocation: give each weight a share of remaining budget
        budget_per_weight = max(2, remaining // n_weights)

        for w in weights:
            if eval_count >= self.budget - 2:
                break
            if time.time() > deadline:
                break

            # Tchebycheff aggregation: max_i (w_i * (f_i - z_i))
            # Use reference point as ideal approximation
            w_eff = w / ref

            # Find best starting point: minimize weighted sum over archive
            if F_archive:
                F_arr = np.array(F_archive)
                F_scaled = F_arr / ref
                scores = F_scaled @ w
                best_idx = np.argmin(scores)
                x0 = X_archive[best_idx].copy()
            else:
                x0 = mid.copy()

            # Try 2 starting points: best from archive + perturbation
            n_starts = 2
            for s in range(n_starts):
                if eval_count >= self.budget - 2:
                    break
                if time.time() > deadline:
                    break

                if s == 0:
                    x_start = x0
                else:
                    x_start = x0 + 0.1 * width * (np.random.rand(self.dim) - 0.5)
                    x_start = np.clip(x_start, lo, hi)

                def tcheb_obj(x, w_eff=w_eff):
                    f = safe_problem(x)
                    return np.max(w_eff * f)

                try:
                    max_iter = max(1, budget_per_weight // n_starts)
                    res = minimize(tcheb_obj, x_start, method='L-BFGS-B',
                                   bounds=bounds,
                                   options={'maxiter': max_iter, 'maxls': 5, 'ftol': 1e-12})
                except Exception:
                    pass

        if time.time() > deadline:
            return X_archive

        # Phase 4: Evolutionary refinement of the non-dominated front
        nd_idx = non_dominated_front(F_archive)
        if not nd_idx:
            nd_idx = list(range(len(F_archive)))

        step_scale = 0.1
        min_step = 0.01
        stagnation = 0
        max_stag = 6

        while eval_count < self.budget - 2:
            if time.time() > deadline:
                break

            nd_idx = non_dominated_front(F_archive)
            if not nd_idx:
                nd_idx = list(range(len(F_archive)))

            max_parents = min(len(nd_idx), 8)
            if len(nd_idx) > max_parents:
                parent_indices = np.random.choice(nd_idx, size=max_parents, replace=False)
            else:
                parent_indices = nd_idx

            n_offspring = min(max_parents, self.budget - eval_count - 2)
            if n_offspring <= 0:
                break

            improved = False
            for _ in range(n_offspring):
                if eval_count >= self.budget - 2:
                    break
                if time.time() > deadline:
                    break

                pi = parent_indices[np.random.randint(len(parent_indices))]
                parent_x = X_archive[pi]
                child_x = parent_x + np.random.randn(self.dim) * (step_scale * width)
                child_x = np.clip(child_x, lo, hi)
                f_child = safe_problem(child_x)

                # Check if child is non-dominated
                is_dominated = False
                for f_old in F_archive:
                    if np.all(f_old <= f_child) and np.any(f_old < f_child):
                        is_dominated = True
                        break
                if not is_dominated:
                    improved = True

            if improved:
                stagnation = 0
                step_scale = min(step_scale * 1.1, 0.25)
            else:
                stagnation += 1
                if stagnation >= max_stag:
                    step_scale = max(step_scale * 0.6, min_step)
                    stagnation = 0

        # Phase 5: Final L-BFGS-B polish on top non-dominated points
        if time.time() < deadline and eval_count < self.budget - 1:
            nd_idx = non_dominated_front(F_archive)
            if nd_idx:
                n_polish = min(4, len(nd_idx), max(1, (self.budget - eval_count - 1) // 3))
                if n_polish > 0:
                    polish_indices = np.random.choice(nd_idx, size=n_polish, replace=False)
                    for idx in polish_indices:
                        if eval_count >= self.budget - 1:
                            break
                        if time.time() > deadline:
                            break

                        x0 = X_archive[idx].copy()
                        f0 = F_archive[idx].copy()

                        # Weight based on the point's position
                        f_pos = f0 / ref
                        w = f_pos / np.sum(f_pos) if np.sum(f_pos) > 0 else np.ones(self.n_obj) / self.n_obj
                        w_eff = w / ref

                        def tcheb_obj(x, w_eff=w_eff):
                            f = safe_problem(x)
                            return np.max(w_eff * f)

                        try:
                            max_iter = max(1, (self.budget - eval_count - 1) // n_polish)
                            minimize(tcheb_obj, x0, method='L-BFGS-B',
                                     bounds=bounds,
                                     options={'maxiter': max_iter, 'maxls': 5, 'ftol': 1e-12})
                        except Exception:
                            pass

        # Phase 6: Fill remaining budget
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + width * np.random.rand(self.dim)
            safe_problem(x)

        return X_archive
