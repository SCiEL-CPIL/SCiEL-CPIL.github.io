# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using structured exploration and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)

        F_i = F[:, np.newaxis, :]  # (n, 1, n_obj)
        F_j = F[np.newaxis, :, :]  # (1, n, n_obj)

        leq = np.all(F_i <= F_j, axis=2)  # (n, n)
        lt = np.any(F_i < F_j, axis=2)    # (n, n)

        dominates = leq & lt
        dominated = np.any(dominates, axis=0)

        return ~dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(123)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        ref_point = np.array([1705.0, 12.0, 0.3])

        def time_left():
            return deadline - time.time()

        def is_deadline():
            return time.time() > deadline - 10

        # Phase 1: Structured initialization
        # Use a grid-like sampling with random perturbations
        # 5 variables, bounds [1, 3] for each
        n_grid = 4  # 4 points per dimension -> 4^5 = 1024 total, too many
        # Use 3 per dim: 3^5 = 243 points
        n_per_dim = 3
        grid_points = []
        for d1 in range(n_per_dim):
            for d2 in range(n_per_dim):
                for d3 in range(n_per_dim):
                    for d4 in range(n_per_dim):
                        for d5 in range(n_per_dim):
                            x = np.array([
                                lo[0] + (hi[0] - lo[0]) * d1 / (n_per_dim - 1),
                                lo[1] + (hi[1] - lo[1]) * d2 / (n_per_dim - 1),
                                lo[2] + (hi[2] - lo[2]) * d3 / (n_per_dim - 1),
                                lo[3] + (hi[3] - lo[3]) * d4 / (n_per_dim - 1),
                                lo[4] + (hi[4] - lo[4]) * d5 / (n_per_dim - 1)
                            ])
                            grid_points.append(x)
        grid_points = np.array(grid_points)
        
        # Shuffle and take first budget//3 points for initial population
        n_initial = min(len(grid_points), budget // 3)
        indices = rng.permutation(len(grid_points))[:n_initial]
        X = grid_points[indices]
        
        F = np.array([problem(x) for x in X])
        used = n_initial
        all_F = F.copy()

        # Phase 2: Evolutionary search with adaptive mutation
        pop_size = min(40, budget // 4)
        search_budget = int(budget * 0.70)

        while used < search_budget:
            if is_deadline():
                break

            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(F[:, 0]), np.argmin(F[:, 1]), np.argmin(F[:, 2])])
                nd_idx = np.unique(nd_idx)

            progress = used / search_budget
            mut_scale = 0.30 * (0.992 ** used)
            mut_scale = max(mut_scale, 0.015)

            n_children = min(20, search_budget - used)
            for _ in range(n_children):
                if is_deadline():
                    break
                if used >= search_budget:
                    break

                # Parent selection: 75% from non-dominated, 25% random
                if rng.random() < 0.75:
                    parent_idx = nd_idx[rng.integers(len(nd_idx))]
                else:
                    parent_idx = rng.integers(len(F))

                parent = X[parent_idx].copy()

                # Mutation: Gaussian with occasional larger jumps
                if rng.random() < 0.08:
                    jump_scale = mut_scale * 4.0
                else:
                    jump_scale = mut_scale

                child = parent + jump_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                all_F = np.vstack([all_F, fc])

                # Replacement: find worst point in population
                worst_idx = self._find_worst_index(F, ref_point)
                
                child_better = np.any(fc < F[worst_idx])
                child_not_worse_all = not np.all(fc > F[worst_idx])

                if child_better and child_not_worse_all:
                    X[worst_idx] = child
                    F[worst_idx] = fc

        # Phase 3: Local refinement with weighted scalarization
        if used < budget and time_left() > 30:
            from scipy.optimize import minimize

            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) > 0:
                # Generate weight vectors on the simplex
                n_weights = min(12, max(3, (budget - used) // 3))
                W = rng.random((n_weights, n_obj))
                W /= W.sum(axis=1, keepdims=True)

                # Refine more non-dominated points with more iterations
                points_to_refine = nd_idx[:min(8, len(nd_idx))]
                for pt_idx in points_to_refine:
                    if used >= budget or time_left() < 30:
                        break

                    x0 = X[pt_idx].copy()

                    for w in W[:4]:
                        if used >= budget or time_left() < 30:
                            break

                        eval_count = [0]

                        def obj(x, w=w):
                            eval_count[0] += 1
                            return np.dot(w, problem(x))

                        try:
                            res = minimize(obj, x0, method='L-BFGS-B',
                                          bounds=list(zip(lo, hi)),
                                          options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-8})
                            if res.success and eval_count[0] > 0:
                                fx = problem(res.x)
                                used += eval_count[0] + 1
                                all_F = np.vstack([all_F, fx])

                                if np.any(fx < F[pt_idx]):
                                    X[pt_idx] = res.x
                                    F[pt_idx] = fx
                        except:
                            used += eval_count[0]

        return X

    def _find_worst_index(self, F, ref_point):
        """Find the index of the worst point in the population."""
        if len(F) == 0:
            return 0

        nd_mask = self._non_dominated_mask(F)

        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0
        F_norm = (F - F_min) / F_range

        dist = np.sqrt(np.sum(F_norm ** 2, axis=1))
        penalty = np.where(~nd_mask, 50.0, 0.0)

        quality = dist + penalty
        return np.argmax(quality)
# EVOLVE-BLOCK-END
