# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using Tchebycheff scalarizations and simulated annealing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)

        # Vectorized domination check
        F_i = F[:, np.newaxis, :]  # (n, 1, n_obj)
        F_j = F[np.newaxis, :, :]  # (1, n, n_obj)

        leq = np.all(F_i <= F_j, axis=2)  # (n, n)
        lt = np.any(F_i < F_j, axis=2)    # (n, n)

        dominates = leq & lt
        dominated = np.any(dominates, axis=0)

        return ~dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        ref_point = np.array([1705.0, 12.0, 0.3])

        def time_left():
            return deadline - time.time()

        def is_deadline():
            return time.time() > deadline - 5

        # Phase 1: Tchebycheff scalarization with local optimization
        # Generate weight vectors on the simplex
        n_weights = 120
        # Use random points on the simplex
        W = rng.random((n_weights, n_obj))
        W /= W.sum(axis=1, keepdims=True)

        # Also add some structured weights
        structured_W = np.zeros((n_obj, n_obj))
        for i in range(n_obj):
            structured_W[i, i] = 1.0
        W = np.vstack([W, structured_W])

        # For each weight vector, run a local optimization from a random starting point
        all_X = []
        all_F = []
        used = 0

        def tchebycheff_objective(x, w, F_min, F_range):
            """Tchebycheff scalarization objective."""
            F = problem(x)
            # Normalize objectives
            F_norm = (F - F_min) / F_range
            # Tchebycheff: max_i w_i * F_norm_i
            return np.max(w * F_norm)

        # Initial F_min and F_range estimates
        # Use a few random points to estimate
        init_X = lo + (hi - lo) * rng.random((10, dim))
        init_F = np.array([problem(x) for x in init_X])
        used += 10
        F_min = init_F.min(axis=0)
        F_max = init_F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0

        # For each weight vector, run Nelder-Mead or similar
        from scipy.optimize import minimize

        n_starts_per_weight = 2
        for w in W:
            if used >= budget or is_deadline():
                break

            for _ in range(n_starts_per_weight):
                if used >= budget or is_deadline():
                    break

                # Random starting point
                x0 = lo + (hi - lo) * rng.random(dim)

                # Define the scalarized objective
                def obj(x):
                    nonlocal used
                    if is_deadline():
                        return 1e10
                    F = problem(x)
                    used += 1
                    F_norm = (F - F_min) / F_range
                    return np.max(w * F_norm)

                # Use Nelder-Mead for derivative-free optimization
                try:
                    res = minimize(
                        obj,
                        x0,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 30, 'xatol': 0.01, 'fatol': 0.001}
                    )
                    if res.x is not None and not np.any(np.isnan(res.x)):
                        x_opt = np.clip(res.x, lo, hi)
                        F_opt = problem(x_opt)
                        used += 1
                        all_X.append(x_opt)
                        all_F.append(F_opt)
                except Exception:
                    pass

        # Convert to arrays
        if len(all_X) > 0:
            X = np.array(all_X)
            F = np.array(all_F)
        else:
            X = init_X
            F = init_F

        # Phase 2: Simulated annealing refinement
        if used < budget and not is_deadline():
            # Get non-dominated points
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) > 0:
                # Start from each non-dominated point
                for idx in nd_idx[:min(20, len(nd_idx))]:
                    if used >= budget or is_deadline():
                        break

                    x_current = X[idx].copy()
                    F_current = F[idx].copy()
                    T = 0.5  # Initial temperature
                    T_min = 0.01
                    cooling_rate = 0.95
                    max_iter = min(50, (budget - used) // 20)

                    for _ in range(max_iter):
                        if used >= budget or is_deadline():
                            break
                        if T < T_min:
                            break

                        # Generate perturbation
                        scale = T * 0.3
                        x_new = x_current + scale * (hi - lo) * rng.standard_normal(dim)
                        x_new = np.clip(x_new, lo, hi)

                        F_new = problem(x_new)
                        used += 1

                        # Check if new point is non-dominated or improves hypervolume
                        # Simple acceptance: accept if not dominated by current, or with probability
                        dominates_current = np.all(F_new <= F_current) and np.any(F_new < F_current)
                        dominated_by_current = np.all(F_current <= F_new) and np.any(F_current < F_new)

                        if dominates_current:
                            x_current = x_new
                            F_current = F_new
                        elif not dominated_by_current:
                            # Non-dominated with respect to current: accept with probability
                            # Compute a simple score difference
                            score_current = np.sum(F_current / ref_point)
                            score_new = np.sum(F_new / ref_point)
                            delta = score_new - score_current
                            if delta < 0 or rng.random() < np.exp(-delta / T):
                                x_current = x_new
                                F_current = F_new

                        T *= cooling_rate

                    # Add the best point found from this start
                    if not np.all(np.isnan(F_current)):
                        all_X.append(x_current)
                        all_F.append(F_current)

            # Rebuild arrays
            if len(all_X) > 0:
                X = np.array(all_X)
                F = np.array(all_F)

        # Phase 3: Final local refinement of best points
        if used < budget and not is_deadline():
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) > 0:
                # Refine the top few non-dominated points
                for idx in nd_idx[:min(5, len(nd_idx))]:
                    if used >= budget or is_deadline():
                        break

                    x0 = X[idx].copy()

                    # Use a weighted sum with weights based on the point's position
                    F_min = F.min(axis=0)
                    F_max = F.max(axis=0)
                    F_range = F_max - F_min
                    F_range[F_range == 0] = 1.0
                    F_norm = (F[idx] - F_min) / F_range

                    # Invert: points with lower normalized objectives get higher weight
                    w = 1.0 / (F_norm + 0.1)
                    w /= np.sum(w)

                    def obj(x):
                        nonlocal used
                        if is_deadline():
                            return 1e10
                        F = problem(x)
                        used += 1
                        F_norm_local = (F - F_min) / F_range
                        return np.dot(w, F_norm_local)

                    try:
                        res = minimize(
                            obj,
                            x0,
                            method='Nelder-Mead',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 20, 'xatol': 0.005, 'fatol': 0.0005}
                        )
                        if res.x is not None and not np.any(np.isnan(res.x)):
                            x_opt = np.clip(res.x, lo, hi)
                            F_opt = problem(x_opt)
                            used += 1
                            all_X.append(x_opt)
                            all_F.append(F_opt)
                    except Exception:
                        pass

        # Final result
        if len(all_X) > 0:
            X = np.array(all_X)
            F = np.array(all_F)

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
