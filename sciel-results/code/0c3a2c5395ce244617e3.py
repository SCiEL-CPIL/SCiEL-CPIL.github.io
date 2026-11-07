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

        # Phase 1: Sobol quasi-random sequence for uniform space coverage
        from scipy.integrate import quad
        from scipy.stats import qmc
        
        n_sobol = 150
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sampler.random(n_sobol)
        # Map to bounds
        X_sobol = lo + sobol_points * (hi - lo)
        
        X = X_sobol.copy()
        F = np.array([problem(x) for x in X])
        used = n_sobol
        all_F = F.copy()

        # Phase 2: Hypervolume-contribution-based local search
        remaining_budget = budget - used
        n_local = remaining_budget

        for i in range(n_local):
            if is_deadline():
                break
            if used >= budget:
                break

            # Compute hypervolume contribution for each point
            # Use a simple approximation: for each point, compute the hypervolume
            # that would be lost if that point were removed
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]
            
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))
            
            # Find the point with lowest marginal hypervolume contribution
            # Approximate by distance to reference point in normalized space
            # Points closer to reference have lower contribution
            F_min = F.min(axis=0)
            F_max = F.max(axis=0)
            F_range = F_max - F_min
            F_range[F_range == 0] = 1.0
            F_norm = (F - F_min) / F_range
            
            # Hypervolume contribution proxy: product of (ref - F) for each obj
            # Lower means less contribution
            contrib = np.prod(ref_point - F, axis=1)
            contrib[~nd_mask] = -np.inf  # Dominated points have 0 contribution
            
            # Pick the point with lowest contribution (most redundant)
            target_idx = np.argmin(contrib)
            
            # Generate a perturbation around the target point
            # Use adaptive step size based on how far we've gone
            progress = i / max(1, n_local)
            step_size = 0.2 * (1.0 - 0.7 * progress)
            
            child = X[target_idx].copy()
            child += step_size * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            all_F = np.vstack([all_F, fc])
            
            # Replace if the child is non-dominated or improves hypervolume
            # Simple check: if child is better than target in at least one objective
            # and not worse in all others
            if np.any(fc < F[target_idx]) and not np.all(fc > F[target_idx]):
                X[target_idx] = child
                F[target_idx] = fc

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
