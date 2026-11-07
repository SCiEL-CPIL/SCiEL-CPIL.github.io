# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE34 using adaptive mutation with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        # Initial population: 30 random points
        init_pop = 30
        X = lo + (hi - lo) * rng.random((init_pop, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = init_pop

        # Adaptive mutation scale: start large, shrink over time
        n_children_per_iter = 10
        total_iters = max(1, (budget - init_pop) // n_children_per_iter)

        for it in range(total_iters):
            # Check time limit
            if time.time() > deadline - 40:
                break

            # Progress from 0 to 1
            progress = it / max(1, total_iters - 1)
            # Mutation scale: 0.3 early -> 0.05 late
            mut_scale = 0.3 - 0.25 * progress

            # Find non-dominated solutions (vectorized)
            non_dom_mask = self._non_dominated_mask_vec(F)
            if np.any(non_dom_mask):
                parents = X[non_dom_mask]
            else:
                parents = X

            # Generate children
            for _ in range(n_children_per_iter):
                if used >= budget:
                    break
                if time.time() > deadline - 40:
                    break

                # Select a parent (prefer non-dominated)
                p_idx = rng.integers(len(parents))
                parent = parents[p_idx]

                # Mutate with adaptive scale
                child = parent + mut_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Replace worst solution using crowding distance
                worst_idx = self._find_worst_by_crowding(F, fc)
                X[worst_idx] = child
                F[worst_idx] = fc

        # Final polish: local optimization on best non-dominated solutions
        if time.time() < deadline - 10 and used < budget:
            X, F, used = self._local_polish(problem, X, F, lo, hi, budget, used, deadline)

        return X

    def _non_dominated_mask_vec(self, F):
        """Vectorized non-dominated mask. Returns True for non-dominated solutions."""
        n = len(F)
        if n <= 1:
            return np.ones(n, dtype=bool)
        
        le = (F[np.newaxis, :, :] <= F[:, np.newaxis, :]).all(axis=2)
        lt = (F[np.newaxis, :, :] < F[:, np.newaxis, :]).any(axis=2)
        dom = le & lt
        dominated = dom.any(axis=1)
        return ~dominated

    def _crowding_distance(self, F):
        """Compute crowding distance for each solution in F."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        n_obj = F.shape[1]
        cd = np.zeros(n)
        
        for m in range(n_obj):
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            fmin = F[order[0], m]
            fmax = F[order[-1], m]
            if fmax - fmin > 1e-15:
                for i in range(1, n - 1):
                    cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / (fmax - fmin)
        
        return cd

    def _find_worst_by_crowding(self, F, new_f):
        """Find the index to replace."""
        n = len(F)
        
        non_dom = self._non_dominated_mask_vec(F)
        dominated_mask = ~non_dom
        
        if np.any(dominated_mask):
            dominated_indices = np.where(dominated_mask)[0]
            sums = np.sum(F[dominated_indices], axis=1)
            worst_local = np.argmax(sums)
            return dominated_indices[worst_local]
        
        cd = self._crowding_distance(F)
        return int(np.argmin(cd))

    def _local_polish(self, problem, X, F, lo, hi, budget, used, deadline):
        """Local polish using scipy minimize on scalarized objectives."""
        remaining = budget - used
        if remaining <= 0:
            return X, F, used
        
        non_dom_mask = self._non_dominated_mask_vec(F)
        if not np.any(non_dom_mask):
            return X, F, used
        
        non_dom_indices = np.where(non_dom_mask)[0]
        
        n_polish = min(len(non_dom_indices), remaining // 3, 5)
        if n_polish <= 0:
            return X, F, used
        
        cd = self._crowding_distance(F[non_dom_indices])
        top_idx = np.argsort(-cd)[:n_polish]
        
        bounds_list = [(lo[i], hi[i]) for i in range(self.dim)]
        
        for idx in top_idx:
            if used >= budget:
                break
            if time.time() > deadline - 5:
                break
            
            orig_idx = non_dom_indices[idx]
            x0 = X[orig_idx].copy()
            f0 = F[orig_idx].copy()
            
            for _ in range(2):
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                w = np.random.dirichlet(np.ones(self.n_obj))
                
                def scalarized(x, w=w):
                    fx = problem(x)
                    return np.dot(w, fx)
                
                try:
                    res = minimize(scalarized, x0, method='L-BFGS-B', bounds=bounds_list,
                                   options={'maxiter': 20, 'ftol': 1e-8})
                    if res.success and used < budget:
                        fx_new = problem(res.x)
                        used += 1
                        if np.any(fx_new < f0) or np.all(fx_new <= f0):
                            X[orig_idx] = res.x
                            F[orig_idx] = fx_new
                            f0 = fx_new
                except Exception:
                    pass
        
        return X, F, used
# EVOLVE-BLOCK-END
