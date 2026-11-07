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
        deadline = start_time + 290  # safety margin before 300s limit

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
            if time.time() > deadline - 30:
                break

            # Progress from 0 to 1
            progress = it / max(1, total_iters - 1)
            # Mutation scale: 0.35 early -> 0.05 late (slightly larger initial)
            mut_scale = 0.35 - 0.30 * progress

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
                if time.time() > deadline - 30:
                    break

                # Select a parent (prefer non-dominated)
                p_idx = rng.integers(len(parents))
                parent = parents[p_idx]

                # Occasional large jump for exploration (10% chance early, 5% late)
                jump_prob = 0.10 - 0.05 * progress
                if rng.random() < jump_prob:
                    # Large exploratory jump: random point near parent
                    child = parent + 0.8 * (hi - lo) * rng.standard_normal(dim)
                else:
                    # Standard adaptive mutation
                    child = parent + mut_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Replace worst solution using crowding distance
                worst_idx = self._find_worst_by_crowding(F, fc)
                X[worst_idx] = child
                F[worst_idx] = fc

        # Final polish: local optimization on best non-dominated solutions
        if time.time() < deadline - 15 and used < budget:
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
        """Find the index to replace. Prefer replacing dominated solutions,
        and among non-dominated, replace the one with lowest crowding distance."""
        n = len(F)
        
        # Check if new_f dominates any existing solutions
        # If so, replace the dominated one with worst crowding
        le = (new_f[None, :] <= F).all(axis=1)
        lt = (new_f[None, :] < F).any(axis=1)
        dominated_by_new = le & lt
        
        if np.any(dominated_by_new):
            # Among those dominated by new, pick the one with worst crowding
            dom_indices = np.where(dominated_by_new)[0]
            if len(dom_indices) == 1:
                return int(dom_indices[0])
            cd_dom = self._crowding_distance(F[dom_indices])
            return int(dom_indices[np.argmin(cd_dom)])
        
        # No solution dominated by new; find worst in current population
        non_dom = self._non_dominated_mask_vec(F)
        dominated_mask = ~non_dom
        
        if np.any(dominated_mask):
            dominated_indices = np.where(dominated_mask)[0]
            # Pick dominated solution with highest sum (worst)
            sums = np.sum(F[dominated_indices], axis=1)
            worst_local = np.argmax(sums)
            return int(dominated_indices[worst_local])
        
        # All non-dominated: replace lowest crowding distance
        cd = self._crowding_distance(F)
        return int(np.argmin(cd))

    def _local_polish(self, problem, X, F, lo, hi, budget, used, deadline):
        """Local polish using scipy minimize on scalarized objectives with structured weights."""
        remaining = budget - used
        if remaining <= 0:
            return X, F, used
        
        non_dom_mask = self._non_dominated_mask_vec(F)
        if not np.any(non_dom_mask):
            return X, F, used
        
        non_dom_indices = np.where(non_dom_mask)[0]
        n_nd = len(non_dom_indices)
        
        # Use up to 6 polish points, each with 2-3 scalarizations
        n_polish = min(n_nd, remaining // 4, 6)
        if n_polish <= 0:
            return X, F, used
        
        cd = self._crowding_distance(F[non_dom_indices])
        top_idx = np.argsort(-cd)[:n_polish]
        
        bounds_list = [(lo[i], hi[i]) for i in range(self.dim)]
        
        # Structured weight vectors: evenly spaced on the 3-simplex
        # For 3 objectives, use weights like (1,0,0), (0,1,0), (0,0,1), (1/3,1/3,1/3), etc.
        weight_sets = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1/3, 1/3, 1/3],
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
        ]
        
        for idx in top_idx:
            if used >= budget:
                break
            if time.time() > deadline - 10:
                break
            
            orig_idx = non_dom_indices[idx]
            x0 = X[orig_idx].copy()
            f0 = F[orig_idx].copy()
            
            # Try 2-3 weight vectors for this point
            n_weights = min(3, max(1, remaining // 2))
            for w_idx in range(n_weights):
                if used >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                
                w = np.array(weight_sets[w_idx % len(weight_sets)])
                
                def scalarized(x, w=w):
                    fx = problem(x)
                    return np.dot(w, fx)
                
                try:
                    res = minimize(scalarized, x0, method='L-BFGS-B', bounds=bounds_list,
                                   options={'maxiter': 30, 'ftol': 1e-10})
                    if res.success and used < budget:
                        fx_new = problem(res.x)
                        used += 1
                        # Accept if it dominates or is non-dominated vs current
                        if np.any(fx_new < f0):
                            X[orig_idx] = res.x
                            F[orig_idx] = fx_new
                            f0 = fx_new
                            x0 = res.x.copy()
                except Exception:
                    pass
        
        return X, F, used
# EVOLVE-BLOCK-END
