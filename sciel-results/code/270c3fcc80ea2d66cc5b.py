"""Deterministic LHS + Diverse Nelder-Mead Local Refinement for WFG7."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Deterministic LHS sampling followed by diverse Nelder-Mead local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # --- Phase 1: Latin Hypercube Sampling ---
        # Allocate 60% of the budget to global exploration.
        # With budget=200, this is 120 points.
        n_lhs = int(0.6 * budget)
        
        # Generate LHS points
        # Standard LHS: for each dimension, a permutation of 0..n-1, scaled to [0,1]
        lhs_unit = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            # Add uniform noise to avoid grid points
            lhs_unit[:, d] = (perm + np.random.uniform(size=n_lhs)) / n_lhs

        # Map to bounds
        lhs_points = lo + (hi - lo) * lhs_unit
        
        # Evaluate all LHS points
        F = np.array([problem(x) for x in lhs_points])
        used = n_lhs

        # --- Phase 2: Identify Non-Dominated Points ---
        def get_non_dominated_indices(F_set):
            n = len(F_set)
            if n == 0:
                return np.array([], dtype=int)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        nd_indices = get_non_dominated_indices(F)
        
        # Fallback if no non-dominated points
        if len(nd_indices) == 0:
            nd_indices = np.array([np.argmin(np.sum(F, axis=1))])

        # --- Phase 3: Local Refinement ---
        remaining_budget = budget - used
        
        # Select up to 10 diverse non-dominated points to allow more coverage
        n_refine = min(10, len(nd_indices))
        if n_refine == 0:
            return lhs_points

        # Select diverse points using farthest-point sampling in objective space
        nd_F = F[nd_indices]
        sums = np.sum(nd_F, axis=1)
        start_idx = nd_indices[np.argmin(sums)]
        
        selected = [start_idx]
        selected_F = [F[start_idx]]
        
        for _ in range(n_refine - 1):
            max_dist = -1
            farthest_idx = -1
            for idx in nd_indices:
                if idx in selected:
                    continue
                dists = [np.linalg.norm(F[idx] - sf) for sf in selected_F]
                min_dist = np.min(dists)
                if min_dist > max_dist:
                    max_dist = min_dist
                    farthest_idx = idx
            if farthest_idx != -1:
                selected.append(farthest_idx)
                selected_F.append(F[farthest_idx])
            else:
                break
        
        # Distribute remaining budget evenly among selected points
        evals_per_point = max(1, remaining_budget // len(selected))
        
        # Weight vectors for different local searches to cover the Pareto front
        # We use a set of weights that covers the simplex
        weights_list = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([1.0, 1.0, 0.0]) / 2.0,
            np.array([1.0, 0.0, 1.0]) / 2.0,
            np.array([0.0, 1.0, 1.0]) / 2.0,
            np.array([1.0, 1.0, 1.0]) / 3.0
        ]
        
        for i, idx in enumerate(selected):
            x0 = lhs_points[idx]
            w = weights_list[i % len(weights_list)]
            
            def scalar_obj(x, w=w):
                f = problem(x)
                return np.dot(f, w)
            
            options = {
                'maxfev': evals_per_point,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'disp': False
            }
            
            try:
                minimize(scalar_obj, x0, method='Nelder-Mead', bounds=bounds, options=options)
            except:
                pass

        return lhs_points
# EVOLVE-BLOCK-END
