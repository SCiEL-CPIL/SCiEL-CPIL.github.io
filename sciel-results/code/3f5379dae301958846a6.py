"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Latin Hypercube Sampling (LHS)
        # We use a fixed number of LHS points to ensure coverage.
        # Let's allocate 60% of the budget to LHS, and 40% to local refinement.
        n_lhs = int(self.budget * 0.6)
        n_refine_budget = self.budget - n_lhs
        
        # Generate LHS points
        # For each dimension, we divide [lo, hi] into n_lhs equal intervals and pick one point in each.
        # To make it deterministic and reproducible per seed (though seed is set by evaluator),
        # we use np.random.permutation.
        rng = np.random.default_rng(0)
        
        X_lhs = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            # Create intervals
            interval_width = (hi[d] - lo[d]) / n_lhs
            # Pick random positions within each interval
            # Or just permutation of indices and scale
            perm = rng.permutation(n_lhs)
            # x = lo + (perm + 0.5) * width  -> Center of intervals
            # Or x = lo + (perm * interval_width) + rng.random() * interval_width
            X_lhs[:, d] = lo[d] + (perm + 0.5) * interval_width
            
        # Evaluate LHS points
        F_lhs = np.array([problem(x) for x in X_lhs])
        
        # 2. Local Refinement using Nelder-Mead on Non-Dominated Points
        # We need to identify non-dominated points from the LHS set.
        # Since n_lhs is ~120, finding ND set is O(n^2) which is fine.
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            is_nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nondom[i]:
                    continue
                for j in range(n):
                    if i == j or not is_nondom[j]:
                        continue
                    if dominates(F[j], F[i]):
                        is_nondom[i] = False
                        break
            return np.where(is_nondom)[0]

        nd_idx = non_dominated_indices(F_lhs)
        
        # We have a limited budget for refinement.
        # We will pick the top K non-dominated points based on some metric (e.g., distance to ideal or random) 
        # and run Nelder-Mead on them.
        # Nelder-Mead is for single-objective. We need to adapt it or use it on a scalarized objective.
        # Common strategy: Minimize a weighted sum or a specific scalarization.
        # Given WFG7 is a 3-objective problem, we can try to minimize the sum of objectives 
        # or use a reference-point-based scalarization.
        # Let's try minimizing the sum of objectives for simplicity, or perhaps the max deviation from ideal.
        # Actually, a better approach for MOO with Nelder-Mead is to minimize a scalar function 
        # that guides the search towards the Pareto front.
        # Let's use the sum of objectives as the scalar objective for Nelder-Mead.
        
        # Select candidates for refinement
        # We have n_refine_budget evaluations left.
        # Nelder-Mead in 12D might be expensive. 
        # Standard NM uses (dim+1) points for the simplex.
        # Each iteration involves 1 evaluation.
        # Let's allow a fixed number of iterations per point, say 50.
        # If we refine 5 points, that's 250 evaluations. We only have ~80.
        # So we can refine about 1-2 points deeply, or more points shallowly.
        
        # Let's pick the 3 best non-dominated points (e.g., smallest sum) and refine them.
        if len(nd_idx) > 0:
            # Sort ND points by sum of objectives
            sums = np.sum(F_lhs[nd_idx], axis=1)
            sorted_nd = nd_idx[np.argsort(sums)]
            
            # How many to refine?
            # Let's say we refine the top 2.
            # Budget per point: n_refine_budget // 2
            n_refine_points = min(2, len(sorted_nd))
            evals_per_point = n_refine_budget // n_refine_points if n_refine_points > 0 else 0
            
            for i in range(n_refine_points):
                if time.time() > deadline:
                    break
                
                start_idx = sorted_nd[i]
                x0 = X_lhs[start_idx].copy()
                
                # Define scalar objective: sum of objectives
                def scalar_obj(x):
                    f = problem(x)
                    return np.sum(f)
                
                # Run Nelder-Mead
                # scipy.optimize.minimize with method='Nelder-Mead'
                # We need to be careful with bounds. NM doesn't handle bounds well.
                # We can clip inside the objective function or use a penalty.
                # The problem(x) already clips. So x can go out of bounds, but problem handles it.
                # However, NM might wander far. Let's hope the landscape is smooth enough.
                
                try:
                    # maxiter is the number of function calls
                    # We have evals_per_point available
                    # Note: NM consumes 1 eval per iteration roughly, but setup costs dim+1.
                    # So effective iterations are less.
                    # Let's just set maxiter to evals_per_point.
                    res = scipy.optimize.minimize(
                        scalar_obj, 
                        x0, 
                        method='Nelder-Mead',
                        options={'maxiter': evals_per_point, 'xatol': 1e-5, 'fatol': 1e-5}
                    )
                    
                    # Record the result
                    # The result res.x is the best point found.
                    # We should add this to our recorded set?
                    # The task says "Every returned objective vector is recorded".
                    # The __call__ return value is ignored, but problem(x) calls are recorded.
                    # So we just need to ensure we don't exceed budget.
                    # The local variables X and F are not returned, only the side effect of recording.
                    
                except Exception:
                    pass
        
        return None

# Need to import scipy
import scipy.optimize
# EVOLVE-BLOCK-END
