"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer using Latin Hypercube Sampling for global coverage
    and Nelder-Mead for local refinement on the best scalarized point.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 30s total, aim to finish in 270s (450s limit in some contexts, 
        # but contract says 300s wall clock, so 280s is safe).
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Latin Hypercube Sampling
        # Allocate 80% of budget to global exploration
        n_lhs = int(self.budget * 0.8)
        
        X_lhs = self._generate_lhs(n_lhs, lo, hi)
        
        # Evaluate LHS points
        F_lhs = np.empty((n_lhs, self.n_obj))
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            F_lhs[i] = problem(X_lhs[i])
        
        # If we ran out of time or budget, return what we have
        # (Though return is ignored, we must not exceed budget)
        
        # Phase 2: Nelder-Mead Refinement
        # Allocate remaining budget to local exploitation
        remaining_budget = self.budget - n_lhs
        
        # Find the best point in LHS set based on sum of objectives
        # This is a simple scalarization to pick a starting point for local search
        if n_lhs > 0:
            sums = np.sum(F_lhs, axis=1)
            # Handle potential NaNs or infinities if problem is weird, though contract says finite
            valid_sums = np.where(np.isfinite(sums), sums, np.inf)
            best_idx = np.argmin(valid_sums)
            x_start = X_lhs[best_idx].copy()
            
            # Run Nelder-Mead with strict budget
            if remaining_budget > 0:
                # Define scalar objective for NM
                # We minimize the sum of objectives
                def scalar_obj(x):
                    fx = problem(x)
                    # Clip to bounds as per problem description, though NM might go out
                    # The problem function itself clips, so we just call it.
                    return np.sum(fx)
                
                # Run budget-aware NM
                # We need to track evaluations inside NM
                eval_counter = {'count': 0, 'max': remaining_budget}
                
                def nm_obj(x):
                    if eval_counter['count'] >= eval_counter['max']:
                        # Return a dummy value, the loop will break
                        return np.sum(problem(x)) 
                    eval_counter['count'] += 1
                    return scalar_obj(x)
                
                # Run NM
                self._nelder_mead(x_start, nm_obj, lo, hi, eval_counter)
                
        return X_lhs

    def _generate_lhs(self, n_samples, lo, hi):
        """Generate n_samples points using Latin Hypercube Sampling."""
        dim = self.dim
        # For each dimension, divide into n_samples intervals and pick one point in each
        # This ensures coverage
        X = np.empty((n_samples, dim))
        for d in range(dim):
            # Random permutation of indices 0 to n_samples-1
            perm = np.random.permutation(n_samples)
            # Random uniform within each interval
            # Interval width: (hi - lo) / n_samples
            # Point: lo + (perm + random_offset) * width
            width = (hi[d] - lo[d]) / n_samples
            offsets = np.random.rand(n_samples)
            X[:, d] = lo[d] + (perm + offsets) * width
            
        return X

    def _nelder_mead(self, x0, func, lo, hi, eval_counter):
        """
        Budget-aware Nelder-Mead optimization.
        """
        dim = self.dim
        n_simplex = dim + 1
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        # Initialize simplex
        # x0 is the best point. Generate others by perturbing x0
        simplex = np.zeros((n_simplex, dim))
        simplex[0] = x0
        
        # Calculate initial step size
        # Use 10% of the range as a heuristic for initial simplex size
        step_size = 0.1 * (hi - lo)
        
        for i in range(1, n_simplex):
            simplex[i] = simplex[0].copy()
            simplex[i][i-1] += step_size[i-1]
            # Ensure within bounds
            simplex[i] = np.clip(simplex[i], lo, hi)
            
        # Evaluate initial simplex
        f_simplex = np.zeros(n_simplex)
        for i in range(n_simplex):
            if eval_counter['count'] >= eval_counter['max']:
                return
            f_simplex[i] = func(simplex[i])
            
        # Main loop
        max_iter = 100 # Safety cap on iterations, though eval counter is primary
        for _ in range(max_iter):
            if eval_counter['count'] >= eval_counter['max']:
                break
                
            # Sort simplex by function value
            idx = np.argsort(f_simplex)
            simplex = simplex[idx]
            f_simplex = f_simplex[idx]
            
            x_best = simplex[0]
            f_best = f_simplex[0]
            x_worst = simplex[-1]
            f_worst = f_simplex[-1]
            
            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, lo, hi)
            f_ref = func(x_ref)
            
            if f_ref < f_best:
                # Expansion
                x_exp = centroid + gamma * (centroid - x_worst)
                x_exp = np.clip(x_exp, lo, hi)
                f_exp = func(x_exp)
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_simplex[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
            elif f_ref < f_simplex[-2]:
                # Reflection is good
                simplex[-1] = x_ref
                f_simplex[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (x_worst - centroid)
                x_con = np.clip(x_con, lo, hi)
                f_con = func(x_con)
                
                if f_con < f_worst:
                    simplex[-1] = x_con
                    f_simplex[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n_simplex):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], lo, hi)
                        f_simplex[i] = func(simplex[i])
                        
                        if eval_counter['count'] >= eval_counter['max']:
                            return
# EVOLVE-BLOCK-END
