"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Global-Local Hybrid: Random Sampling + Multi-start Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget, rng, step_size=0.1):
        dim = self.dim
        n = dim
        
        # Initialize simplex
        simplex = []
        fvals = []
        
        x0 = np.clip(x0, -5, 5)
        f0 = func(x0)
        simplex.append(x0.copy())
        fvals.append(f0)
        
        for i in range(n):
            xi = x0.copy()
            xi[i] += step_size
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            fvals.append(fi)
            
        best_x = x0.copy()
        best_f = f0
        evals = n + 1
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while evals < budget:
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Convergence check
            if max(fvals) - min(fvals) < 1e-12:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            x_worst = simplex[-1]
            f_worst = fvals[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            evals += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                evals += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    x_c = centroid + 0.5 * (x_r - centroid)
                else:
                    x_c = centroid - 0.5 * (x_worst - centroid)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                evals += 1
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrinkage
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        evals += 1
                        if evals >= budget:
                            break
                    if evals >= budget:
                        break
                    
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()
                
        return best_x, best_f, evals

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        # Phase 1: Global Sampling
        n_samples = 20
        sample_budget = min(n_samples, total_budget)
        
        points = []
        fvals = []
        
        # Include origin and random points
        x_origin = np.zeros(dim)
        f_origin = func(x_origin)
        points.append(x_origin)
        fvals.append(f_origin)
        
        for _ in range(sample_budget - 1):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            points.append(x)
            fvals.append(f)
            
        # Sort by function value
        indices = np.argsort(fvals)
        points = [points[i] for i in indices]
        fvals = [fvals[i] for i in indices]
        
        best_x = points[0].copy()
        best_f = fvals[0]
        remaining_budget = total_budget - sample_budget
        
        # Phase 2: Local Refinement on top candidates
        n_refine = 3
        budget_per_refine = remaining_budget // n_refine
        
        for i in range(n_refine):
            if remaining_budget <= 0:
                break
            
            # Use the i-th best point from sampling as start
            x_start = points[i]
            
            # Run Nelder-Mead
            x_res, f_res, evals_used = self._nelder_mead(func, x_start, budget_per_refine, rng)
            
            remaining_budget -= evals_used
            
            if f_res < best_f:
                best_f = f_res
                best_x = x_res
                
        return best_x, best_f
# EVOLVE-BLOCK-END
