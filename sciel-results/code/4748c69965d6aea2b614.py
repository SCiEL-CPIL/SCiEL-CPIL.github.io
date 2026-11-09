"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with coordinate-descent polish and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Initial point
        x = np.zeros(self.dim)
        fx = func(x)
        
        # Nelder-Mead setup
        # Simplex vertices
        simplex = np.zeros((self.dim + 1, self.dim))
        fvals = np.zeros(self.dim + 1)
        
        # First vertex is x
        simplex[0] = x
        fvals[0] = fx
        
        # Initialize other vertices
        for i in range(1, self.dim + 1):
            simplex[i] = x.copy()
            simplex[i][i-1] += 0.5  # Small offset in each dimension
            fvals[i] = func(simplex[i])
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        calls_made = 1 + self.dim  # 1 initial + dim for simplex init
        
        # Main Nelder-Mead loop
        while calls_made < self.budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Check for convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = func(x_ref)
            calls_made += 1
            
            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                f_exp = func(x_exp)
                calls_made += 1
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Just replace worst with reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                f_con = func(x_con)
                calls_made += 1
                
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrinkage
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                    
                    # Check deadline after shrinkage
                    if time.time() > deadline:
                        break
        
        # Get best point from simplex
        best_idx = np.argmin(fvals)
        x_best = simplex[best_idx].copy()
        fx_best = fvals[best_idx]
        
        # Coordinate descent polish
        step_size = 0.1
        while calls_made < self.budget and time.time() < deadline:
            improved = False
            for i in range(self.dim):
                # Try positive step
                x_plus = x_best.copy()
                x_plus[i] += step_size
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = func(x_plus)
                calls_made += 1
                
                if f_plus < fx_best:
                    x_best = x_plus
                    fx_best = f_plus
                    improved = True
                elif time.time() > deadline:
                    return x_best, fx_best
                
                # Try negative step
                x_minus = x_best.copy()
                x_minus[i] -= step_size
                x_minus = np.clip(x_minus, -5, 5)
                f_minus = func(x_minus)
                calls_made += 1
                
                if f_minus < fx_best:
                    x_best = x_minus
                    fx_best = f_minus
                    improved = True
                elif time.time() > deadline:
                    return x_best, fx_best
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        return x_best, fx_best
# EVOLVE-BLOCK-END
