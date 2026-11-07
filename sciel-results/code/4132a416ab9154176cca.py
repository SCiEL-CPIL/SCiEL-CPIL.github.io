"""Rewritten program using Nelder-Mead simplex optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimizer for the BBOB-style f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Adaptive Nelder-Mead with better initial simplex and convergence handling
        # For "attractive sector" structure, start near origin with moderate spread
        
        # Initial point: try a few starting points to get a good initial simplex
        # The attractive sector typically has a basin of attraction around some point
        # Start with origin and build simplex with adaptive step size
        
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_used = 1
        
        # Build initial simplex with step size based on box size
        # Use step = 0.5 for initial exploration (box is [-5,5])
        step = 0.5
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            simplex.append(xi)
            simplex_f.append(func(xi))
            calls_used += 1
        
        # Sort simplex by function value
        def sort_simplex():
            nonlocal simplex, simplex_f
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]
        
        sort_simplex()
        best_x = simplex[0].copy()
        best_f = simplex_f[0]
        
        # Nelder-Mead parameters - tuned for better convergence
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        # Adaptive parameters for handling stagnation
        stagnation_count = 0
        max_stagnation = 20
        shrink_factor = 0.7
        
        # Track history for adaptive step size
        f_history = [best_f]
        
        while calls_used < budget:
            # Check convergence
            f_spread = simplex_f[-1] - simplex_f[0]
            x_spread = np.max(np.abs(np.array(simplex) - np.array(simplex[0])))
            
            if f_spread < 1e-12 and x_spread < 1e-10:
                # Stagnation: try to escape with adaptive perturbation
                stagnation_count += 1
                if stagnation_count > max_stagnation:
                    # Reset with larger perturbation
                    perturbation = 0.3 * (1.0 + stagnation_count / 50.0)
                    new_x = best_x + perturbation * np.random.randn(dim)
                    new_x = np.clip(new_x, -5, 5)
                    new_f = func(new_x)
                    calls_used += 1
                    
                    if new_f < best_f:
                        best_x = new_x.copy()
                        best_f = new_f
                        f_history.append(best_f)
                    
                    # Rebuild simplex around new best
                    simplex = [best_x.copy()]
                    simplex_f = [best_f]
                    step = 0.3 * (1.0 + stagnation_count / 100.0)
                    for i in range(dim):
                        xi = best_x.copy()
                        xi[i] += step
                        xi = np.clip(xi, -5, 5)
                        simplex.append(xi)
                        simplex_f.append(func(xi))
                        calls_used += 1
                        if calls_used >= budget:
                            break
                    
                    sort_simplex()
                    stagnation_count = 0
                    continue
                else:
                    # Small perturbation
                    simplex[0] = simplex[0] + 0.05 * np.random.randn(dim)
                    simplex[0] = np.clip(simplex[0], -5, 5)
                    simplex_f[0] = func(simplex[0])
                    calls_used += 1
                    sort_simplex()
                    continue
            
            # Reset stagnation counter if we made progress
            if f_spread > 1e-8:
                stagnation_count = 0
            
            # Compute centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_used += 1
            
            if f_r < simplex_f[0]:
                # Try expansion
                if calls_used >= budget:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                    sort_simplex()
                    break
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_used += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if calls_used >= budget:
                    break
                # Outside contraction if f_r < f_w, else inside contraction
                if f_r < simplex_f[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_used += 1
                
                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink - adaptive shrink factor
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + shrink_factor * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= budget:
                            break
                    if calls_used >= budget:
                        break
            
            # Update best
            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best_f:
                best_f = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()
                f_history.append(best_f)
            
            sort_simplex()
            
            # Periodic reset if no progress in last 50 calls
            if len(f_history) > 50 and f_history[-1] == f_history[-50]:
                # No progress in 50 calls, try a different strategy
                # Move best point slightly and rebuild simplex
                perturbation = 0.2
                new_x = best_x + perturbation * np.random.randn(dim)
                new_x = np.clip(new_x, -5, 5)
                new_f = func(new_x)
                calls_used += 1
                
                if new_f < best_f:
                    best_x = new_x.copy()
                    best_f = new_f
                    f_history.append(best_f)
                
                # Rebuild simplex
                simplex = [best_x.copy()]
                simplex_f = [best_f]
                step = 0.4
                for i in range(dim):
                    xi = best_x.copy()
                    xi[i] += step
                    xi = np.clip(xi, -5, 5)
                    simplex.append(xi)
                    simplex_f.append(func(xi))
                    calls_used += 1
                    if calls_used >= budget:
                        break
                
                sort_simplex()
                f_history = [best_f]
        
        return best_x, best_f
# EVOLVE-BLOCK-END
