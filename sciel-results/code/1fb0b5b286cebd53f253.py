"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead optimizer with multi-restart strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        
        x_best_global = rng.uniform(-5, 5, n)
        f_best_global = func(x_best_global)
        eval_count = 1
        
        # Parameters for Nelder-Mead
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Multi-restart strategy
        # We divide the budget into chunks. 
        # If no improvement is made in a chunk, we restart with a new random center or larger step.
        max_evals_per_restart = 50
        initial_step = 1.0
        
        while eval_count < total_budget:
            # Determine starting point for this restart
            # If we just started, use x_best_global. Otherwise, use x_best_global.
            # To ensure exploration, we can sometimes restart from a new random point.
            # Let's stick to restarting from the current best but with a fresh simplex.
            # If the simplex shrinks too much, we might get stuck.
            
            # Create initial simplex
            # Point 0 is x_best_global
            # Other points are x_best_global + initial_step * e_i
            simplex = [x_best_global.copy()]
            for i in range(n):
                p = x_best_global.copy()
                p[i] += initial_step
                p = np.clip(p, -5, 5)
                simplex.append(p)
            
            # Evaluate initial simplex
            fvals = []
            for p in simplex:
                if eval_count >= total_budget:
                    break
                f = func(p)
                fvals.append(f)
                eval_count += 1
                if f < f_best_global:
                    f_best_global = f
                    x_best_global = p.copy()
            
            if len(fvals) < n + 1:
                break
                
            # Nelder-Mead Loop
            # We have n+1 points
            # We will run for a limited number of iterations or until budget is hit
            # To avoid infinite loops or getting stuck in tiny steps, we limit iterations per restart
            max_iters = 30 # Approx max_evals_per_restart / (n+1)
            
            for _ in range(max_iters):
                if eval_count >= total_budget:
                    break
                    
                # Sort simplex by function value
                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]
                
                # Check for convergence (simple heuristic)
                if fvals[-1] - fvals[0] < 1e-8:
                    break
                    
                # Centroid of the best n points (excluding the worst)
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                eval_count += 1
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    eval_count += 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    eval_count += 1
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            eval_count += 1
                            if eval_count >= total_budget:
                                break
                        
                        if eval_count >= total_budget:
                            break
                
                # Update global best
                for i in range(n + 1):
                    if fvals[i] < f_best_global:
                        f_best_global = fvals[i]
                        x_best_global = simplex[i].copy()
                        
            # After a restart block, if we haven't used all budget, loop again
            # We might want to increase initial_step if we are stuck, but for simplicity, 
            # restarting from the same best point with a fresh simplex (which might be different 
            # due to previous modifications) is standard. However, if the simplex collapsed, 
            # we should restart from a random point or larger step.
            # For now, let's just restart the loop. The simplex is rebuilt from scratch 
            # using x_best_global and initial_step. This acts as a reset.
            # If initial_step is too small, we might stall. If too large, we might oscillate.
            # A fixed initial_step of 1.0 is reasonable for [-5,5]^5.
            
        return x_best_global, f_best_global
# EVOLVE-BLOCK-END
