"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead with gradient polish for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        call_count = 0
        
        # Helper to evaluate with budget tracking
        def evaluate(x):
            nonlocal call_count
            if call_count >= budget:
                return None
            call_count += 1
            return func(x)
        
        # === Phase 1: Nelder-Mead Simplex Method ===
        # Initialize simplex around origin
        x0 = np.zeros(n)
        f0 = evaluate(x0)
        if f0 is None:
            return x0, 0.0
        
        # Initial step size - exploratory
        alpha = 1.0
        simplex = [x0.copy()]
        for i in range(n):
            xi = x0.copy()
            xi[i] += alpha
            simplex.append(xi)
        
        # Evaluate all simplex points
        fvals = [f0]
        for i in range(n):
            fi = evaluate(simplex[i])
            if fi is None:
                break
            fvals.append(fi)
        
        # If we ran out of budget during initialization, return best
        if len(fvals) < n + 1:
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]
        
        # Nelder-Mead parameters
        alpha_nm = 1.0   # reflection
        gamma = 2.0      # expansion
        rho = 0.5        # contraction
        sigma = 0.5      # shrink
        
        max_nm_iter = 300  # Safety cap on iterations
        
        for iter_num in range(max_nm_iter):
            # Time check
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            f_range = fvals[-1] - fvals[0]
            x_range = np.max([np.linalg.norm(simplex[-1] - simplex[0]), 
                             np.max([np.max(np.abs(simplex[i] - simplex[0])) for i in range(n)])])
            if f_range < 1e-10 and x_range < 1e-6:
                break
            
            # Check if we have enough budget for next iteration
            # Each iteration needs at most 2 evaluations (reflection + expansion/contraction)
            if call_count + 3 > budget:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha_nm * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = evaluate(x_ref)
            if f_ref is None:
                break
            
            if f_ref < fvals[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = evaluate(x_exp)
                if f_exp is None:
                    # Out of budget, use reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                    break
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                if f_ref < fvals[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = evaluate(x_con)
                if f_con is None:
                    break
                if f_con <= max(f_ref, fvals[-1]):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi_shrink = evaluate(simplex[i])
                        if fi_shrink is None:
                            break
                        fvals[i] = fi_shrink
                    if fi_shrink is None:
                        break
            
            # Early stop if we've used most of our budget
            if call_count > budget * 0.7:
                break
        
        # Find best point from Nelder-Mead
        best_idx = np.argmin(fvals)
        x_best = simplex[best_idx].copy()
        f_best = fvals[best_idx]
        
        # === Phase 2: Gradient Polish with Central Differences ===
        # Use remaining budget for fine-tuning
        remaining = budget - call_count
        if remaining > 2 * n + 1:  # Need at least 2n+1 for central diff gradient + one step
            # Central difference gradient estimation
            eps = 1e-6
            lr = 0.1  # Learning rate for polish
            min_lr = 1e-8
            
            for polish_iter in range(50):  # Cap polish iterations
                if time.time() > deadline:
                    break
                if call_count + 2 * n + 1 > budget:
                    break
                
                # Estimate gradient using central differences
                grad = np.zeros(n)
                for i in range(n):
                    x_plus = x_best.copy()
                    x_plus[i] += eps
                    x_minus = x_best.copy()
                    x_minus[i] -= eps
                    
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    
                    f_plus = evaluate(x_plus)
                    f_minus = evaluate(x_minus)
                    if f_plus is None or f_minus is None:
                        break
                    grad[i] = (f_plus - f_minus) / (2 * eps)
                
                if f_plus is None or f_minus is None:
                    break
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-12:
                    break
                
                # Gradient descent step with adaptive learning rate
                x_new = x_best - lr * grad
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = evaluate(x_new)
                if f_new is None:
                    break
                
                if f_new < f_best:
                    x_best = x_new
                    f_best = f_new
                    # Slightly increase learning rate
                    lr = min(lr * 1.1, 1.0)
                else:
                    # Decrease learning rate
                    lr *= 0.7
                    if lr < min_lr:
                        break
        
        return x_best, f_best
# EVOLVE-BLOCK-END
