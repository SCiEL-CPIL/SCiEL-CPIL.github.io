"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Multi-Start Nelder-Mead Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, f0):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline or calls_made >= budget - 5:
                return x0, f0
            
            # Initialize simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            # Standard NM initialization
            for i in range(dim):
                new_x = x0.copy()
                new_x[i] += 0.5 if new_x[i] < 0 else -0.5
                new_x = np.clip(new_x, -5, 5)
                if np.allclose(new_x, x0):
                    new_x[i] += 0.1
                f_new = safe_call(new_x)
                simplex.append(new_x)
                simplex_f.append(f_new)
                
                if time.time() > deadline or calls_made >= budget - 5:
                    return best_x, best_f

            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Run NM until stagnation or budget/time limit
            # We allow a significant portion of the budget for this local refinement
            max_iter = 100 
            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget - 5:
                    break
                
                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Check for convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-8 * (1 + abs(simplex_f[0])):
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_call(x_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_call(x_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - 0.5 * (x_ref - centroid)
                        
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_call(x_con)
                    
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = safe_call(simplex[i])
                            
                # Update global best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f

        # Phase 1: Initial Exploration
        # Use a deterministic grid + random sampling to find a good starting point quickly
        # For dim=5, a full grid is too large, so we use a stratified approach.
        
        # 1. Origin
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        
        # 2. Axis-aligned points (positive and negative)
        for i in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[i] = sign * 4.0
                if time.time() < deadline and calls_made < budget - 50:
                    f = safe_call(x)
                    if f < f0:
                        x0 = x
                        f0 = f
                        
        # 3. Random sampling to catch shifted minima
        # Budget allocation: ~10% for initial exploration
        explore_budget = int(budget * 0.1)
        while calls_made < explore_budget and time.time() < deadline:
            x = np.random.uniform(-5, 5, dim)
            f = safe_call(x)
            if f < f0:
                x0 = x
                f0 = f
                
        # Phase 2: Local Refinement
        # Run NM from the best point found
        nelder_mead(x0, f0)
        
        # Phase 3: Adaptive Multi-Start
        # If we still have budget, start new NM runs from points near the current best
        # or from other promising areas.
        # We generate perturbations of the best point and run NM if they are better than current best.
        
        # Strategy: Generate 5-10 new starting points.
        # Some close to best (local), some far (global).
        n_starts = 10
        for i in range(n_starts):
            if time.time() > deadline or calls_made >= budget - 20:
                break
            
            # Mix of local and global restarts
            if i < n_starts // 2:
                # Local: perturb best
                scale = 1.0 / (i + 1)
                x_new = best_x + np.random.normal(0, scale, dim)
            else:
                # Global: random
                x_new = np.random.uniform(-5, 5, dim)
                
            x_new = np.clip(x_new, -5, 5)
            f_new = safe_call(x_new)
            
            # Only refine if this point is significantly better than current best
            # or if we haven't found a good solution yet
            if f_new < best_f * 0.9 or best_f > 10: 
                nelder_mead(x_new, f_new)
            elif f_new < best_f:
                # Even if not huge improvement, if it's the new best, refine it
                nelder_mead(x_new, f_new)

        return best_x, best_f
# EVOLVE-BLOCK-END
