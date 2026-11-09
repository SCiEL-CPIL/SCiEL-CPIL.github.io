"""Rastrigin f3 optimizer exploiting separable periodic structure with robust global search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer that exploits the separable periodic structure of Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_fx = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Search using Coarse-to-Fine Grid and Random Restarts
        # We use a set of candidate points to evaluate.
        # 1. Origin
        # 2. Random points
        # 3. Grid points around the best found so far
        
        # Initial candidates
        candidates = [np.zeros(dim)]
        
        # Add some random restarts to find different basins
        for _ in range(10):
            candidates.append(rng.uniform(-5, 5, dim))
        
        # Evaluate all candidates first to find a good starting point
        for c in candidates:
            if calls >= budget - 100 or time.time() > deadline - 10:
                break
            evaluate(c)
        
        if best_fx == float('inf'):
            return best_x, best_fx

        # Now perform coordinate-wise refinement from the best point found
        x = best_x.copy()
        fx = best_fx

        # Coarse-to-fine coordinate descent
        # Steps: 1.0, 0.2, 0.05, 0.01, 0.002
        steps = [1.0, 0.2, 0.05, 0.01, 0.002, 0.0005]
        
        for step in steps:
            if calls >= budget - 50 or time.time() > deadline - 5:
                break
            
            improved = False
            # Iterate through dimensions multiple times for this step size
            for _pass in range(2):
                if calls >= budget - 50 or time.time() > deadline - 5:
                    break
                for d in range(dim):
                    if calls >= budget - 50 or time.time() > deadline - 5:
                        break
                    
                    # Try a small grid around the current coordinate
                    # Number of points depends on step size and remaining budget
                    # We want to cover +/- 2*step roughly
                    num_points = 5
                    # If step is large, we might want to look further, but Rastrigin period is 2*pi ~ 6.28
                    # So +/- 2*step is reasonable for local refinement.
                    # For the first step (1.0), +/- 2.0 is good.
                    
                    center = x[d]
                    # Ensure we don't go out of bounds
                    lower = max(-5.0, center - 2 * step)
                    upper = min(5.0, center + 2 * step)
                    
                    if lower >= upper:
                        continue
                        
                    grid = np.linspace(lower, upper, num_points)
                    
                    best_val = fx
                    best_coord = x[d]
                    
                    for g in grid:
                        if calls >= budget - 50 or time.time() > deadline - 5:
                            break
                        trial = x.copy()
                        trial[d] = g
                        ft = evaluate(trial)
                        if ft < best_val:
                            best_val = ft
                            best_coord = g
                    
                    if best_coord != x[d]:
                        x[d] = best_coord
                        fx = best_val
                        improved = True

            # If no improvement in a full pass with this step size, we can break early
            # or just continue to next smaller step.
            if not improved:
                # Try a random jump to escape potential local minima if step is large
                if step > 0.05 and calls < budget - 100:
                    # Random restart from current best
                    x_new = x + rng.uniform(-1, 1, dim) * step
                    x_new = np.clip(x_new, -5, 5)
                    fx_new = evaluate(x_new)
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        improved = True
                # If still no improvement, we might be at a local minimum for this resolution
                # Continue to finer resolution

        # Phase 2: Nelder-Mead Polish
        if calls < budget - 10 and time.time() < deadline - 5:
            x_nm = best_x.copy()
            f_nm = best_fx
            
            # Build simplex
            simplex = [x_nm.copy()]
            fvals = [f_nm]
            
            # Initial step size for NM
            # Use a small value, e.g., 0.1 or based on previous step
            step_nm = 0.1 
            for i in range(dim):
                if calls >= budget:
                    break
                pert = x_nm.copy()
                # Perturb in dimension i
                if x_nm[i] + step_nm > 5.0:
                    pert[i] -= step_nm
                elif x_nm[i] - step_nm < -5.0:
                    pert[i] += step_nm
                else:
                    pert[i] += step_nm
                pert = np.clip(pert, -5.0, 5.0)
                
                f_pert = evaluate(pert)
                simplex.append(pert)
                fvals.append(f_pert)
            
            if len(simplex) < dim + 1:
                return best_x, best_fx

            # NM Iterations
            max_nm_iters = 200
            for _ in range(max_nm_iters):
                if time.time() > deadline - 5 or calls >= budget:
                    break
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check convergence
                if np.std(fvals) < 1e-12:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5.0, 5.0)
                fr = evaluate(xr)
                
                if calls >= budget:
                    break
                
                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5.0, 5.0)
                    fe = evaluate(xe)
                    if calls < budget:
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    # Try inside contraction if worst is worse than centroid
                    beta = 0.5
                    if fvals[-1] < centroid and False: # Logic check: standard NM uses outside/inside
                        pass
                    
                    # Standard Contraction
                    # If fr >= f[-2], contract
                    # Inside contraction if worst is better than centroid? No, usually outside if worst is bad.
                    # Let's stick to standard:
                    # If fr >= f[-2], contraction.
                    # If f[-1] < centroid, inside contraction, else outside.
                    
                    if fvals[-1] < np.mean(fvals[:-1]):
                        # Inside contraction
                        xc = np.clip(centroid + beta * (simplex[-1] - centroid), -5.0, 5.0)
                    else:
                        # Outside contraction (actually standard NM often just does inside if reflection fails, 
                        # or outside if reflection was worse than worst? 
                        # Let's use standard: 
                        # If fr >= f[-2], contraction.
                        # xc = centroid + 0.5 * (worst - centroid)  (Inside)
                        # If that fails, shrink.
                        # Many implementations just do inside contraction.
                        xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5.0, 5.0)
                    
                    fc = evaluate(xc)
                    if calls >= budget:
                        break
                    
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls >= budget:
                                break
                            simplex[i] = np.clip(0.5 * (simplex[i] + simplex[0]), -5.0, 5.0)
                            fvals[i] = evaluate(simplex[i])
                
                # Update best
                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
