import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def eval_func(x):
            nonlocal calls
            if calls >= B:
                return np.inf # Should not happen if loop controlled
            calls += 1
            val = func(x)
            return val

        # Nelder-Mead implementation
        # Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Strategy: 
        # We will run multiple phases.
        # Phase 1: Global exploration with random restarts.
        # Phase 2: Local refinement around the best found.
        
        # Let's allocate budget.
        # Say, 10 restarts of 40 calls each = 400 calls.
        # Remaining 100 calls for final refinement.
        
        num_restarts = 10
        calls_per_restart = max(10, (B - 50) // num_restarts)
        
        for r in range(num_restarts):
            # Initialize Simplex
            # Start point: if r>0, perturb best_x, else random
            if r == 0 or best_x is None:
                x_center = rng.uniform(-5, 5, d)
            else:
                # Perturb best_x
                # Step size decreases with restarts? Or keep moderate.
                # For exploration, keep it somewhat large initially, smaller later.
                # Let's use a fixed perturbation size that shrinks.
                perturb = 2.0 * (0.8 ** r) 
                x_center = best_x + perturb * rng.standard_normal(d)
                x_center = np.clip(x_center, -5, 5)
                
            # Build Simplex
            simplex = []
            # First point is center
            simplex.append(x_center.copy())
            # Other points differ in one coordinate
            for i in range(d):
                p = x_center.copy()
                # Initial step size for simplex
                step = 0.5 
                p[i] += step
                p[i] = np.clip(p[i], -5, 5)
                simplex.append(p)
                
            # Evaluate Simplex
            fvals = []
            for p in simplex:
                if calls >= B:
                    break
                f = eval_func(p)
                fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = p.copy()
            
            if calls >= B:
                break
            
            # Nelder-Mead Loop
            max_iter = calls_per_restart
            for _ in range(max_iter):
                if calls >= B:
                    break
                
                # Sort simplex by fvals
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check convergence: if range of fvals is small
                if fvals[-1] - fvals[0] < 1e-8:
                    break
                    
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = eval_func(xr)
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                if fvals[0] <= fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expand
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = eval_func(xe)
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contract
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = eval_func(xc)
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            break # Shrink
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = eval_func(xc)
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            break # Shrink
                            
                if calls >= B:
                    break
            # End NM Loop for this restart
            
        # Final refinement: Use remaining budget to do a fine-grained search around best_x
        # Or just stop. NM usually converges well.
        # Let's add a final local search with small steps if budget remains.
        remaining = B - calls
        if remaining > 0 and best_x is not None:
            # Simple coordinate descent or small random steps
            step_size = 0.1
            for _ in range(remaining):
                if calls >= B:
                    break
                # Try moving in random direction with small step
                dir = rng.standard_normal(d)
                dir = dir / np.linalg.norm(dir)
                x_new = best_x + step_size * dir
                x_new = np.clip(x_new, -5, 5)
                f_new = eval_func(x_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new
                else:
                    # Reduce step size if we fail often? 
                    # For simplicity, keep constant or reduce slightly.
                    pass
                    
        return best_x, best_f
