"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Coordinate Descent with Nelder-Mead polish for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        rng = np.random.default_rng(42)
        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
                calls[0] += 1
                return val
            except:
                calls[0] += 1
                return float('inf')

        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(x0, max_iter=200):
            nonlocal best_x, best_f, calls, budget
            if calls[0] >= budget or time.time() > deadline:
                return x0, best_f
            
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [safe_call(x0)]
            update_best(x0, fvals[0])
            
            for i in range(1, dim + 1):
                if calls[0] >= budget or time.time() > deadline:
                    break
                xi = x0.copy()
                step = 0.1 if i == 1 else 0.2
                xi[i - 1] += step
                simplex.append(xi)
                fi = safe_call(xi)
                fvals.append(fi)
                update_best(xi, fi)
            
            if len(simplex) < dim + 1:
                return x0, best_f

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            for it in range(max_iter):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-10:
                    break
                
                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_call(xr)
                update_best(xr, fr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_call(xe)
                    update_best(xe, fe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = safe_call(xc)
                        update_best(xc, fc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fvals[i] = safe_call(simplex[i])
                                update_best(simplex[i], fvals[i])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = safe_call(xc)
                        update_best(xc, fc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fvals[i] = safe_call(simplex[i])
                                update_best(simplex[i], fvals[i])
            
            # Return best point from simplex
            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        def adaptive_coordinate_descent(x0, max_iters=100):
            nonlocal best_x, best_f, calls, budget
            x = x0.copy()
            f = safe_call(x)
            update_best(x, f)
            
            # Adaptive step sizes per dimension
            steps = np.ones(dim) * 1.0
            
            for it in range(max_iters):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                improved = False
                for i in range(dim):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    
                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += steps[i]
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    f_plus = safe_call(x_plus)
                    update_best(x_plus, f_plus)
                    
                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        steps[i] *= 1.5
                        improved = True
                        continue
                    
                    # Try negative step
                    x_minus = x.copy()
                    x_minus[i] -= steps[i]
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    f_minus = safe_call(x_minus)
                    update_best(x_minus, f_minus)
                    
                    if f_minus < f:
                        x = x_minus
                        f = f_minus
                        steps[i] *= 1.5
                        improved = True
                        continue
                    
                    # No improvement, shrink step
                    steps[i] *= 0.5
                
                if not improved:
                    # If no dimension improved, check if steps are too small
                    if np.min(steps) < 1e-6:
                        break
                    # Optional: random restart within small neighborhood if stuck
                    # But for ridge, usually one dimension will keep improving
                    # Let's just break if we're really stuck
                    if np.all(steps < 1e-4):
                        break
            
            return x, f

        # Main loop
        # 1. Global exploration with a few random starts
        # 2. Adaptive Coordinate Descent
        # 3. Nelder-Mead Polish
        
        # Initial random points to find a good basin
        x_current = rng.uniform(-4, 4, dim)
        
        # Run Adaptive Coordinate Descent
        x_current, f_current = adaptive_coordinate_descent(x_current, max_iters=50)
        
        # If budget remains, try a few more random starts if we haven't converged well
        # f13 is unimodal-ish but shifted/rotated. ACD is good but might miss the global minimum if initial point is bad.
        # However, 500 budget is small. Let's stick to one main trajectory but add a polish.
        
        # Nelder-Mead Polish
        if calls[0] < budget - 10:
            x_polished, f_polished = nelder_mead(x_current, max_iter=100)
            update_best(x_polished, f_polished)
            
            # If NM improved, do another round of ACD with smaller initial steps?
            # Or just another NM polish?
            if calls[0] < budget - 10:
                x_polished2, f_polished2 = nelder_mead(x_polished, max_iter=50)
                update_best(x_polished2, f_polished2)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
