import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        # Multi-start Nelder-Mead
        # Strategy:
        # 1. Generate K diverse starting points
        # 2. For each starting point, run a limited Nelder-Mead simplex
        # 3. Keep track of global best
        
        K = 10
        evals_per_start = (budget - K) // K
        # Reserve some evals for final polishing if needed? 
        # Let's just distribute evenly.
        
        # Generate initial points
        # Spread them out in the box [-5, 5]^dim
        # Use a grid-like or random spread
        # Simple: random points, but ensure diversity
        # Or use Halton sequence for better coverage?
        # Let's use random but seeded.
        
        init_points = []
        for k in range(K):
            # Mix of random and structured
            if k == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, size=dim)
            init_points.append(x0)
            
        # Run Nelder-Mead for each start
        # We need to be careful with evaluation budget
        # Standard NM in dim D requires D+1 points.
        # Each iteration takes 1 evaluation (the worst point is replaced).
        # Max iterations per start: max_iter
        
        # Total budget: 500
        # K=10 starts.
        # 10 * (D+1) for initial simplex setup? 
        # Actually, we can just start with a small simplex.
        
        # Let's allocate budget.
        # If we use 10 starts, and we have 500 evals.
        # 50 evals per start.
        # Dim 5. Simplex has 6 points.
        # Initial setup: 6 evals.
        # Remaining 44 evals for iterations.
        
        # To be safe and allow for better starts, let's use fewer starts but more iterations?
        # Or more starts?
        # The function is Step Ellipsoidal. It has a "step" in the landscape.
        # Global optimization is needed to find the right basin.
        # 10 starts seems reasonable.
        
        max_iter_per_start = 40
        
        for k in range(K):
            if evals_used >= budget or time.time() > deadline:
                break
                
            x0 = init_points[k]
            
            # Initialize Simplex
            # Standard NM:
            # x_0, x_0 + alpha*e_i for i=1..D
            
            alpha = 0.1 # Small initial step
            
            simplex = [x0.copy()]
            fs = [func(x0)]
            evals_used += 1
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = x0.copy()
                
            for i in range(dim):
                if evals_used >= budget or time.time() > deadline:
                    break
                xi = x0.copy()
                xi[i] += alpha
                simplex.append(xi)
                fi = func(xi)
                fs.append(fi)
                evals_used += 1
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                    
            # If we didn't finish simplex init, skip
            if len(simplex) < dim + 1:
                continue
                
            # Run NM iterations
            for iter_idx in range(max_iter_per_start):
                if evals_used >= budget or time.time() > deadline:
                    break
                    
                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Check convergence
                if fs[0] == fs[-1]:
                    break
                if np.std(fs) < 1e-10:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                evals_used += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                    
                if fr < fs[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    evals_used += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    if fr < fs[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    evals_used += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < fr:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            evals_used += 1
                            fs[i] = fi
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                        
                        if evals_used >= budget or time.time() > deadline:
                            break

        # Final polishing with local search around best_x
        if evals_used < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            sigma_local = 0.1
            
            # Simple random local search
            for _ in range(budget - evals_used):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_new = x_local + sigma_local * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                evals_used += 1
                
                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    sigma_local *= 1.05
                else:
                    sigma_local *= 0.95
                    
                if sigma_local < 1e-6:
                    sigma_local = 1e-6

        return best_x, best_f
