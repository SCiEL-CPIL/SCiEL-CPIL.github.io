import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: stop at 290s to avoid 330s kill
        deadline = start_time + 290
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        dim = self.dim

        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                return best_f
            if calls_made >= self.budget:
                return best_f
            
            calls_made += 1
            f = func(x)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return f

        # Phase 1: Global Search (Latin Hypercube or Random)
        # f11 is smooth and unimodal (shifted/rotated/offset). 
        # A good initial point is crucial. 
        # We use a small number of random points to find a decent starting region.
        n_global = min(10, self.budget)
        
        if n_global == 0:
             return best_x, best_f

        # Generate points
        # Using a simple random start is often sufficient for 5D if the function is smooth.
        # But to be safe against bad luck, let's try a few.
        starts = []
        for _ in range(n_global):
            if time.time() > deadline: break
            x = np.random.uniform(-5, 5, dim)
            f = wrapped_func(x)
            starts.append((f, x))
            
        if best_x is None:
            x0 = np.zeros(dim)
            f0 = wrapped_func(x0)
            best_x = x0
            best_f = f0
            return best_x, best_f

        # Sort and pick the best start
        starts.sort(key=lambda item: item[0])
        x0, f0 = starts[0]

        # Phase 2: Nelder-Mead
        # Nelder-Mead is very fast for smooth 5D functions.
        # It requires O(dim) evaluations per iteration.
        # Budget remaining: ~490.
        # We can afford many iterations.
        
        # Initialize Simplex
        # Start at x0.
        # Create vertices by moving along axes.
        # The size of the initial steps matters. 
        # Since the domain is [-5, 5], a step of 0.1-0.5 is reasonable.
        # Let's use 0.5 to cover a decent area quickly, but f11 is smooth so it might overshoot?
        # Actually, f11 is "Discus", which is flat near the minimum and steep far away.
        # A larger initial step helps locate the basin.
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        eps = 1e-8
        
        # Build initial simplex
        simplex = []
        fvals = []
        
        # Vertex 0: x0
        simplex.append(x0.copy())
        fvals.append(f0)
        
        # Vertices 1..dim: x0 + delta * e_i
        # Choose delta based on distance to boundary or fixed.
        # Let's try 0.5.
        delta = 0.5
        
        for i in range(dim):
            if time.time() > deadline: break
            xi = x0.copy()
            xi[i] += delta
            # Clip to box? The contract says func clips, but we should keep our internal state valid if possible.
            # Actually, if we clip, we might duplicate points. 
            # The contract says func(x) clips. So we can pass out-of-bounds points.
            # But Nelder-Mead assumes a continuous domain. 
            # Since the function is defined on the box and clipped, the effective domain is the box.
            # Let's just pass the point.
            fi = wrapped_func(xi)
            simplex.append(xi)
            fvals.append(fi)
            
        # If we ran out of budget or time during initialization
        if len(simplex) < dim + 1:
             return best_x, best_f

        # Main Loop
        # Stop if no improvement for many iterations or budget/time runs out.
        max_iter = 500 # Safety cap
        no_improve_count = 0
        max_no_improve = 50
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            
            # Sort simplex
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            best_f_curr = fvals[0]
            
            # Calculate centroid of all but the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst_x = simplex[-1]
            worst_f = fvals[-1]
            
            # Reflection
            xr = centroid + alpha * (centroid - worst_x)
            fr = wrapped_func(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst_x)
                fe = wrapped_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                no_improve_count = 0
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
                no_improve_count = 0
            else:
                # Contraction
                # Inside contraction
                xc = centroid + rho * (worst_x - centroid)
                fc = wrapped_func(xc)
                if fc < worst_f:
                    simplex[-1] = xc
                    fvals[-1] = fc
                    no_improve_count = 0
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = wrapped_func(simplex[i])
                    no_improve_count += 1
                    
                    # Check convergence
                    if no_improve_count > max_no_improve:
                        break
                        
                    # If simplex is very small, break
                    spread = np.max(np.array(simplex), axis=0) - np.min(np.array(simplex), axis=0)
                    if np.max(spread) < 1e-6:
                        break

        return best_x, best_f
