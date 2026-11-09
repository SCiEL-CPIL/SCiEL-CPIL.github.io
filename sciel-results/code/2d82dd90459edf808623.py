import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_y = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_y
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            y = func(x)
            calls_made += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Nelder-Mead implementation
        def nelder_mead(start_x, max_iter=80):
            nonlocal best_x, best_y, calls_made
            
            if best_x is None:
                return
            
            # Initialize simplex
            simplex = [start_x.copy()]
            for i in range(d):
                v = start_x.copy()
                if abs(v[i]) > 1e-10:
                    v[i] *= 1.05
                else:
                    v[i] += 0.5
                simplex.append(v)
            
            # Evaluate initial simplex
            fvals = []
            for i in range(d + 1):
                f = call_func(simplex[i])
                if f is None:
                    return
                fvals.append(f)
            
            # Sort simplex
            for _ in range(d + 1):
                idx = np.argmin(fvals)
                simplex[idx], simplex[0] = simplex[0], simplex[idx]
                fvals[idx], fvals[0] = fvals[0], fvals[idx]
            
            # Nelder-Mead iterations
            for it in range(max_iter):
                if not time_ok() or calls_made >= budget:
                    break
                
                # Update best
                if fvals[0] < best_y:
                    best_y = fvals[0]
                    best_x = simplex[0].copy()
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fr = call_func(xr)
                if fr is None:
                    return
                
                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fe = call_func(xe)
                    if fe is None:
                        return
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
                    rho = 0.5
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = call_func(xc)
                        if fc is None:
                            return
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                f = call_func(simplex[i])
                                if f is None:
                                    return
                                fvals[i] = f
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - worst)
                        xc = np.clip(xc, -5, 5)
                        fc = call_func(xc)
                        if fc is None:
                            return
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                f = call_func(simplex[i])
                                if f is None:
                                    return
                                fvals[i] = f
                
                # Sort simplex
                for i in range(d):
                    for j in range(d - i):
                        if fvals[j] > fvals[j + 1]:
                            fvals[j], fvals[j + 1] = fvals[j + 1], fvals[j]
                            simplex[j], simplex[j + 1] = simplex[j + 1], simplex[j]
                
                # Check convergence
                if d > 0 and np.max(fvals) - np.min(fvals) < 1e-10:
                    break
        
        # Phase 1: Sobol-like quasi-random initial sampling
        # Use Halton sequence for low-discrepancy sampling
        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    f = 1.0
                    i_val = i + 1
                    base = j + 2
                    while i_val > 0:
                        f /= base
                        i_val = (i_val - 1) // base
                    points[i, j] = f
            return points
        
        n_init = 40
        # Generate Halton points and map to [-5, 5]
        halton_pts = halton_sequence(n_init, d)
        halton_pts = -5 + 10 * halton_pts
        
        # Interleave with some random points
        for i in range(n_init):
            if not time_ok() or calls_made >= budget:
                break
            if i % 2 == 0:
                x = halton_pts[i // 2]
            else:
                x = rng.uniform(-5, 5, d)
            call_func(x)
        
        if calls_made >= budget or not time_ok():
            return best_x, best_y
        
        # Phase 2: Main optimization loop with Nelder-Mead
        # Maintain a list of promising starting points
        candidates = []
        if best_x is not None:
            candidates.append(best_x.copy())
        
        # Add some diverse points to the candidate list
        for _ in range(5):
            x = rng.uniform(-5, 5, d)
            f = call_func(x)
            if f is not None and best_x is not None:
                candidates.append(x)
        
        # Sort candidates by function value (we need to evaluate them)
        # Actually, we already have best_x, so let's just use it as primary
        # and generate perturbations around it
        
        while time_ok() and calls_made < budget:
            remaining = budget - calls_made
            if remaining < 20:
                break
            
            # Choose starting point
            r = rng.random()
            if r < 0.5 and best_x is not None:
                # Exploit: perturb current best with decreasing noise
                noise_scale = 1.0 * np.exp(-calls_made / 150.0)
                start_x = best_x + rng.normal(0, noise_scale, d)
            elif r < 0.8:
                # Semi-explore: random point near best
                start_x = best_x + rng.uniform(-2, 2, d) if best_x is not None else rng.uniform(-5, 5, d)
            else:
                # Explore: random point
                start_x = rng.uniform(-5, 5, d)
            
            start_x = np.clip(start_x, -5, 5)
            
            # Run Nelder-Mead
            nelder_mead(start_x)
        
        # Final polish: very fine Nelder-Mead from best point
        if best_x is not None and time_ok() and calls_made < budget:
            # Create a small perturbation around best
            start_x = best_x + rng.normal(0, 0.05, d)
            start_x = np.clip(start_x, -5, 5)
            nelder_mead(start_x, max_iter=50)
        
        return best_x, best_y
