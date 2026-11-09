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
        def nelder_mead(start_x, max_iter=60):
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

        # Phase 1: Initial sampling - diverse random points
        n_init = 30
        for _ in range(n_init):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            call_func(x)
        
        if calls_made >= budget or not time_ok():
            return best_x, best_y

        # Phase 2: Population-based adaptive search
        # Maintain a set of elite points (top performers)
        elites = [(best_y, best_x.copy())]
        last_improvement = 0
        
        while time_ok() and calls_made < budget:
            remaining = budget - calls_made
            if remaining < 15:
                break
            
            # Determine exploration vs exploitation based on progress
            progress_ratio = calls_made / budget
            # Early: more exploration; Late: more exploitation
            explore_prob = 0.3 * (1 - progress_ratio) + 0.1
            
            r = rng.random()
            
            if r < explore_prob and len(elites) > 1:
                # Explore: pick a random elite and perturb significantly
                elite_idx = rng.integers(0, len(elites))
                _, elite_x = elites[elite_idx]
                noise_scale = 1.5 * np.exp(-calls_made / 100.0)
                start_x = elite_x + rng.normal(0, noise_scale, d)
            elif r < explore_prob + 0.4:
                # Semi-exploit: perturb best with moderate noise
                noise_scale = 0.8 * np.exp(-calls_made / 150.0)
                start_x = best_x + rng.normal(0, noise_scale, d)
            else:
                # Exploit: perturb best with small noise
                noise_scale = 0.3 * np.exp(-calls_made / 200.0)
                start_x = best_x + rng.normal(0, noise_scale, d)
            
            start_x = np.clip(start_x, -5, 5)
            
            # Run Nelder-Mead from this starting point
            nelder_mead(start_x)
            
            # Update elites with the result
            if best_x is not None:
                if len(elites) == 0 or best_y < elites[0][0]:
                    elites.insert(0, (best_y, best_x.copy()))
                elif len(elites) < 5:
                    # Add to elites if it's better than the worst elite
                    if best_y < elites[-1][0]:
                        elites.append((best_y, best_x.copy()))
                        elites.sort(key=lambda x: x[0])
                        elites = elites[:5]
                else:
                    # Replace worst elite if this is better
                    if best_y < elites[-1][0]:
                        elites[-1] = (best_y, best_x.copy())
                        elites.sort(key=lambda x: x[0])
            
            # Check for stagnation: if no improvement in last 50 calls, add a random restart
            if calls_made - last_improvement > 50 and rng.random() < 0.3:
                # Force a random restart
                start_x = rng.uniform(-5, 5, d)
                start_x = np.clip(start_x, -5, 5)
                nelder_mead(start_x)
                if best_x is not None and best_y < elites[0][0]:
                    elites[0] = (best_y, best_x.copy())
                    last_improvement = calls_made
        
        # Final polish: very fine Nelder-Mead from best point
        if best_x is not None and time_ok() and calls_made < budget:
            start_x = best_x + rng.normal(0, 0.05, d)
            start_x = np.clip(start_x, -5, 5)
            nelder_mead(start_x, max_iter=40)
        
        return best_x, best_y
