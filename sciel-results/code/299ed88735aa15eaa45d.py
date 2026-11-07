"""
Optimizer using Multi-Start (1+1)-ES with Adaptive Step Size and Nelder-Mead Polish
for BBOB f20 (Schwefel).
Structural change: Replaces the heavy multi-restart Nelder-Mead core with a fast
(1+1)-Evolution Strategy for broad exploration and rapid early progress, followed
by a precise Nelder-Mead polish for final convergence. This addresses the
AOCC scoring metric which heavily weights early improvements.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Global state
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def try_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Generate diverse starting points
        # 1. Origin
        # 2. Structured points (corners, mid-points)
        # 3. Random points
        starts = []
        starts.append(np.zeros(n))
        starts.append(np.ones(n) * 3.0)
        starts.append(np.ones(n) * -3.0)
        # Add a few random starts to ensure coverage
        rng_starts = np.random.default_rng(42)
        for _ in range(5):
            starts.append(rng_starts.uniform(-4.5, 4.5, n))
        
        # Allocate budget:
        # Reserve ~50 calls for final Nelder-Mead polish
        # Remaining for (1+1)-ES exploration
        polish_budget = 50
        es_budget = budget - polish_budget
        
        # We will run (1+1)-ES on each start point, but limit the number of steps
        # per start to ensure we visit multiple starts.
        # With 7 starts and ~450 ES budget, that's ~64 calls per start.
        
        for i, x0 in enumerate(starts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # Check if we should stop exploring and start polishing
            # If we have enough budget left for polish, and we've done some work,
            # we might want to stop early if we are close to convergence?
            # For now, just run ES until budget for this start is exhausted or global budget hit.
            
            # Simple heuristic: allocate remaining ES budget / remaining starts
            remaining_starts = len(starts) - i
            if calls_made + polish_budget >= budget:
                # Not enough budget left for a full polish + this start's minimal work?
                # Just break and go to polish if we have a best_x
                if best_x is not None:
                    break
                else:
                    # Must make at least one call
                    pass
            
            # (1+1)-ES parameters
            sigma = 1.5
            sigma_min = 1e-6
            sigma_max = 4.0
            successes = 0
            steps = 0
            no_improve = 0
            
            # Initial evaluation
            fx = try_func(x0)
            if fx is None:
                break
            
            x = x0.copy()
            
            # Limit steps per start to avoid getting stuck in one basin
            # Heuristic: 50-80 steps is usually enough for (1+1)-ES to find a local min
            max_steps_per_start = 60
            
            for _ in range(max_steps_per_start):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Generate candidate
                y = np.clip(x + sigma * np.random.standard_normal(n), -5, 5)
                fy = try_func(y)
                if fy is None:
                    break
                
                steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    no_improve = 0
                    # Increase sigma on success
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.9, sigma_min)
                
                # Adaptive sigma based on success rate (1/5 rule approximation)
                if steps >= 10:
                    success_rate = successes / steps
                    if success_rate > 0.3:
                        sigma = min(sigma * 1.1, sigma_max)
                    elif success_rate < 0.1:
                        sigma = max(sigma * 0.9, sigma_min)
                    # Reset counters periodically
                    if steps >= 50:
                        successes = 0
                        steps = 0
                
                # If stuck, try a larger jump
                if no_improve > 20:
                    sigma = min(sigma * 2.0, sigma_max)
                    no_improve = 0

        # Final Polish: Nelder-Mead
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Use a simple Nelder-Mead implementation to avoid scipy overhead/dependency issues
            # or use scipy if available. Given the constraint, let's use a lightweight internal NM.
            
            # Initialize simplex around best_x
            x_curr = best_x.copy()
            f_curr = best_f
            
            # Step size for simplex
            step = 0.5
            simplex = np.zeros((n + 1, n))
            simplex[0] = x_curr
            for i in range(n):
                simplex[i + 1] = x_curr
                simplex[i + 1, i] += step
            
            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            init_ok = True
            for i in range(n + 1):
                if calls_made >= budget or time.time() > deadline:
                    init_ok = False
                    break
                fx = func(simplex[i])
                fvals[i] = fx
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()
            
            if not init_ok:
                # If we couldn't even evaluate the simplex, just return best found so far
                pass
            else:
                # NM Parameters
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_shrink = 0.5
                
                max_iter = 50  # Limit iterations to save time
                
                for it in range(max_iter):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Sort
                    order = np.argsort(fvals)
                    simplex = simplex[order]
                    fvals = fvals[order]
                    
                    # Check convergence
                    spread = np.max(np.abs(simplex - simplex[0]), axis=0)
                    if np.all(spread < 1e-5):
                        break
                    
                    # Centroid
                    centroid = np.mean(simplex[:n], axis=0)
                    
                    # Reflection
                    x_ref = centroid + alpha * (centroid - simplex[n])
                    x_ref = np.clip(x_ref, -5, 5)
                    f_ref = func(x_ref)
                    calls_made += 1
                    
                    if f_ref < fvals[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = func(x_exp)
                        calls_made += 1
                        
                        if f_exp < f_ref:
                            simplex[n] = x_exp
                            fvals[n] = f_exp
                        else:
                            simplex[n] = x_ref
                            fvals[n] = f_ref
                    elif f_ref < fvals[n-1]:
                        simplex[n] = x_ref
                        fvals[n] = f_ref
                    else:
                        # Contraction
                        x_con = centroid + rho * (simplex[n] - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        calls_made += 1
                        
                        if f_con < fvals[n]:
                            simplex[n] = x_con
                            fvals[n] = f_con
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = func(simplex[i])
                                calls_made += 0 # Note: func calls are counted in try_func? No, here we call func directly.
                                # Wait, I used `func` directly here. I need to increment calls_made.
                                calls_made += 1
                                
                                if fvals[i] < best_f:
                                    best_f = fvals[i]
                                    best_x = simplex[i].copy()
                            
                            # Check budget inside shrink loop
                            if calls_made >= budget or time.time() > deadline:
                                break
                    
                    # Update global best
                    if fvals[0] < best_f:
                        best_f = fvals[0]
                        best_x = simplex[0].copy()

        # Fallback: ensure at least one call was made
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
