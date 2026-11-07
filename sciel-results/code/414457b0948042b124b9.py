"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead simplex method for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        def time_left():
            return deadline - time.time()
        
        def nelder_mead(func, x0, f0, budget_left, deadline, rng):
            """
            Run Nelder-Mead from x0 with f0.
            Returns (best_x, best_f, calls_used).
            """
            dim = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            fs = [f0]
            
            # Create other vertices by perturbing one coordinate at a time
            for i in range(dim):
                xi = x0.copy()
                step = 0.5 if xi[i] >= 0 else -0.5
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                simplex.append(xi)
                fs.append(fi)
            
            calls_used = dim + 1
            
            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            # Sort initial simplex
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = [fs[i] for i in order]
            
            while calls_used < budget_left and time_left() > 0.1:
                # Check for convergence
                if np.std(fs) < 1e-8:
                    break
                
                # Sort simplex
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1
                
                if fr < fs[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_used += 1
                    
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
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_used += 1
                            fs[i] = fi
                        
                        if calls_used >= budget_left or time_left() < 0.1:
                            break
                
                # Update global best
                for i in range(dim + 1):
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = simplex[i].copy()
                
                # Early exit if very close to optimum? We don't know f_opt, so no.
                # But we can check if simplex is tiny.
                if np.max(np.abs(np.array(fs) - fs[0])) < 1e-10:
                    break
            
            return best_x, best_f, calls_used

        # Phase 1: Global Search (Random Sampling)
        # Spend ~20% of budget on random sampling to find diverse basins.
        n_random = max(10, budget // 5)
        
        candidates = []
        for i in range(n_random):
            if calls_made >= budget or time_left() < 1.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            candidates.append((fx, x.copy()))
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-start Nelder-Mead
        # Allocate remaining budget among top candidates.
        # Let's use the top 5 unique candidates.
        used_points = [best_x]
        selected_starts = []
        for fx, x in candidates:
            if len(selected_starts) >= 5:
                break
            # Ensure diversity: distance from existing starts > 1.0
            is_diverse = True
            for existing in used_points:
                if np.linalg.norm(x - existing) < 1.0:
                    is_diverse = False
                    break
            if is_diverse:
                selected_starts.append((fx, x))
                used_points.append(x)
        
        if not selected_starts:
            selected_starts = [(best_f, best_x)]
        
        remaining_budget = budget - calls_made
        n_starts = len(selected_starts)
        
        # Allocate budget to each start, giving more to better starts
        # Simple heuristic: equal allocation, but cap per start to allow restarts if stuck
        # Actually, just run them sequentially with a portion of the budget.
        # Let's give each start a chunk of the remaining budget.
        # But we should stop a start if it stagnates.
        
        for i, (fx_start, x_start) in enumerate(selected_starts):
            if calls_made >= budget or time_left() < 1.0:
                break
            
            # Allocate budget: give each start at least 20 calls, but not more than remaining
            # Let's just pass the remaining budget to nelder_mead, and it will stop when it runs out.
            # However, if one start consumes all budget, others won't run.
            # Let's reserve some budget for later starts?
            # For simplicity, let's just run them one by one, each with the *current* remaining budget.
            # This means the first start gets the most budget. To be fair, we could split.
            # Let's split the remaining budget evenly among the remaining starts.
            budget_per_start = (budget - calls_made) // (n_starts - i)
            if budget_per_start < 10:
                budget_per_start = (budget - calls_made)
            
            # Run Nelder-Mead
            # Note: nelder_mead updates global best_f and best_x via closure? 
            # No, it returns them. Let's pass best_f/best_x as references or just return and update.
            # To avoid confusion, let's have nelder_mead return the local best, and we update global.
            
            # Actually, the nelder_mead implementation above uses closure variables best_f/best_x.
            # This is dangerous if we want to run multiple instances.
            # Let's refactor nelder_mead to be self-contained and return local best.
            
            pass # Placeholder, will rewrite nelder_mead below
            
        # Re-implementing nelder_mead to be self-contained
        
        def nelder_mead_isolated(func, x0, f0, max_calls, deadline, dim, rng):
            """
            Run Nelder-Mead from x0 with f0.
            Returns (best_x, best_f, calls_used).
            """
            simplex = [x0.copy()]
            fs = [f0]
            
            for i in range(dim):
                xi = x0.copy()
                step = 0.5 if xi[i] >= 0 else -0.5
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                simplex.append(xi)
                fs.append(fi)
            
            calls_used = dim + 1
            local_best_x = x0.copy()
            local_best_f = f0
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = [fs[i] for i in order]
            
            if fs[0] < local_best_f:
                local_best_f = fs[0]
                local_best_x = simplex[0].copy()
                
            while calls_used < max_calls and time_left() > 0.1:
                if np.std(fs) < 1e-8:
                    break
                
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1
                
                if fr < local_best_f:
                    local_best_f = fr
                    local_best_x = xr.copy()
                
                if fr < fs[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_used += 1
                    
                    if fe < local_best_f:
                        local_best_f = fe
                        local_best_x = xe.copy()
                        
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    if fr < fs[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    
                    if fc < local_best_f:
                        local_best_f = fc
                        local_best_x = xc.copy()
                        
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_used += 1
                            fs[i] = fi
                            
                            if fi < local_best_f:
                                local_best_f = fi
                                local_best_x = simplex[i].copy()
                            
                            if calls_used >= max_calls or time_left() < 0.1:
                                break
                        
                        if calls_used >= max_calls or time_left() < 0.1:
                            break
                
            return local_best_x, local_best_f, calls_used

        # Now run the multi-start loop properly
        for i, (fx_start, x_start) in enumerate(selected_starts):
            if calls_made >= budget or time_left() < 1.0:
                break
            
            # Allocate budget for this start
            # Let's give it a portion of the remaining budget.
            # If it's the last start, give it all remaining.
            if i == n_starts - 1:
                alloc = budget - calls_made
            else:
                # Reserve some for future starts?
                # Let's just give it half the remaining budget, or a fixed chunk.
                # A fixed chunk might be better to ensure all starts get a chance.
                # Let's try allocating 100 calls per start, or remaining if less.
                alloc = min(100, budget - calls_made)
                if alloc < 10:
                    break
                    
            # Run Nelder-Mead
            x_best, f_best, used = nelder_mead_isolated(func, x_start, fx_start, alloc, deadline, dim, rng)
            calls_made += used
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
                
            # If we made significant progress, we might want to restart from this new best?
            # Or just continue to next start.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
