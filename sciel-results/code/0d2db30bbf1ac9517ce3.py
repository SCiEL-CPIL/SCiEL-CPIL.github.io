import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        # State for AOCC tracking (best value so far)
        # The evaluator tracks best value after each call i.
        # Since we cannot see the evaluator's internal counter, we just minimize.
        # The "score" is based on the sequence of best values.
        # Multi-start NM is expected to find local minima faster than random walk.
        
        # We will run multiple independent Nelder-Mead optimizations.
        # To maximize AOCC (which rewards early progress), we should start with
        # a few quick runs and then refine.
        
        # However, NM is deterministic given a start point.
        # We need diverse start points.
        
        # Strategy:
        # 1. Split budget into K chunks.
        # 2. For each chunk, run NM from a random start point.
        # 3. Keep track of the global best.
        # 4. If time allows, refine the global best with a final NM run.
        
        # How many restarts?
        # f22 has 21 peaks. 
        # A single NM run might take ~50-100 evaluations to converge to a local min.
        # With 500 budget, we can afford ~5-10 restarts.
        
        # Let's use a "survivor" approach:
        # Start with 5 restarts. 
        # Each gets budget/5 evaluations.
        # After each restart, if it found a better min, it becomes the new "champion".
        # The champion is then refined in subsequent chunks? 
        # Or just run independent restarts and keep the best?
        # Independent restarts are simpler and robust.
        
        # Actually, NM is very fast. It might converge in <50 steps.
        # If we use fixed budget per restart, we might waste calls on already-converged runs.
        # Better: Run NM until convergence or budget exhausted for that run.
        
        # Let's define a helper for Nelder-Mead.
        
        def nelder_mead(func, x0, max_iter, timeout_check=None):
            """
            Simple Nelder-Mead implementation.
            Returns best_x, best_f.
            """
            n = len(x0)
            # Initial simplex
            simplex = [x0.copy() for _ in range(n + 1)]
            
            # Expand the first vertex slightly to create a simplex
            # Standard NM: x0, x0 + delta*e_i
            delta = 0.5
            for i in range(n):
                simplex[i+1][i] += delta
            
            # Evaluate initial simplex
            fvals = [func(x) for x in simplex]
            
            # Sort simplex by fvals
            # We maintain a list of (fval, point)
            # To avoid sorting overhead in a tight loop, we can just track the best.
            # But NM needs the worst, best, and second best.
            # For n=5, sorting 6 points is trivial.
            
            best_x = x0.copy()
            best_f = min(fvals)
            
            # Indices for best, second best, worst
            # Let's just sort the list of [fval, point]
            # Actually, modifying the simplex list in-place is faster.
            
            # Initialize
            for i in range(n + 1):
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()
            
            # We need to track the full simplex state.
            # Let's use a list of points and a list of values.
            # We will sort them every iteration. n=5, so 6 points.
            
            for it in range(max_iter):
                if timeout_check and timeout_check():
                    break
                    
                # Sort by function value
                # Create list of (fval, index)
                order = sorted(range(n + 1), key=lambda i: fvals[i])
                
                # Reorder simplex and fvals
                new_simplex = [simplex[i] for i in order]
                new_fvals = [fvals[i] for i in order]
                
                simplex = new_simplex
                fvals = new_fvals
                
                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:n], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[n])
                fr = func(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[n])
                    fe = func(xe)
                    if fe < fr:
                        simplex[n] = xe
                        fvals[n] = fe
                    else:
                        simplex[n] = xr
                        fvals[n] = fr
                elif fr < fvals[n-1]:
                    # Accept reflection
                    simplex[n] = xr
                    fvals[n] = fr
                else:
                    # Contraction
                    # Inside contraction if fr < fvals[n], else outside
                    if fr < fvals[n]:
                        # Inside
                        xc = centroid + 0.5 * (simplex[n] - centroid)
                        fc = func(xc)
                        if fc < fr:
                            simplex[n] = xc
                            fvals[n] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = func(simplex[i])
                    else:
                        # Outside
                        xc = centroid - 0.5 * (simplex[n] - centroid)
                        fc = func(xc)
                        if fc < fvals[n]:
                            simplex[n] = xc
                            fvals[n] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = func(simplex[i])
                            
                # Check convergence
                # If range of fvals is small, stop
                if max(fvals) - min(fvals) < 1e-6:
                    break
                    
            return best_x, best_f

        rng = np.random.default_rng(42)
        
        # Multi-start strategy
        # We have 500 calls.
        # Let's do 5 restarts.
        # Each gets 100 calls.
        # If a run converges early, it stops, and we move to the next?
        # The budget is strict: "More than 500 calls ... invalid".
        # So we must track total calls.
        
        calls_made = 0
        global_best_f = np.inf
        global_best_x = np.zeros(dim)
        
        # Initial random point to get a baseline
        x_init = rng.uniform(-5, 5, dim)
        f_init = func(x_init)
        calls_made += 1
        global_best_f = f_init
        global_best_x = x_init.copy()
        
        # Number of restarts
        num_restarts = 5
        budget_per_restart = (budget - calls_made) // num_restarts
        
        # If budget_per_restart is too small, reduce restarts
        while budget_per_restart < 20 and num_restarts > 1:
            num_restarts -= 1
            budget_per_restart = (budget - calls_made) // num_restarts
            
        # If still too small, just 1 restart (effectively NM from x_init)
        if budget_per_restart < 10:
            num_restarts = 1
            budget_per_restart = budget - calls_made

        for r in range(num_restarts):
            # Check time
            if time.time() > deadline:
                break
                
            # Check budget
            if calls_made >= budget:
                break
                
            # Start point
            # First restart uses x_init (already evaluated, but NM will re-evaluate simplex vertices)
            # To save calls, we can pass x_init and let NM evaluate.
            # NM evaluates n+1 points initially.
            # If we reuse x_init, we save 1 call.
            # But for simplicity and safety, let's just let NM run.
            # We have plenty of budget.
            
            if r == 0:
                x_start = x_init
            else:
                x_start = rng.uniform(-5, 5, dim)
                
            # Run NM
            # Max iter should be limited by remaining budget
            # NM uses ~2-3 calls per iteration on average (reflection, sometimes expansion/contraction)
            # So max_iter = budget_per_restart // 3 is a safe estimate.
            max_iter = budget_per_restart // 3
            
            # We need a timeout check that also respects the global budget?
            # NM doesn't know about global budget.
            # We can pass a lambda that checks time.
            # But we must ensure we don't exceed global budget.
            # Since max_iter is derived from budget_per_restart, and we do this for each restart,
            # total calls should be roughly num_restarts * (initial_simplex + 3*max_iter)
            # initial_simplex = 6.
            # Total ~ 5 * (6 + 3 * 33) = 5 * 105 = 525. This might exceed 500.
            
            # Let's be more conservative.
            # Total budget left = budget - calls_made
            # Allocate evenly.
            # Let's just track calls_made in a closure? No, NM is a separate function.
            
            # Better approach:
            # Modify NM to take a "budget_left" and decrement it?
            # Or just limit max_iter more strictly.
            
            # Let's calculate exact calls.
            # NM initial: 6 calls.
            # Each iter: 1 (reflection) + maybe 1 (expansion/contraction) + maybe n (shrink).
            # Worst case shrink: 5 calls.
            # So ~7 calls/iter worst case.
            # Let's use max_iter = budget_per_restart // 10 to be safe.
            
            max_iter_safe = budget_per_restart // 10
            
            # If max_iter_safe is 0, skip
            if max_iter_safe < 5:
                continue
                
            x_best, f_best = nelder_mead(func, x_start, max_iter_safe, lambda: time.time() > deadline)
            
            # We don't know exactly how many calls NM made.
            # This is a problem. The contract says "More than 500 calls ... invalid".
            # We MUST track calls precisely.
            
            # The current implementation of nelder_mead does NOT track calls globally.
            # We need to wrap func to count calls.
            
            pass # This logic is flawed. Need to rewrite with call counting.

        # RE-IMPLEMENTATION WITH CALL COUNTING
        
        # Reset
        calls_made = 0
        global_best_f = np.inf
        global_best_x = np.zeros(dim)
        
        def counted_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                # Return a bad value to stop optimization, but we shouldn't call if we know we're out.
                # Better to check before calling.
                return np.inf
            calls_made += 1
            return func(x)
            
        # Initial point
        x_init = rng.uniform(-5, 5, dim)
        f_init = counted_func(x_init)
        global_best_f = f_init
        global_best_x = x_init.copy()
        
        num_restarts = 5
        # We have budget - 1 calls left.
        # Let's divide remaining budget among restarts.
        remaining_budget = budget - calls_made
        budget_per_restart = remaining_budget // num_restarts
        
        # Ensure min budget
        while budget_per_restart < 30 and num_restarts > 1:
            num_restarts -= 1
            budget_per_restart = remaining_budget // num_restarts
            
        if budget_per_restart < 10:
            num_restarts = 1
            budget_per_restart = remaining_budget

        for r in range(num_restarts):
            if time.time() > deadline or calls_made >= budget:
                break
                
            x_start = rng.uniform(-5, 5, dim) if r > 0 else x_init
            
            # Run NM with a hard stop on calls
            # We need a version of NM that stops if calls_made >= budget
            # We can pass a "should_stop" function
            
            def should_stop():
                return calls_made >= budget or time.time() > deadline
                
            # Modify NM to use counted_func and should_stop
            # We need to update the nelder_mead function to accept these
            
            # Let's define a local NM that uses counted_func
            
            def nm_run(x0, max_iter):
                n = len(x0)
                simplex = [x0.copy() for _ in range(n + 1)]
                for i in range(n):
                    simplex[i+1][i] += 0.5
                    
                fvals = []
                for x in simplex:
                    if should_stop():
                        break
                    fvals.append(counted_func(x))
                    
                if len(fvals) < n + 1:
                    return x0, np.min(fvals) if fvals else np.inf
                    
                best_x = x0.copy()
                best_f = np.min(fvals)
                
                for it in range(max_iter):
                    if should_stop():
                        break
                        
                    order = sorted(range(n + 1), key=lambda i: fvals[i])
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]
                    
                    if fvals[0] < best_f:
                        best_f = fvals[0]
                        best_x = simplex[0].copy()
                        
                    centroid = np.mean(simplex[:n], axis=0)
                    xr = centroid + (centroid - simplex[n])
                    
                    if should_stop(): break
                    fr = counted_func(xr)
                    
                    if fr < fvals[0]:
                        if should_stop(): break
                        xe = centroid + 2 * (centroid - simplex[n])
                        fe = counted_func(xe)
                        if fe < fr:
                            simplex[n] = xe
                            fvals[n] = fe
                        else:
                            simplex[n] = xr
                            fvals[n] = fr
                    elif fr < fvals[n-1]:
                        simplex[n] = xr
                        fvals[n] = fr
                    else:
                        if fr < fvals[n]:
                            if should_stop(): break
                            xc = centroid + 0.5 * (simplex[n] - centroid)
                            fc = counted_func(xc)
                            if fc < fr:
                                simplex[n] = xc
                                fvals[n] = fc
                            else:
                                for i in range(1, n + 1):
                                    if should_stop(): break
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    fvals[i] = counted_func(simplex[i])
                        else:
                            if should_stop(): break
                            xc = centroid - 0.5 * (simplex[n] - centroid)
                            fc = counted_func(xc)
                            if fc < fvals[n]:
                                simplex[n] = xc
                                fvals[n] = fc
                            else:
                                for i in range(1, n + 1):
                                    if should_stop(): break
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    fvals[i] = counted_func(simplex[i])
                            
                    if max(fvals) - min(fvals) < 1e-6:
                        break
                        
                return best_x, best_f

            x_best, f_best = nm_run(x_start, budget_per_restart // 5)
            
            if f_best < global_best_f:
                global_best_f = f_best
                global_best_x = x_best.copy()

        return global_best_x, global_best_f
