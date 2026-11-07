"""Optimizer for BBOB-style f17 (Schaffers F7) using Multi-Restart Nelder-Mead with Adaptive Initialization."""
import numpy as np
import time


class Optimizer:
    """Multi-restart Nelder-Mead with adaptive initialization for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: kill at 285s, limit is 300s (evaluator) / 330s (hard kill)
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        count = 0
        
        def evaluate(x):
            nonlocal count, best_x, best_f
            try:
                f = func(x)
                count += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return float('inf')
        
        # Helper to check if we should stop
        def should_stop():
            return count >= budget or time.time() > deadline
        
        # Phase 1: Initial exploration with Halton-like points
        # We want to spread out the initial evaluations to find the basin.
        # For dim=5, we can use a small set of deterministic points.
        
        # Generate a set of initial points
        # Center
        init_points = [np.zeros(dim)]
        
        # Points along axes
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0  # 3.0 is 60% of the half-range
                init_points.append(p)
        
        # Some random points
        # Use a fixed seed for reproducibility within the instance if needed, 
        # but here we just want diversity.
        rng = np.random.default_rng(42)
        for _ in range(10):
            p = rng.uniform(-4, 4, size=dim)
            init_points.append(p)
            
        # Evaluate initial points
        # We have 1 + 2*5 + 10 = 21 points.
        # Budget is 500. So we have plenty of room.
        
        # Sort initial points by fitness to identify the best basin
        init_results = []
        for p in init_points:
            if should_stop():
                break
            f = evaluate(p)
            init_results.append((f, p))
        
        if not init_results:
            return best_x, best_f
            
        # Sort by fitness
        init_results.sort(key=lambda x: x[0])
        
        # Phase 2: Nelder-Mead refinement from the best few initial points
        # We will run Nelder-Mead from the top 3-5 initial points, 
        # allocating remaining budget to them.
        
        # Let's pick the best initial point as the primary seed.
        # And maybe 2-3 others as secondary seeds if budget allows.
        
        # Nelder-Mead implementation
        def nelder_mead(func, x0, budget_remaining, max_iter=100):
            """
            Robust Nelder-Mead implementation.
            """
            n = len(x0)
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            eps = 1e-8
            
            # Initial simplex
            # Start with x0, then add points with small perturbations
            simplex = [x0.copy()]
            f0 = evaluate(x0)
            current_budget = budget_remaining
            
            # Create initial simplex
            for i in range(n):
                x_new = x0.copy()
                # Perturb along axis i
                # Initial step size: 0.1 * range or similar
                # Box is [-5, 5], so range is 10. 0.1 * 10 = 1.0
                # But if x0 is near boundary, clip might be needed.
                # Let's use a relative step size.
                step = 0.5
                x_new[i] += step
                x_new = np.clip(x_new, -5, 5)
                simplex.append(x_new)
                f = evaluate(x_new)
                current_budget -= 1
                if current_budget <= 0:
                    break
            
            if len(simplex) < n + 1:
                return best_x, best_f
                
            # Store functions
            simplex_f = []
            for i, x in enumerate(simplex):
                if i == 0:
                    simplex_f.append(f0)
                else:
                    # We already evaluated these, but let's re-evaluate for consistency 
                    # or just keep track. Since we can't easily retrieve previous evals 
                    # without storing them, let's just re-evaluate or assume we stored them.
                    # Actually, we can't retrieve them from `evaluate` easily.
                    # Let's modify the strategy: Nelder-Mead needs to evaluate points.
                    # We should pass a function that we can control.
                    # But `evaluate` increments count.
                    # Let's just run the loop and let `evaluate` handle it.
                    pass
            
            # Re-evaluate to get consistent state in simplex_f
            # This wastes some budget but ensures correctness.
            # Better: store the evaluations we just did.
            # Let's restart the Nelder-Mead logic to properly track simplex.
            
            # Revised Nelder-Mead
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            # We already created n points. Let's re-do this cleanly.
            # Actually, the points created above are in `simplex` but `simplex_f` is incomplete.
            # Let's just re-evaluate the simplex points to be safe, or better, 
            # structure the code to avoid this.
            
            # Let's write a cleaner Nelder-Mead that takes a starting simplex.
            # But we don't have a starting simplex easily.
            # Let's just use the first point x0 and generate the simplex inside, 
            # accepting the cost of n evaluations.
            
            # Since we already did n evaluations in the "Initial simplex" loop above,
            # we have wasted them if we don't use them.
            # Let's just restart the NM logic with x0 and let it generate the simplex,
            # wasting the previous n evaluations.
            # To avoid waste, we can pass the already evaluated points.
            
            # Let's implement a version that accepts initial points and their values.
            # But we didn't store the values from the loop above properly.
            
            # OK, let's just implement a standard NM that generates its own simplex,
            # and accept the small overhead.
            
            # Reset simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            # Generate initial simplex
            for i in range(n):
                x_new = x0.copy()
                step = 0.5
                x_new[i] += step
                x_new = np.clip(x_new, -5, 5)
                simplex.append(x_new)
                f = evaluate(x_new)
                simplex_f.append(f)
                if count >= budget or time.time() > deadline:
                    return best_x, best_f
            
            # Main NM loop
            for _ in range(max_iter):
                if count >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by fitness
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Calculate centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)
                if count >= budget or time.time() > deadline:
                    break
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if count >= budget or time.time() > deadline:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction if f_r < f_w, inside otherwise
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if count >= budget or time.time() > deadline:
                        break
                    
                    if f_c <= min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_s = evaluate(simplex[i])
                            simplex_f[i] = f_s
                            if count >= budget or time.time() > deadline:
                                return best_x, best_f
            
            return best_x, best_f

        # We have 3 good initial points.
        # Allocate remaining budget to them.
        # Let's pick the best 3.
        seeds = [p for f, p in init_results[:3]]
        
        remaining_budget = budget - count
        # Allocate budget to each seed.
        # Simple strategy: give 1/3 of remaining to each, or run sequentially.
        # Running sequentially allows us to stop early if one converges.
        
        # Let's run NM on the best seed first.
        # If it doesn't converge well, we might not have time for others.
        # But NM is fast.
        
        # Run NM on best seed
        if not should_stop():
            nelder_mead(func, seeds[0], remaining_budget)
            
        # Run NM on second best seed if time/budget allows
        if not should_stop():
            # Use a smaller budget for secondary seeds
            nelder_mead(func, seeds[1], min(remaining_budget // 2, 100))
            
        # Run NM on third best seed
        if not should_stop():
            nelder_mead(func, seeds[2], min(remaining_budget // 3, 50))

        return best_x, best_f
