"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import cma

class Optimizer:
    """Multi-Start CMA-ES based optimizer for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        x_global_best = None
        f_global_best = np.inf
        calls = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def evaluate(x):
            nonlocal calls, f_global_best, x_global_best
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < f_global_best:
                f_global_best = f
                x_global_best = x.copy()
            return f

        # Phase 1: Global Exploration
        # We sample a diverse set of points to identify multiple promising basins.
        # f22 is highly multimodal, so a single start point is risky.
        n_init = 50
        candidates = []
        
        for _ in range(n_init):
            if not check_time() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            candidates.append((f, x))

        if not candidates:
            return x_global_best, f_global_best

        # Sort candidates by fitness and select top K distinct points for multi-start
        # We want starts that are sufficiently far apart to cover different basins.
        candidates.sort(key=lambda t: t[0])
        
        # Select top 5 points, ensuring they are not too close to each other
        starts = []
        min_dist = 1.0  # Minimum Euclidean distance between start points
        for f, x in candidates:
            if len(starts) >= 5:
                break
            is_distinct = True
            for _, sx in starts:
                if np.linalg.norm(x - sx) < min_dist:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append((f, x))
        
        # If we didn't find 5 distinct points, just take the top ones
        if len(starts) < 5:
            for f, x in candidates:
                if (f, x) not in starts:
                    starts.append((f, x))
                if len(starts) >= 5:
                    break

        # Phase 2: Parallel Multi-Start CMA-ES
        # Split remaining budget among the starts
        remaining_budget = budget - calls
        n_starts = len(starts)
        if n_starts == 0:
            return x_global_best, f_global_best
            
        budget_per_start = max(10, remaining_budget // n_starts)
        
        # Standard CMA-ES population size for dim=5 is around 10-15.
        # Let's use a slightly smaller population to allow more generations per start
        # or more starts. 
        pop_size = 4 + int(3 * np.log(dim))
        if pop_size < 10:
            pop_size = 10
            
        # Initial sigma: scale to the box size
        sigma0 = 1.5
        
        for i, (f_start, x_start) in enumerate(starts):
            if not check_time() or calls >= budget:
                break
                
            # Budget for this specific CMA-ES run
            # We subtract the calls already made in this loop iteration context?
            # No, 'calls' is global. We just limit the CMA-ES run to not exceed total budget.
            
            # Create a wrapper for CMA-ES that respects global budget and time
            def wrapped_func(x, _calls_ref=[calls]):
                nonlocal calls
                if not check_time():
                    raise cma.StopIteration("Time limit reached")
                if calls >= budget:
                    raise cma.StopIteration("Budget exceeded")
                return evaluate(x)

            # We need to be careful: cma.fmin uses its own internal logic.
            # We pass x_start as the initial mean.
            # We set a maxiter high enough, but rely on StopIteration to stop it.
            
            opts = {
                'maxiter': 1000, 
                'seed': 42 + i,
                'verbose': -9, 
                'tolx': 1e-8,
                'tolfun': 1e-10,
                'bounds': [-5, 5],
                'popsize': pop_size,
                'verbose_eval': -9
            }
            
            try:
                # cma.fmin returns (best_x, result, es)
                # It will call wrapped_func.
                best_x, result, es = cma.fmin(wrapped_func, x_start, sigma0, opts)
                
                # Explicitly evaluate the best found to ensure it's in our global tracking
                # (though wrapped_func already does this)
                if best_x is not None:
                    f_best = evaluate(best_x)
                
            except (cma.StopIteration, Exception):
                # StopIteration is expected when budget/time is hit
                # Other exceptions might occur if cma fails internally
                pass

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
