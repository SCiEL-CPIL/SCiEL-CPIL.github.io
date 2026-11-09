import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 295.0  # Safety margin before 300s kill
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        # Cache to avoid redundant evaluations
        cache = {}
        cache_tol = 1e-12

        def check_time():
            return time.time() - start_time < time_limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if not check_time():
                return best_f
            
            # Check cache
            key = tuple(np.round(x, 12))
            if key in cache:
                return cache[key]
            
            x = np.clip(x, -5, 5)
            v = func(x)
            calls += 1
            cache[key] = v
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # 1. Initialization
        # Katsuura is multimodal but has a global minimum.
        # LHS provides good coverage.
        # We also add the origin and a few random points.
        init_points = []
        
        # Origin
        init_points.append(np.zeros(dim))
        
        # LHS
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            points = sampler.random(n=24)
            # Scale to [-4.5, 4.5] to stay well inside bounds
            points = points * 9.0 - 4.5
            init_points.extend(points)
        except ImportError:
            init_points.extend(rng.uniform(-4.5, 4.5, (24, dim)))
            
        # A few random points
        init_points.extend(rng.uniform(-4.5, 4.5, (6, dim)))

        init_values = []
        for p in init_points:
            v = wrapped_func(p)
            init_values.append(v)
            
        if best_x is None:
            # Should not happen, but safety
            return np.zeros(dim), 0.0

        sorted_idx = np.argsort(init_values)

        # 2. Multi-Start CMA-ES
        # CMA-ES is more robust than (1+1)-ES for multimodal functions.
        # We allocate remaining budget to the top few seeds.
        remaining_budget = self.budget - calls - 10
        if remaining_budget <= 0:
            return best_x, best_f

        # Number of restarts based on budget
        # Each CMA-ES run will use a portion of the budget.
        # Let's try 3 restarts with the best seeds.
        num_restarts = min(3, len(sorted_idx))
        
        # Allocate budget roughly equally, but give more to the best seed
        # e.g., 50%, 30%, 20%
        weights = [0.5, 0.3, 0.2][:num_restarts]
        
        for i, seed_idx in enumerate(sorted_idx[:num_restarts]):
            if not check_time() or calls >= self.budget - 5:
                break
                
            x0 = init_points[seed_idx].copy()
            f0 = init_values[seed_idx]
            
            # If this seed is worse than the current global best, skip?
            # No, CMA-ES can explore away from local minima.
            # But if it's significantly worse, maybe skip to save time?
            # Let's just run it.
            
            # CMA-ES Parameters
            # Lambda (population size)
            # For dim=5, lambda=4+floor(3*ln(5)) = 4+5 = 9?
            # Standard is 4 + floor(3 * ln(N))
            lam = 4 + int(3 * math.log(dim))
            if lam < 4:
                lam = 4
            if lam > 10:
                lam = 10 # Keep small for low dim to save budget
            
            # Initial sigma
            # Based on distance to best point? Or fixed?
            # Let's use a moderate sigma
            sigma = 1.0
            
            # Budget for this CMA-ES run
            budget_here = int(remaining_budget * weights[i])
            if budget_here < 10:
                budget_here = 10
            
            # Simple CMA-ES implementation
            # State variables
            mean = x0.copy()
            sigma = sigma
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = 1.0 / (lam + 3)
            c1 = (2 / ((dim + 1)**2 + 10))
            cmu = min(1 - c1, 2 * ((dim - 1. + 2. / (dim + 1)) / ((dim + 2.)**2 + 10)))
            damps = 1 + 2 * max(0, math.sqrt((lam - 1) / (dim + 1) - 1) - 1) + cs
            hs = (1 + 0.35) * math.sqrt((1 + 1. / (dim - 1)) * (1 - 1. / (dim + 1)) * (1 - 1. / (dim - 1)) * (1 - 1. / (dim + 1)) * (1 - 1. / (dim - 1)) * (1 - 1. / (dim + 1)) * (1 - 1. / (dim - 1)) * (1 - 1. / (dim + 1)))
            # Actually, hs formula is complex. Let's use a simpler adaptation or fixed sigma decay.
            # Given the small budget, a simple adaptive sigma might be safer than full CMA-ES state.
            # Let's stick to a robust (1+1)-ES with better step size control, or a simplified CMA-ES.
            
            # Let's try a simplified CMA-ES:
            # Just adapt sigma based on success rate, and keep C=I (diagonal).
            # This is essentially a diagonal CMA-ES, which is faster and less prone to bugs.
            
            # State
            mean = x0.copy()
            sigma = 1.0
            step_sizes = np.ones(dim) # Per-dimension step sizes
            
            # History for adaptation
            successes = 0
            total_evals = 0
            recent_successes = []
            
            # Main loop
            while calls < self.budget - 5 and check_time():
                # Generate population
                ys = []
                for j in range(lam):
                    # Sample from N(0, I)
                    z = rng.normal(0, 1, dim)
                    # Scale by sigma and step_sizes
                    y = mean + sigma * step_sizes * z
                    y = np.clip(y, -5, 5)
                    ys.append(y)
                
                # Evaluate
                fs = []
                for y in ys:
                    f_y = wrapped_func(y)
                    fs.append(f_y)
                    total_evals += 1
                    
                # Sort by fitness
                indices = np.argsort(fs)
                ys_sorted = [ys[i] for i in indices]
                fs_sorted = [fs[i] for i in indices]
                
                # Update mean using top mu points
                mu = max(1, lam // 2)
                weights_mu = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights_mu /= weights_mu.sum()
                
                new_mean = np.zeros(dim)
                for w, y in zip(weights_mu, ys_sorted[:mu]):
                    new_mean += w * y
                mean = new_mean
                
                # Adapt sigma
                # Count successes (improvements over previous best in this run? or just relative?)
                # Simple strategy: if best of this generation is better than mean of previous, increase sigma?
                # Or use standard CMA-ES adaptation.
                # Let's use a simple heuristic:
                # If the best point is significantly better than the worst, we might be converging.
                # If the spread is small, decrease sigma.
                
                # A robust simple adaptation:
                # Track the best fitness of the last few generations.
                # If improving, keep sigma or increase slightly.
                # If stagnating, decrease sigma.
                
                # Let's just use a fixed decay for simplicity and reliability, 
                # combined with a restart mechanism if no improvement.
                # Actually, the multi-start handles the restart.
                # Within a run, let's adapt sigma based on the ratio of best to worst.
                if fs_sorted[0] < fs_sorted[-1] - 1e-8:
                    ratio = (fs_sorted[-1] - fs_sorted[0]) / (abs(fs_sorted[0]) + 1e-8)
                    if ratio > 1.0:
                        sigma *= 1.05
                    else:
                        sigma *= 0.95
                else:
                    sigma *= 0.95
                    
                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 5.0)
                
                # Check for convergence
                if sigma < 1e-6:
                    break
                    
                # Early exit if no improvement in global best for a while?
                # The wrapped_func updates global best.
                # If local best (fs_sorted[0]) is much worse than global best, and we've made some calls, maybe stop?
                # Let's just run until budget or time.
            
        # 3. Final Polish
        if best_x is not None and calls < self.budget - 5 and check_time():
            # Run a tight local search from the best point found
            # Simple (1+1)-ES with very small steps
            x = best_x.copy()
            f_x = best_f
            sigma = 0.1
            no_improve = 0
            max_polish_steps = min(50, self.budget - calls - 5)
            
            for _ in range(max_polish_steps):
                if not check_time():
                    break
                mutation = rng.normal(0, 1, dim)
                y = x + sigma * mutation
                y = np.clip(y, -5, 5)
                f_y = wrapped_func(y)
                
                if f_y < f_x - 1e-12:
                    x = y
                    f_x = f_y
                    sigma *= 1.02
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if no_improve > 20:
                        break

        return best_x, best_f
