import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Time limit safety margin
        time_limit = 325.0
        
        # State
        self.best_x = np.zeros(n)
        self.best_f = np.inf
        self.calls_made = 0
        
        def call_func(x):
            if self.calls_made >= budget:
                return np.inf
            fx = func(x)
            self.calls_made += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Coarse Global Search (Grid-like + Random)
        # The problem has a hidden shift and rotation. A grid helps find the basin.
        # We divide the box [-5, 5]^5 into a coarse grid.
        # 5 dimensions, box size 10.
        # Let's try a 3x3x3x3x3 grid = 243 points.
        # Plus some random points to break symmetry if grid misses.
        
        grid_points = []
        # 3 points per dimension: -4, 0, 4 (covering most of the box, avoiding edges where optimum is unlikely if it's in [-4,4])
        # Actually, optimum is in [-4, 4]. Let's use -4, 0, 4.
        coords = [-4.0, 0.0, 4.0]
        
        # Generate grid
        import itertools
        for combo in itertools.product(coords, repeat=n):
            grid_points.append(np.array(combo))
            
        # Add some random points to ensure coverage
        for _ in range(50):
            grid_points.append(rng.uniform(-4.5, 4.5, size=n))
            
        # Shuffle to mix grid and random, but maybe grid first is better for systematic coverage?
        # Let's shuffle to avoid bias towards corner if shift is central.
        # Actually, systematic is often better for "finding" the region.
        # But we have limited budget. 243 + 50 = 293 calls.
        # Leaves ~200 calls for refinement.
        
        # Let's evaluate the grid points.
        # To maximize AOCC, we want to find a decent value early.
        # Random shuffling might find a good point sooner on average if the landscape is flat-ish?
        # Or systematic might hit the center (0,0,0,0,0) early?
        # Center (0,0,0,0,0) is index 136 in 3^5 grid (middle).
        # Let's prioritize the center and then expand outwards?
        # Or just random order? Random is safer against specific shift patterns.
        
        rng.shuffle(grid_points)
        
        # Evaluate grid
        for x in grid_points:
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                break
            call_func(x)
            
        # Phase 2: Local Refinement (Basin Hopping / Nelder-Mead style)
        # Use the best point found so far as the starting point.
        # We will perform a series of local optimizations using Nelder-Mead or a simple gradient-free local search.
        # Since we don't have gradients, Nelder-Mead is a good choice for small dimensions.
        # However, implementing Nelder-Mead from scratch is verbose. 
        # Alternative: A simple (1+1) evolution strategy with adaptive step size, or a random walk around the best point.
        # Given the "Sharp ridge" nature, a local search that can handle sharpness is needed.
        # CMA-ES is good but we used it in Phase 1? No, Phase 1 was grid.
        # Let's use a simplified local search: Random perturbations with decreasing step size.
        
        current_x = self.best_x.copy()
        current_f = self.best_f
        step_size = 0.5 # Initial local step size
        
        # Number of local steps depends on remaining budget
        remaining_budget = budget - self.calls_made
        
        # We'll do a loop of local improvements
        # If we find a better point, keep it. If not, reduce step size or restart from best.
        
        # Strategy: 
        # 1. Try Nelder-Mead if scipy is available and fast enough? 
        #    scipy.optimize.minimize with method='Nelder-Mead' is robust.
        #    But it might use many function calls. We can limit maxiter.
        
        # Let's try a hybrid: 
        # If remaining budget > 50, use a few iterations of a simple local search.
        
        # Actually, let's use a more structured local search: 
        # Coordinate descent? No, rotation makes axes non-aligned.
        # Random walk with cooling?
        
        # Let's try a simple Adaptive Random Walk:
        # 1. Generate random direction.
        # 2. Step forward.
        # 3. If improved, keep, maybe increase step.
        # 4. If not, try backward? Or just reject and decrease step.
        
        # To be robust against the "Sharp ridge", we need to be able to move along the ridge.
        # A random walk might struggle if the ridge is thin.
        # However, the grid should have put us *on* or *near* the ridge.
        
        # Let's implement a simple Nelder-Mead manually to avoid scipy overhead/complexity?
        # Or just use scipy if it's fast.
        # scipy.optimize.minimize(func, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
        # This uses function calls. We need to wrap func to count calls and check time.
        
        def wrapped_func(x):
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                return np.inf
            return call_func(x)

        # Use scipy Nelder-Mead for local refinement
        try:
            from scipy.optimize import minimize
            
            # Run Nelder-Mead from the best point found
            # Limit maxiter to ensure we don't run out of time/budget
            # 500 calls total. Grid used ~300. Remaining ~200.
            # Nelder-Mead in 5D might use 5*100 = 500 calls per run if not careful.
            # We should limit maxiter to ~30-40 to stay within budget.
            
            res = minimize(wrapped_func, current_x, method='Nelder-Mead', 
                           options={'maxiter': 30, 'xatol': 1e-3, 'fatol': 1e-3})
            
            # Update best if scipy found something better (it updates self.best_f via wrapped_func)
            
            # If we still have budget, do another round from the new best?
            if self.calls_made < budget and time.time() - self.start_time < time_limit:
                current_x = self.best_x.copy()
                # Maybe a second run with smaller tolerance?
                res2 = minimize(wrapped_func, current_x, method='Nelder-Mead', 
                                options={'maxiter': 20, 'xatol': 1e-5, 'fatol': 1e-5})
                
        except Exception:
            # Fallback to simple random local search if scipy fails
            step_size = 0.5
            while self.calls_made < budget and time.time() - self.start_time < time_limit:
                direction = rng.standard_normal(n)
                direction /= np.linalg.norm(direction)
                y = current_x + step_size * direction
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < current_f:
                    current_x = y
                    current_f = fy
                    step_size *= 1.1 # Expand slightly
                else:
                    step_size *= 0.8 # Shrink
                    if step_size < 1e-5:
                        break

        return self.best_x, self.best_f
