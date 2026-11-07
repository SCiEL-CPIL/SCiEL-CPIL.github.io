import numpy as np
import time

class Optimizer:
    """
    Optimizer using a harmonic analysis / lattice-based approach for Rastrigin.
    Exploits the periodic structure of f3 to estimate the global minimum location.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Time guard
        start_time = time.time()
        time_limit = 280.0  # 50s safety margin before 330s kill
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        
        calls_made = 0
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_fx
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Lattice scan on integer grid
        # Rastrigin has local minima at integer coordinates.
        # The global minimum is at some integer point (after shift/rotation, but the structure is preserved).
        # We scan a coarse integer grid to find promising regions.
        
        # For dim=5, scanning all integers in [-4, 4]^5 is 9^5 = 59049 points.
        # With 500 budget, we can't scan all. Let's be smarter.
        
        # Strategy: 
        # 1. First, sample a sparse set of integer points to get a rough idea.
        # 2. Then refine around the best found.
        
        # Actually, let's think about the Rastrigin structure more carefully.
        # f(x) = 10*dim + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The periodic part has period 2. The quadratic part pushes towards 0.
        # Local minima are at integer points.
        
        # Let's do a multi-scale approach:
        # 1. Coarse scan: Sample integer points in a reduced grid
        # 2. Fine scan: Refine around the best coarse point
        
        # Phase 1: Coarse integer grid scan
        # Sample every 2nd integer to keep it manageable
        # Grid: [-4, -2, 0, 2, 4] in each dim -> 5^5 = 3125 points. Still too many.
        
        # Let's use a smarter approach: random sampling of integer points first.
        
        # Phase 1: Random sampling of integer points
        n_samples = 100
        for i in range(n_samples):
            if time.time() - start_time > time_limit:
                break
            if calls_made >= self.budget:
                break
            
            # Random integer point in [-4, 4]^dim
            x = rng.integers(-4, 5, self.dim).astype(float)
            evaluate(x)
        
        # Phase 2: Local refinement around best point
        # Use a (1+1)-ES with small step size to refine
        
        if calls_made < self.budget and time.time() - start_time < time_limit:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1  # Small step for local refinement
            
            no_improve = 0
            stagnation_limit = 20
            
            while calls_made < self.budget:
                if time.time() - start_time > time_limit:
                    break
                
                step = rng.standard_normal(self.dim)
                y = x + sigma * step
                y = np.clip(y, -5.0, 5.0)
                
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    no_improve = 0
                    sigma *= 1.2
                else:
                    no_improve += 1
                    sigma *= 0.9
                    
                    if no_improve >= stagnation_limit:
                        # Try a few random restarts in the neighborhood
                        for _ in range(5):
                            if calls_made >= self.budget or time.time() - start_time > time_limit:
                                break
                            x_new = best_x + rng.normal(0, 0.5, self.dim)
                            x_new = np.clip(x_new, -5.0, 5.0)
                            evaluate(x_new)
                        break
        
        return best_x, best_fx
