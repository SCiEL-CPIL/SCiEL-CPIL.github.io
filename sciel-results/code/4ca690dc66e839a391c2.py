import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        eval_count = 0
        max_evals = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate weight vectors
        # We want to cover the simplex of weights for 3 objectives
        # Simple strategy: Random Dirichlet samples + Vertices
        n_weights = 10
        weights = []
        # Add vertices
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
        # Add random points
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in weights generation? 
        # The task says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)". 
        # This affects np.random, not necessarily np.random.default_rng. 
        # I should use np.random to respect the seed if I want reproducibility across seeds, 
        # but usually, the seed is for the problem's noise or initial state. 
        # The prompt says "np.random.seed(seed)" is called. 
        # If I use np.random, my weights will depend on the seed. 
        # If I use default_rng, they won't. 
        # It's probably better to use np.random to align with the evaluator's seed setting, 
        # although for a deterministic algorithm, fixed weights are fine. 
        # Let's use fixed weights to ensure consistent behavior.
        
        # Let's just use a fixed set of well-distributed weights.
        # For 3 objectives, the simplex is a triangle.
        # We can sample a grid or use a specific set.
        # 10 weights: 3 vertices, 3 edge midpoints, 3 random, 1 center.
        
        fixed_weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1/3, 1/3, 1/3],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
        
        for w in fixed_weights:
            if eval_count >= max_evals:
                break
            if time.time() > deadline:
                break
                
            w = np.array(w)
            
            def scalar_obj(x, w=w):
                nonlocal eval_count
                if eval_count >= max_evals:
                    raise TimeoutError
                if time.time() > deadline:
                    raise TimeoutError
                eval_count += 1
                f = np.array(problem(x))
                return np.dot(w, f)
            
            # Initial point: random within bounds
            x0 = lo + (hi - lo) * np.random.rand(self.dim)
            
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 4, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
                # res.x is the best point found
                # We don't need to record res explicitly, as problem calls are recorded
            except (TimeoutError, Exception):
                # Stop if we hit limits
                break

        return None
