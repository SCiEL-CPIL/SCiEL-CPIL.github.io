"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective Optimizer using Weighted Sum Scalarization with L-BFGS-B."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        eval_count = 0
        all_points = []
        all_objs = []

        # Define scalar objective for a given weight vector
        def scalar_obj(x, w):
            nonlocal eval_count
            if eval_count >= self.budget:
                return 1e6
            try:
                f = np.array(problem(x))
                eval_count += 1
                all_points.append(x.copy())
                all_objs.append(f)
                return np.dot(w, f)
            except Exception:
                eval_count += 1
                return 1e6

        # Generate weight vectors on a simplex grid
        # For 3 objectives, we need weights w1, w2, w3 >= 0, sum(w) = 1
        # We use a grid with 'levels' divisions on each axis
        levels = 10 
        weight_vectors = []
        for w1 in np.linspace(0, 1, levels):
            for w2 in np.linspace(0, 1 - w1, levels):
                w3 = 1 - w1 - w2
                if w3 < -1e-9:
                    continue
                if w3 < 0:
                    w3 = 0
                weight_vectors.append([w1, w2, w3])
        
        # Sort weight vectors to ensure systematic coverage? 
        # Or just iterate. 
        # To maximize diversity, we might want to prioritize extreme points first?
        # L-BFGS-B is a local method. Starting from diverse points is key.
        
        # Initial random starts for each weight vector
        # We have budget 200. 
        # Number of weight vectors is ~ (levels*(levels+1)/2) = 55 for levels=10.
        # 200 / 55 ~ 3.6 evaluations per weight vector.
        # This is very tight for L-BFGS-B which might take 10-20 iterations.
        # We need a strategy that is robust to limited evaluations.
        
        # Alternative: 
        # 1. Generate a set of diverse initial points (e.g. 20 random points).
        # 2. For each initial point, run a short L-BFGS-B with a few different weights?
        #    Or, assign weights to initial points?
        
        # Better Strategy:
        # 1. Create a list of candidate weight vectors (dense grid).
        # 2. Create a list of starting points (random + corners + center).
        # 3. Pair them up or iterate through weights, starting from a random point.
        #    To ensure we don't waste budget on long runs, we limit maxiter.
        
        # Let's refine the weight vector count.
        # With 200 budget, if each L-BFGS-B takes ~5-10 evals, we can afford ~20-40 runs.
        # Let's pick a subset of weight vectors.
        
        # Select a subset of weight vectors that covers the simplex well.
        # We can use a stratified sampling or just a coarser grid.
        # Let's try levels=6 -> 21 vectors.
        # 200 / 21 ~ 9.5 evals per run. This is feasible for L-BFGS-B with maxiter=5-10.
        
        levels = 6
        weight_vectors = []
        for w1 in np.linspace(0, 1, levels):
            for w2 in np.linspace(0, 1 - w1, levels):
                w3 = 1 - w1 - w2
                if w3 < -1e-9:
                    continue
                if w3 < 0:
                    w3 = 0
                weight_vectors.append([w1, w2, w3])
                
        # Shuffle weight vectors? No, systematic is better for coverage if starts are random.
        # But if we use the same random start for all, we might get stuck in local minima.
        # Let's generate a pool of starting points.
        
        n_starts = len(weight_vectors)
        # Generate random starts
        starts = []
        for _ in range(n_starts):
            if time.time() > deadline or eval_count >= self.budget:
                break
            # Random point in bounds
            x0 = lo + (hi - lo) * np.random.rand(dim)
            starts.append(x0)
            
        # Also add some deterministic points if we have room?
        # Corners and center are good for multi-modal problems.
        # But random is usually better for unknown problems.
        
        # Run optimization for each weight vector
        for i, w in enumerate(weight_vectors):
            if time.time() > deadline or eval_count >= self.budget:
                break
            if i >= len(starts):
                break
                
            x0 = starts[i]
            
            # L-BFGS-B parameters
            # maxiter should be small to fit in budget
            # 200 budget, ~21 vectors. ~9 evals per vector.
            # L-BFGS-B uses 1 eval per iteration (plus line search).
            # Let's set maxiter=5 to be safe.
            res = minimize(
                scalar_obj, 
                x0, 
                args=(w,), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 5, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            # The scalar_obj function already records the points and objectives.
            # We don't need to do anything with res here, as the recording is side-effect.
            
        return None
# EVOLVE-BLOCK-END
