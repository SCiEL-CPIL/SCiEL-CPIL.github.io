import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We have 200 evaluations.
        # Strategy:
        # 1. Use Sobol sequence to generate diverse starting points.
        # 2. For each starting point, run a local optimizer (L-BFGS-B) with a scalarized objective.
        #    Since L-BFGS-B minimizes a single scalar, we use weighted sum method.
        #    To cover the Pareto front, we use different weight vectors.
        # 3. We need to be careful with the budget. L-BFGS-B can use many iterations.
        #    We will limit the number of function evaluations per local optimization.
        
        # Let's allocate budget:
        # - Reserve some for initial Sobol sampling if needed, but actually we can just use Sobol points as starts.
        # - Let's use 10 different weight vectors (directions) to cover the 3D objective space.
        # - For each weight vector, we pick a starting point from the Sobol sequence.
        # - We run L-BFGS-B. We need to count evaluations.
        #   L-BFGS-B doesn't easily expose a strict "stop after N evals" in all versions without callbacks, 
        #   but we can use a callback to count and raise an exception or stop.
        #   However, simpler: just run it and hope it's fast? No, we must respect budget strictly.
        
        # Alternative: Implement a simple gradient-free local search? No, L-BFGS is better for smooth problems.
        # Let's use `maxiter` and `ftol`/`gtol` to control convergence, but evaluation count is the hard constraint.
        # We can use a callback in scipy.optimize.minimize to count evaluations.
        
        eval_count = 0
        results = []
        
        # Generate Sobol points
        # We want maybe 20-30 starting points? 
        # If we run 20 local optimizations, and each uses ~5-10 evals on average, that's 100-200 evals.
        # Let's aim for 15 local optimizations.
        
        n_starts = 15
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        # Generate 2**k points. 2**5 = 32.
        sobol_points = sobol.random(n_starts)
        
        # Map to bounds
        X_starts = sobol_points * (hi - lo) + lo
        
        # Weight vectors for Weighted Sum Method
        # We want to cover the unit simplex in 3D.
        # Simple approach: random points on the simplex, or grid.
        # Let's use a small set of well-spread weight vectors.
        # E.g., (1,0,0), (0,1,0), (0,0,1), (1/3,1/3,1/3), and some intermediates.
        weights = []
        # Add corners
        weights.append([1.0, 0.0, 0.0])
        weights.append([0.0, 1.0, 0.0])
        weights.append([0.0, 0.0, 1.0])
        # Add center
        weights.append([1.0/3, 1.0/3, 1.0/3])
        # Add edges midpoints
        weights.append([0.5, 0.5, 0.0])
        weights.append([0.5, 0.0, 0.5])
        weights.append([0.0, 0.5, 0.5])
        # Add some random-ish well-spread points
        rng = np.random.default_rng(42)
        for _ in range(n_starts - len(weights)):
            w = rng.random(3)
            w = w / np.sum(w)
            weights.append(w)
        
        weights = np.array(weights)
        
        # Ensure we have enough weights for starts
        if len(weights) < n_starts:
            # Pad with random
            extra = rng.random((n_starts - len(weights), 3))
            extra = extra / extra.sum(axis=1, keepdims=True)
            weights = np.vstack([weights, extra])
        else:
            weights = weights[:n_starts]
            
        # Shuffle starts and weights to decorrelate specific bad starts with specific bad weights?
        # Actually, matching good starts with diverse weights is fine.
        
        for i in range(n_starts):
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
            
            x0 = X_starts[i]
            w = weights[i]
            
            # Define scalar objective
            def objective(x):
                nonlocal eval_count
                if time.time() > deadline or eval_count >= self.budget:
                    # Return a large value to stop optimization? 
                    # Better to raise an exception to break out of minimize?
                    # Scipy minimize handles exceptions poorly. 
                    # Let's just return a large number. The optimizer might continue though.
                    # To be safe, we should use a callback to raise an exception.
                    pass
                eval_count += 1
                f = problem(x)
                return np.dot(f, w)
                
            # Callback to stop if budget exceeded
            def callback(xk):
                nonlocal eval_count
                if eval_count >= self.budget or time.time() > deadline:
                    raise Exception("Budget Exceeded")
                return False
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), 
                               options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5},
                               callback=callback)
                if res.success or res.fun < 1e10: # Check if it actually ran
                    fx = problem(res.x)
                    results.append(res.x)
                    eval_count += 1 # The final evaluation in problem(res.x)
            except Exception:
                # If an exception was raised due to budget, we might have already recorded
                # some evaluations inside the objective function. 
                # We don't have the best x from the interrupted run easily.
                # We can try to use the last known good x? 
                # For now, we just skip this start if it fails.
                pass

        # If we have results, we are done.
        # The evaluator records ALL objective vectors returned by problem(x).
        # My code calls problem(x) inside the objective function and once after.
        # This is correct.
        
        # Return something to satisfy interface, though ignored.
        return np.array(results) if results else np.array([])
