"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using scipy.optimize.minimize with weighted sum decomposition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate weight vectors for decomposition
        # Use a more dense grid for 3 objectives
        # Let's use a simplex-lattice with resolution 5
        resolution = 5
        weights = []
        for i in range(resolution + 1):
            for j in range(resolution + 1 - i):
                k = resolution - i - j
                weights.append(np.array([i, j, k]) / resolution)
        W = np.array(weights)
        n_subproblems = len(W)
        
        # We have budget=200 evaluations.
        # Strategy: 
        # 1. Initialize with a set of random points (e.g., 10 points) to get a baseline.
        # 2. For each weight vector, run a local optimization (Nelder-Mead or L-BFGS-B) 
        #    starting from the best known point for that weight or a random point.
        #    We need to manage the budget carefully.
        
        # Since scipy's minimize allows us to control function calls via options,
        # but we need to track global budget.
        # Let's create a wrapper that counts evaluations and stops if budget is exceeded.
        
        eval_count = 0
        all_points = []
        all_objectives = []
        
        # Initialize with some random points
        n_init = 10
        X_init = np.random.uniform(lo, hi, size=(n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        eval_count += n_init
        for x, f in zip(X_init, F_init):
            all_points.append(x)
            all_objectives.append(f)
        
        # Sort initial points by weighted sum for each weight vector to find good starting points
        # Or just pick the best point for each weight vector from the initial set?
        # Better: For each weight vector, pick the point from the initial set that minimizes the weighted sum.
        
        # We have 200 - 10 = 190 evaluations left.
        # Number of weight vectors is (resolution+1)(resolution+2)/2.
        # For resolution=5: 6*7/2 = 21 vectors.
        # 190 / 21 ≈ 9 evaluations per vector. That's too few for a local optimizer to converge well.
        
        # Alternative strategy:
        # Use a smaller set of weight vectors, but allow more evaluations per vector.
        # Or, use a hybrid:
        # 1. Run a few local optimizations with a budget of ~10-20 evaluations each.
        # 2. Use the remaining budget for global exploration (random or local around best points).
        
        # Let's try resolution=3: 4*5/2 = 10 vectors.
        # 190 / 10 = 19 evaluations per vector. Still a bit low for Nelder-Mead in 5D.
        
        # Let's try a different approach:
        # 1. Initialize with 5 random points.
        # 2. For each of the 10 weight vectors (resolution=3), start from the best initial point for that weight.
        # 3. Run Nelder-Mead with maxiter=10 (approx 10-20 function calls).
        # 4. Use remaining budget for local refinements around the best points found.
        
        # Actually, let's just use the full budget for a set of local optimizations.
        # We can define a custom objective function that counts calls and stops if budget is reached.
        
        class BudgetTracker:
            def __init__(self, budget):
                self.count = 0
                self.budget = budget
                self.exhausted = False
                self.last_x = None
                self.last_f = None
                
            def __call__(self, x):
                if self.exhausted:
                    # Return a large penalty to stop the optimizer
                    return 1e10
                self.count += 1
                self.last_x = x.copy()
                self.last_f = problem(x)
                if self.count >= self.budget:
                    self.exhausted = True
                return self.last_f
            
            def get_results(self):
                if self.last_x is None:
                    return None, None
                return self.last_x, self.last_f

        # Let's use a simpler loop-based approach to avoid complex state management with scipy.
        # We will manually implement a simple local search or use scipy with careful budgeting.
        
        # Let's stick to the plan:
        # 1. Init 10 points.
        # 2. For each of 10 weight vectors, run a local optimization.
        #    We will allocate ~18 evaluations per vector.
        #    Total: 10 + 10*18 = 190. Remaining 10 for final polishing.
        
        # Let's use Nelder-Mead for local optimization.
        # It doesn't have a strict "max function calls" option, but maxiter limits iterations.
        # Each iteration of Nelder-Mead in n dimensions can take up to n+1 function calls.
        # In 5D, that's 6 calls per iteration.
        # So maxiter=3 would use ~18 calls.
        
        # Let's try this.
        
        # Re-initialize budget tracker for the whole process
        global_eval_count = n_init
        global_deadline = deadline
        
        # For each weight vector
        for w in W:
            if global_eval_count >= self.budget or time.time() > global_deadline:
                break
            
            # Find the best starting point from all_points for this weight
            best_start_idx = 0
            best_score = np.dot(all_objectives[0], w)
            for i in range(1, len(all_points)):
                score = np.dot(all_objectives[i], w)
                if score < best_score:
                    best_score = score
                    best_start_idx = i
            
            x0 = all_points[best_start_idx].copy()
            
            # Define the objective function for this weight vector
            def weighted_obj(x):
                nonlocal global_eval_count
                if global_eval_count >= self.budget or time.time() > global_deadline:
                    return 1e10
                global_eval_count += 1
                f = problem(x)
                all_points.append(x.copy())
                all_objectives.append(f)
                return np.dot(f, w)
            
            # Run Nelder-Mead
            # maxiter=3, initial simplex size small
            options = {'maxiter': 3, 'xatol': 1e-4, 'fatol': 1e-4}
            try:
                res = minimize(weighted_obj, x0, method='Nelder-Mead', bounds=list(zip(lo, hi)), options=options)
                if res.x is not None and res.fun < 1e9:
                    # res.x is already added to all_points in the callback? No, minimize doesn't use a callback for recording in this simple wrapper.
                    # The last evaluation is already in all_points.
                    pass
            except Exception:
                pass
        
        # Use remaining budget for random sampling or local refinement around best points
        while global_eval_count < self.budget and time.time() < global_deadline:
            # Pick a random point from the best 20% of points and perturb it
            n_best = max(1, len(all_points) // 5)
            # Sort by hypervolume contribution? Too expensive. Just sort by norm.
            # Or just pick a random point from the existing set and perturb.
            idx = np.random.randint(len(all_points))
            x = all_points[idx].copy()
            sigma = 0.1
            x = x + sigma * (hi - lo) * np.random.randn(self.dim)
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            global_eval_count += 1
            all_points.append(x)
            all_objectives.append(f)
            
            # Check if this new point dominates any existing point? 
            # If so, we could remove the dominated point to keep the set small, 
            # but we are just recording all points for the evaluator.
            # The evaluator will compute hypervolume from all recorded points.
            # So we just need to generate good points.
            
        return np.array(all_points)
# EVOLVE-BLOCK-END
