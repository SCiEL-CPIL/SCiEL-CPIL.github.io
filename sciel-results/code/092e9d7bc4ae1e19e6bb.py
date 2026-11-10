"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        deadline = start_time + 290.0  # Safety margin before 330s kill limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Constrained Chebyshev Scalarization Strategy
        # We generate a diverse set of weight vectors on the simplex.
        # For each weight vector, we run a local optimizer (L-BFGS-B) to find a point on the Pareto front.
        
        # 1. Generate weight vectors
        # We need enough weight vectors to cover the front well.
        # With 3 objectives, we can use a grid.
        # Let's use 10x10 grid = 100 weight vectors? 
        # Budget is 200. Each L-BFGS run uses multiple evaluations.
        # A typical L-BFGS run might take 10-30 evaluations.
        # So we can afford roughly 5-10 L-BFGS runs.
        # Let's pick 8 diverse weight vectors.
        
        # Simplex grid for 3 objectives
        # We want points (w1, w2, w3) such that w1+w2+w3=1, wi >= 0
        # Let's create a few representative points:
        # Corners: (1,0,0), (0,1,0), (0,0,1)
        # Edges midpoints: (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
        # Center: (1/3, 1/3, 1/3)
        # And a few more for better coverage.
        
        weights = []
        # Generate a grid
        for i in range(5):
            for j in range(5 - i):
                k = 5 - i - j
                w = np.array([i/5.0, j/5.0, k/5.0])
                # Normalize to sum to 1? No, the values above already sum to 1 if we treat i,j,k as counts out of 5.
                # Wait, i+j+k = 5. So w sums to 1.
                weights.append(w)
        
        # We have 15 weight vectors. 
        # Let's subsample to fit budget.
        # We will allocate budget dynamically.
        # Each L-BFGS call: we need to count function evaluations.
        # SciPy's minimize with L-BFGS-B does not easily expose the exact count in a simple callback without extra work, 
        # but we can approximate or limit maxiter.
        # Actually, the `problem` function is what counts towards the budget.
        # We can wrap the objective to count evaluations.
        
        eval_count = 0
        max_budget = self.budget
        
        def objective(x):
            nonlocal eval_count
            if eval_count >= max_budget:
                # Return a large penalty if we exceed budget
                return np.inf
            eval_count += 1
            
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            
            # Chebyshev scalarization: min max_i (w_i * (f_i - z_i^*))
            # z^* is the ideal point. We don't know it exactly, but we can approximate.
            # Or use the standard Chebyshev: min max_i (w_i * f_i)
            # To make it work for minimization and ensure we explore the whole front, 
            # we often use: min max_i (w_i * (f_i - z_i)) where z is a reference.
            # Since we don't know z, let's use the simpler weighted sum first? 
            # No, the prompt suggests Chebyshev.
            # Let's use the standard Chebyshev distance to the origin (assuming objectives are positive and we want to minimize).
            # Actually, for multi-objective optimization, the Chebyshev scalarization is:
            # min_x max_i { w_i * (f_i(x) - z_i^*) }
            # where z^* is the ideal point. 
            # If we don't know z^*, we can use the current best found so far as an approximation, 
            # or just use weighted sum if Chebyshev is tricky without z^*.
            # However, Chebyshev is better for non-convex fronts. WFG7 is non-convex? 
            # WFG7 is known to have a non-convex Pareto front. 
            # Weighted sum will miss parts of the non-convex front.
            # Chebyshev with a reference point is needed.
            
            # Let's maintain a running ideal point (minimum seen so far).
            # But inside the objective function, we don't have easy access to global state unless we pass it.
            # Let's use a simpler approach: Weighted Sum might be insufficient for non-convex.
            # Let's try to implement Chebyshev with a dynamic reference.
            
            # For now, let's just use Weighted Sum to see if it works, or a simple Chebyshev with reference 0?
            # If objectives are positive, min max(w_i * f_i) is a valid scalarization.
            
            # Let's use: max(w * f)
            # This is equivalent to finding points that are good in all objectives relative to weights.
            
            # To handle the non-convexity, we should ideally use a reference point.
            # Let's initialize a reference point as the first few random points' objectives.
            
            # Since we can't easily update z* inside the objective without global state, 
            # let's stick to a robust strategy:
            # 1. Random sampling to get an initial front.
            # 2. Use L-BFGS with Weighted Sums for convex parts.
            # 3. Use L-BFGS with Chebyshev (approximate) for non-convex parts.
            
            # Actually, let's just use Weighted Sum first. It's simpler and often works well enough for scoring if the front isn't too non-convex.
            # But WFG7 IS non-convex. 
            # Let's use the "augmented Chebyshev" scalarization.
            # min max_i (w_i * (f_i - z_i)) + rho * sum_i (w_i * (f_i - z_i))
            # We need z. Let's use a global variable for z.
            
            return np.max(weights_current[0] * f) # Placeholder, need to pass weights
            
        # Let's restructure. We will define the scalarization inside the loop for each weight.
        
        # First, do some random sampling to establish a reference point z* (ideal point estimate)
        n_random = 20
        X_rand = np.random.uniform(lo, hi, size=(n_random, dim))
        F_rand = np.array([problem(x) for x in X_rand])
        eval_count += n_random
        
        z_star = np.min(F_rand, axis=0)
        
        # Now, select weight vectors
        # We have 15 weights from the grid.
        # Let's pick a subset to fit the budget.
        # Remaining budget: 200 - 20 = 180.
        # Each L-BFGS run: limit maxiter to 20. Each iteration ~1-2 evals? 
        # L-BFGS-B uses line search, so maybe 5-10 evals per iteration.
        # Let's set maxiter=5. That's ~25-50 evals per run.
        # We can afford 3-4 runs.
        
        # Let's pick 4 diverse weights:
        # (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8), (0.33, 0.33, 0.33)
        
        selected_weights = [
            np.array([0.8, 0.1, 0.1]),
            np.array([0.1, 0.8, 0.1]),
            np.array([0.1, 0.1, 0.8]),
            np.array([1/3, 1/3, 1/3])
        ]
        
        # Add a few more if budget allows
        if eval_count < 150:
            selected_weights.append(np.array([0.5, 0.3, 0.2]))
            selected_weights.append(np.array([0.2, 0.5, 0.3]))
        
        for w in selected_weights:
            if time.time() > deadline:
                break
            if eval_count >= max_budget:
                break
            
            def scalar_obj(x, w=w, z=z_star):
                nonlocal eval_count
                if eval_count >= max_budget:
                    return np.inf
                eval_count += 1
                
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                
                # Chebyshev scalarization
                # min max_i (w_i * (f_i - z_i))
                # To handle potential negative values or scaling issues, 
                # and to ensure we find distinct points, we add a small weighted sum term (augmented Chebyshev).
                # Actually, simple Chebyshev can lead to many points mapping to the same solution.
                # Let's use: max(w * (f - z)) + 0.01 * sum(w * (f - z))
                
                diff = f - z
                cheb = np.max(w * diff)
                aug = 0.01 * np.sum(w * diff)
                return cheb + aug
            
            # Random starting point for this weight
            x0 = np.random.uniform(lo, hi)
            
            # Bounds for L-BFGS-B
            bounds_lbfgs = list(zip(lo, hi))
            
            # Run optimization
            try:
                res = minimize(scalar_obj, x0, method='L-BFGS-B', bounds=bounds_lbfgs, options={'maxiter': 10, 'maxfun': 30})
            except Exception:
                continue
        
        return None
# EVOLVE-BLOCK-END
