"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using Weighted Sum Scalarization + Local Refinement.
    
    Strategy:
    1. Generate a set of weight vectors that cover the unit simplex (for 3 objectives).
    2. For each weight vector, perform a local search (Nelder-Mead) to find a point
       that minimizes the weighted sum of objectives.
    3. Collect all resulting points and return them.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Generate weight vectors for 3 objectives
        # We use a systematic grid on the simplex x+y+z=1, x,y,z >= 0
        # To get good coverage, we can use a number of levels.
        # Let's try a grid of size 4x4 (10 points) or 5x5 (15 points).
        # With budget 200, we can afford ~15-20 initial starts, each with local refinement.
        
        # Let's use a set of weight vectors.
        # Simple approach: generate points on the simplex.
        weights = []
        n_levels = 4 # 4 levels means indices 0..3, sum=3? No, let's use a standard method.
        # Generate all (i, j, k) such that i+j+k = S, i,j,k >= 0
        S = 4
        for i in range(S + 1):
            for j in range(S - i + 1):
                k = S - i - j
                if k >= 0:
                    w = np.array([i, j, k], dtype=float)
                    if np.sum(w) > 0:
                        w = w / np.sum(w)
                        weights.append(w)
        
        # Convert to numpy array
        W = np.array(weights)
        n_weights = len(W)
        
        # We have budget 200.
        # Each local optimization might take 10-50 evaluations.
        # Let's limit the number of weights we process to ensure we finish in time.
        # Or, we can just run local optimization for each weight vector.
        
        # Initialize a list to store all evaluated points
        all_X = []
        all_F = []
        
        # We need to be careful with the budget.
        # Let's allocate budget per weight vector.
        # If we have ~15 weight vectors, we can spend ~10-15 evals each on local search.
        # Nelder-Mead in 4D usually converges in < 50 evals.
        
        # Let's just run local search for each weight vector.
        # We will use a fixed number of max iterations for minimize to control cost.
        
        for w in W:
            if time.time() > deadline:
                break
            
            # Define the scalar objective
            def scalar_obj(x):
                # Clip x to bounds for safety, though minimize should handle bounds
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                return np.dot(f, w)
            
            # Initial guess: center of the box
            x0 = (lo + hi) / 2.0
            
            # Use Nelder-Mead with bounds
            # Note: scipy's minimize with method='Nelder-Mead' doesn't support bounds directly in older versions,
            # but we can use 'L-BFGS-B' or 'Powell' which do.
            # 'Powell' is derivative-free and supports bounds, often robust for small problems.
            # 'Nelder-Mead' is simple but doesn't respect bounds strictly without clipping in the objective.
            # Since problem() clips, it's safe to use Nelder-Mead and let it explore outside,
            # but the objective will be constant outside bounds, which might confuse it.
            # Better to use a method that respects bounds or clip inside the objective (done above).
            
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                
                # Record the best point found by this local search
                best_x = np.clip(res.x, lo, hi)
                best_f = problem(best_x)
                
                all_X.append(best_x)
                all_F.append(best_f)
                
            except Exception:
                # If optimization fails, record the initial point
                best_x = x0
                best_f = problem(best_x)
                all_X.append(best_x)
                all_F.append(best_f)

        # If we have time, we can also add some random points to ensure coverage
        # and potentially find non-dominated points missed by the weighted sums
        # (weighted sums can miss non-convex fronts, but RE37 is likely convex enough).
        
        # Let's check how many evaluations we used.
        # We don't track exactly, but we can estimate.
        # To be safe, let's add a few random points if we are under budget.
        # We don't have an explicit counter, but we can assume each minimize call
        # uses roughly 20-30 evals. 15 weights * 30 = 450 evals? That might exceed 200.
        
        # Wait, the budget is 200 evaluations.
        # My loop above might exceed the budget!
        # I need to track evaluations.
        
        # Let's restart with a stricter budget control.
        
        # Re-implementing with explicit evaluation counting.
        
        all_X = []
        all_F = []
        eval_count = 0
        
        # We have 200 evals.
        # Let's use 10 weight vectors.
        # Each local search gets ~15 evals.
        # 10 * 15 = 150 evals.
        # Remaining 50 evals for random sampling.
        
        # Select 10 well-spread weight vectors from W
        if len(W) > 10:
            # Pick 10 evenly spaced points
            indices = np.linspace(0, len(W)-1, 10, dtype=int)
            W_selected = W[indices]
        else:
            W_selected = W
            
        for w in W_selected:
            if time.time() > deadline:
                break
            if eval_count >= 180: # Leave 20 for random
                break
                
            def scalar_obj(x):
                nonlocal eval_count
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                eval_count += 1
                return np.dot(f, w)
            
            x0 = (lo + hi) / 2.0
            
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 20, 'xatol': 1e-3, 'fatol': 1e-3})
                
                best_x = np.clip(res.x, lo, hi)
                # Evaluate final point to get F (this adds 1 eval)
                best_f = problem(best_x)
                eval_count += 1
                
                all_X.append(best_x)
                all_F.append(best_f)
                
            except Exception:
                best_x = x0
                best_f = problem(best_x)
                eval_count += 1
                all_X.append(best_x)
                all_F.append(best_f)

        # Add random points with remaining budget
        while eval_count < self.budget and time.time() < deadline:
            x = lo + (hi - lo) * np.random.random(self.dim)
            f = problem(x)
            eval_count += 1
            all_X.append(x)
            all_F.append(f)

        # Convert to numpy arrays
        if all_X:
            X_final = np.array(all_X)
            F_final = np.array(all_F)
        else:
            # Fallback: return a single random point
            X_final = np.array([lo + (hi - lo) * np.random.random(self.dim)])
            F_final = np.array([problem(X_final[0])])
            eval_count += 1

        return X_final
# EVOLVE-BLOCK-END
