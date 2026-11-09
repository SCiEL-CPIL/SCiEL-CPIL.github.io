"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For dim=5, standard lambda is ceil(4 + 3*ln(5)) = 11. 
        # We use a slightly larger population for better exploration in the early phase.
        pop_size = 12 
        sigma = 2.0  # Initial step size, covering a good portion of [-5, 5]
        
        # State variables
        x_mean = np.zeros(dim)
        best_x = x_mean.copy()
        best_f = float('inf')
        call_count = 0
        
        # CMA-ES internal state
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Pre-compute matrix square root and inverse square root for efficiency
        # We will recompute these if C changes significantly, but for dim=5, 
        # doing it every generation is cheap enough if we are careful.
        
        def update_C(x_mean, C, C_inv, weights, sigma, pop_size):
            # Standard CMA-ES update
            # This is a simplified version. Full CMA-ES requires careful handling of 
            # cumulation paths. For dim=5, a simpler adaptive scheme might suffice, 
            # but let's stick to a robust basic CMA-ES logic.
            
            # Actually, implementing full CMA-ES from scratch is error-prone.
            # SciPy's minimize with method='Powell' or 'Nelder-Mead' is derivative-free but often slower than CMA-ES for Rosenbrock.
            # However, the prompt says "Python standard library, NumPy and SciPy are available".
            # SciPy does NOT have a built-in CMA-ES in `scipy.optimize.minimize`.
            # `scipy.optimize.dual_annealing` is available but stochastic.
            # `scipy.optimize.differential_evolution` is available.
            
            # Let's reconsider. The current program uses L-BFGS-B.
            # A "Structural Change" could be switching to a **Derivative-Free Global Optimizer** 
            # that is known to handle Rosenbrock well without gradients.
            # `scipy.optimize.differential_evolution` is robust but can be slow.
            # `scipy.optimize.dual_annealing` is another option.
            
            # However, a very effective approach for black-box continuous optimization 
            # when gradients are not available (or expensive) is **Nelder-Mead** with multiple restarts, 
            # or **Powell's Method**. 
            # The current program uses L-BFGS-B which *estimates* gradients via finite differences.
            # This costs 5-10 function evaluations per gradient step.
            
            # A true derivative-free method like **Nelder-Mead** uses 1 eval per vertex movement.
            # For dim=5, Nelder-Mead uses 6 vertices.
            # It is often slower to converge than L-BFGS-B for smooth functions, 
            # BUT it doesn't suffer from finite-difference noise or step-size issues in the same way.
            
            # Let's try a different structural change: **Multi-Start Nelder-Mead** 
            # combined with **Random Restarts based on Stagnation**.
            # Nelder-Mead is simple, fast per iteration, and robust.
            
            pass

        # Let's implement a robust Multi-Start Nelder-Mead strategy.
        # Nelder-Mead is derivative-free.
        # Budget: 500 calls.
        # Nelder-Mead in dim=5 typically requires ~50-100 iterations to converge.
        # Each iteration uses ~1-5 function evaluations (reflection, expansion, contraction).
        # So ~100-500 evals per run.
        # We can afford 5-10 runs.
        
        from scipy.optimize import minimize
        
        rng = np.random.default_rng(42)
        
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return best_f
            call_count += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except:
                return 1e100

        bounds = [(-5, 5)] * dim
        
        # Strategy: 
        # 1. Random initialization of a few points to find a good basin.
        # 2. Run Nelder-Mead from the best point found so far.
        # 3. If budget remains, run Nelder-Mead from other good points.
        
        # Initial random sampling to find a good starting point
        # Use 10% of budget for initial sampling
        init_budget = max(10, budget // 10)
        
        # We already have best_x as zeros. Let's sample a few random points.
        for _ in range(init_budget):
            if call_count >= budget:
                break
            x0 = rng.uniform(-5, 5, dim)
            f0 = counted_func(x0)
            
        # Now run Nelder-Mead from the best point found
        # Nelder-Mead options:
        # maxiter: max number of iterations.
        # ftol, xtol: tolerances.
        
        # We will do multiple rounds.
        # Round 1: From best_x
        # Round 2: From second best? We didn't track second best. 
        # Let's just run from best_x, and if it fails to improve significantly, 
        # try a new random start.
        
        # To make it more robust, we'll do a "hill-climbing" restart strategy:
        # 1. Run NM from current best.
        # 2. If it converges (or uses up its local budget), check if we can improve.
        # 3. If not, generate a new random point far from best and run NM again.
        
        nm_maxiter = 100  # Limit iterations per NM run
        nm_budget_per_run = 150 # Limit function calls per NM run approx
        
        while call_count < budget - 20 and time.time() < deadline:
            # Run Nelder-Mead
            try:
                # Nelder-Mead is derivative-free.
                # We pass the counted_func.
                # Note: Nelder-Mead does not support bounds natively.
                # We must rely on the fact that the function clips or we penalize out-of-bounds.
                # The problem statement says: "func(x) clips a finite length-5 vector to the box".
                # So out-of-bounds is safe, it just evaluates the boundary.
                # However, NM might wander far away. 
                # To keep it in bounds, we can use a large initial simplex size but let it shrink.
                
                res = minimize(counted_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': nm_maxiter, 'xatol': 1e-6, 'fatol': 1e-8})
                
                # Update best if NM found something better
                # counted_func already updates best_f and best_x
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
            
            # Check if we made significant progress
            # If NM converged quickly, we might be stuck in a local minimum.
            # If it used up maxiter, it might still be improving.
            
            # Heuristic: If the last NM run didn't improve best_f significantly 
            # or if we have budget for another restart, do a random restart.
            
            # How to detect "stuck"? 
            # We can check if best_f changed during the last run.
            # We need to store best_f before the run.
            
            # Let's restructure the loop to track improvement.
            
            break # The above loop was a bit messy. Let's rewrite the main loop properly.
        
        # Rewriting main loop for clarity and robustness
        
        # Reset state for the structured loop
        # (Note: We already consumed some budget in the initial sampling above)
        
        # Let's do a series of Nelder-Mead runs from random restarts.
        # We'll pick a random point, run NM, keep the best.
        # Repeat until budget is exhausted.
        
        # To ensure we explore different basins, we'll generate random restarts 
        # that are not too close to the current best.
        
        # Since we already did initial sampling, let's continue with NM runs.
        
        # We'll run NM from the current best_x.
        # Then, if we have budget, we'll try a new random point.
        
        # Let's do 5 total NM runs (including the one from the initial best).
        # We already have a best_x from initial sampling.
        
        # Run 1: From current best_x
        try:
            res = minimize(counted_func, best_x, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except:
            pass
        
        # Runs 2-5: From new random points
        for i in range(4):
            if call_count >= budget - 50: # Reserve 50 for final polish
                break
            if time.time() > deadline:
                break
                
            # Generate a random point
            x0 = rng.uniform(-5, 5, dim)
            
            # Optional: Ensure it's not too close to best_x? 
            # For global search, random is fine.
            
            try:
                res = minimize(counted_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except:
                pass
        
        # Final Polish: Use L-BFGS-B from the best point found by NM.
        # L-BFGS-B is very efficient for final convergence on smooth functions.
        if call_count < budget - 10 and time.time() < deadline:
            try:
                res = minimize(counted_func, best_x, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
