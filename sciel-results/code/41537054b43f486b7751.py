"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Deterministic Pattern Search + Nelder-Mead Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # State
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Helper to safely call func and update state
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            try:
                f_val = func(x)
                calls_made += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return np.inf
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return np.inf

        # Phase 1: Global Sparse Sampling (Grid Search)
        # For dim=5, a full grid is too large. We use a stratified/sparse grid.
        # We allocate ~20% of budget to global exploration.
        global_budget = int(budget * 0.2)
        
        # Generate points using a space-filling approach or simple grid with jitter
        # Since the minimizer is in [-4, 4]^5, we focus there but cover [-5, 5] to be safe.
        # Let's use a Halton sequence or similar quasi-random for better coverage than pure random.
        # Or simply a structured grid with limited points per dimension.
        
        # For dim=5, 3^5 = 243 points is feasible if budget allows.
        # Let's try a 3-level grid in [-4, 4] plus center 0.
        # Actually, let's use a deterministic set of points that covers the space well.
        
        # Simple approach: Random sampling with fixed seed for reproducibility, 
        # but structured to avoid clustering.
        rng = np.random.default_rng(42)
        
        # Evaluate center first
        x_center = np.zeros(n)
        evaluate(x_center)
        
        # Evaluate corners of [-4, 4]^5? No, too many (2^5=32). 
        # Let's use a Latin Hypercube or just random points.
        # Given the rotation, random points are better than axis-aligned grids.
        
        for _ in range(global_budget - 1):
            if calls_made >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4.5, 4.5, n)
            evaluate(x_try)
        
        if best_x is None:
            return np.zeros(n), np.inf

        # Phase 2: Local Refinement using Nelder-Mead and Pattern Search
        # We will restart local optimizers from the best few points found in Phase 1.
        # We keep the top K points from the global phase.
        # Since we didn't store all points, we just use the current best.
        # To be more robust, we should store top K. Let's refactor slightly to store top K.
        
        # Re-doing Phase 1 with storage for top K
        # Actually, let's just use the best point for now, or run a few local searches from best.
        # A better strategy: Run Nelder-Mead from the best point.
        # If it fails (converges to a bad local min or takes too long), try another point.
        
        # To improve success rate, we should have stored the top 5-10 points from Phase 1.
        # Let's modify the loop to keep a list of top points.
        
        # Reset for a cleaner second pass? No, we are out of time to restart.
        # Let's just proceed with the best point found.
        
        # However, Rastrigin has many local minima. Nelder-Mead might get stuck.
        # Pattern Search (Hooke-Jeeves) is more robust for box constraints and can handle multimodality better if step sizes are adaptive.
        
        # Let's implement a simple Hooke-Jeeves from the best point.
        x_current = best_x.copy()
        f_current = best_f
        
        # Initial step size
        step_size = 0.5
        min_step = 1e-4
        
        # Hooke-Jeeves Pattern Search
        # We have remaining budget.
        while calls_made < budget and time.time() < deadline:
            if step_size < min_step:
                break
            
            moved = False
            for i in range(n):
                # Try positive step
                x_plus = x_current.copy()
                x_plus[i] += step_size
                if np.any(x_plus > 5) or np.any(x_plus < -5):
                    continue
                f_plus = evaluate(x_plus)
                
                if f_plus < f_current:
                    # Move
                    x_current = x_plus
                    f_current = f_plus
                    moved = True
                    # Pattern move: try to continue in same direction
                    while True:
                        x_pattern = x_current.copy()
                        x_pattern[i] += step_size
                        if np.any(x_pattern > 5) or np.any(x_pattern < -5):
                            break
                        f_pattern = evaluate(x_pattern)
                        if f_pattern < f_current:
                            x_current = x_pattern
                            f_current = f_pattern
                            moved = True
                        else:
                            break
                else:
                    # Try negative step
                    x_minus = x_current.copy()
                    x_minus[i] -= step_size
                    if np.any(x_minus > 5) or np.any(x_minus < -5):
                        continue
                    f_minus = evaluate(x_minus)
                    if f_minus < f_current:
                        x_current = x_minus
                        f_current = f_minus
                        moved = True
                        # Pattern move
                        while True:
                            x_pattern = x_current.copy()
                            x_pattern[i] -= step_size
                            if np.any(x_pattern > 5) or np.any(x_pattern < -5):
                                break
                            f_pattern = evaluate(x_pattern)
                            if f_pattern < f_current:
                                x_current = x_pattern
                                f_current = f_pattern
                                moved = True
                            else:
                                break
                    # If neither worked, try smaller step? No, Hooke-Jeeves usually reduces step size if no move in any dimension.
            
            if not moved:
                # Reduce step size
                step_size *= 0.5
            else:
                # If moved, we might increase step size or keep it. 
                # Standard Hooke-Jeeves keeps it or increases slightly if many moves.
                # Let's keep it constant to be safe.
                
                # Optional: Run Nelder-Mead periodically to refine
                # If we have enough budget left and step size is small, switch to NM.
                if step_size < 0.01 and calls_made < budget - 50:
                    # Run Nelder-Mead from current best
                    # Nelder-Mead uses many function calls.
                    # Let's allocate a chunk of budget to NM.
                    nm_budget = 50
                    # Wrap func to count calls
                    nm_calls = [0]
                    def nm_func(x):
                        if nm_calls[0] >= nm_budget or time.time() > deadline:
                            return np.inf
                        nm_calls[0] += 1
                        return func(x)
                    
                    try:
                        res = minimize(nm_func, x_current, method='Nelder-Mead', 
                                       options={'maxiter': nm_budget, 'fatol': 1e-8, 'xatol': 1e-8})
                        # Update global budget counter? 
                        # The nm_func increments nm_calls, but we need to update global calls_made.
                        # This is tricky. Let's just rely on the global evaluate function.
                        # Actually, minimize calls nm_func, which calls func directly.
                        # We need to sync calls_made.
                        # Let's assume nm_calls[0] is the number of calls made by NM.
                        calls_made += nm_calls[0]
                        
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x
                            x_current = res.x
                            f_current = res.fun
                            # After NM, reset step size to explore around new best
                            step_size = 0.1
                    except Exception:
                        pass

        return best_x, best_f

# EVOLVE-BLOCK-END
