"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Grid-Initialized Nelder-Mead Optimizer for BBOB f11 (Discus).
    
    Strategy:
    1. Coarse Grid Search: Evaluate a small set of points on a coarse grid 
       (e.g., 3x3x3x3x3 or similar) to identify promising basins. 
       This ensures early progress for AOCC and avoids getting stuck in a single bad basin.
    2. Local Refinement: Run a simplified Nelder-Mead (Simplex) method from the 
       best point found in the grid. Nelder-Mead is effective for smooth, 
       moderately convex functions like f11 and requires no gradient information.
    3. Multi-Start: If budget allows, run Nelder-Mead from the top 2-3 grid points.
    
    This approach is deterministic and robust for small dimensions (n=5) and 
    tight budgets (500 calls) compared to population-based methods which may 
    waste evaluations on redundant samples.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Helper to check budget and time
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            # Ensure x is a valid numpy array
            x = np.asarray(x, dtype=float)
            # Clip to box [-5, 5]
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Coarse Grid Initialization ---
        # For dim=5, a 3-point grid per dimension gives 3^5 = 243 points.
        # This is within the 500 budget and provides good coverage of [-5, 5].
        # We can use a slightly sparser grid if we want to save budget for refinement.
        # Let's use a 3x3x3x3x3 grid = 243 calls.
        # This leaves ~257 calls for refinement.
        
        # Define grid points: -5, 0, 5
        # Actually, f11 often has the minimum shifted. 
        # Using -4, 0, 4 might be better if the minimum is in [-4,4].
        # The problem states minimizer is in [-4, 4]^5.
        # So grid points: -4, 0, 4.
        
        grid_points = np.array(np.meshgrid([[-4.0, 0.0, 4.0]] * dim, indexing='ij')).reshape(dim, -1).T
        
        # Evaluate all grid points
        for x in grid_points:
            if not can_call():
                break
            evaluate(x)
            
        if best_x is None:
            # Fallback if something went wrong
            x0 = np.zeros(dim)
            evaluate(x0)
            if best_x is None:
                return np.zeros(dim)

        # --- Phase 2: Local Refinement with Nelder-Mead ---
        # We will run Nelder-Mead from the best point found.
        # If budget allows, we might run it from the second best point too.
        
        # Simple Nelder-Mead implementation
        def nelder_mead(x_start, max_iter=50):
            nonlocal calls_made, best_x, best_f
            
            n = dim
            # Initialize simplex
            # Vertices: x_start and x_start + step * e_i
            step = 0.5
            simplex = [x_start.copy()]
            f0 = best_f # We already evaluated x_start in Phase 1? 
                        # Actually, evaluate() updates best_f. 
                        # But we need the value for the simplex vertex.
                        # Let's re-evaluate or store it. 
                        # To be safe and consistent with call count, 
                        # we assume x_start is the current best_x which was evaluated.
                        # However, Nelder-Mead needs the function value of the initial vertex.
                        # We can just use best_f as f(x_start) since x_start IS best_x.
                        
            for i in range(n):
                v = x_start.copy()
                v[i] += step
                simplex.append(v)
            
            # Evaluate initial simplex vertices (except x_start which we know)
            f_values = [best_f]
            for i in range(1, n + 1):
                if not can_call():
                    return
                fx = evaluate(simplex[i])
                f_values.append(fx)
                
            # Main Nelder-Mead Loop
            for _ in range(max_iter):
                if not can_call():
                    break
                    
                # Sort simplex by function value
                # Create list of (f, x) pairs
                pairs = sorted(zip(f_values, simplex), key=lambda p: p[0])
                f_values = [p[0] for p in pairs]
                simplex = [p[1] for p in pairs]
                
                # Check for convergence
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()
                    
                # If the difference between best and worst is very small, stop
                if abs(f_values[-1] - f_values[0]) < 1e-8:
                    break
                
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                x_best = simplex[0]
                f_best = f_values[0]
                
                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = evaluate(x_ref)
                
                if f_ref < f_best:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - x_worst)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = evaluate(x_con)
                    
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            if not can_call():
                                return
                            f_values[i] = evaluate(simplex[i])
                        # Re-sort needed? The loop sorts at the beginning.
                        # But we updated indices. 
                        # The next iteration will sort.
                        
            # Final update
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()

        # Run Nelder-Mead from the best point
        # We have ~250 calls left. 
        # Nelder-Mead with n=5 uses ~6 calls per iteration (1 initial + 1-2 per step).
        # 250 / 6 ~ 40 iterations. This is reasonable.
        
        nelder_mead(best_x, max_iter=100)
        
        # If we still have significant budget left, try a second best point from grid?
        # We didn't store the second best explicitly, but we can just run NM again 
        # if we detect we haven't improved much? 
        # For simplicity and robustness, one good local search is often sufficient 
        # for f11 if the grid is good enough to find the basin.
        
        return best_x

# EVOLVE-BLOCK-END
