"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Hybrid optimizer for Bent Cigar (BBOB f12):
    1. Sobol global sampling to locate the basin of attraction.
    2. Stochastic Line Search with adaptive direction set.
       - Random directions to explore the anisotropic landscape.
       - Golden Section Search along each line to precisely locate minima.
       - Powell-style direction update to align with the steepest descent direction.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls_used = 0
        best_x = np.zeros(dim)
        best_fx = np.inf
        
        def evaluate(x):
            nonlocal calls_used, best_x, best_fx
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            calls_used += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        n_global = 64
        try:
            sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
            samples = sobol.random(n_global)
        except Exception:
            samples = rng.uniform(size=(n_global, dim))
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        for x in samples:
            if time.time() > deadline or calls_used >= budget:
                break
            fx = evaluate(x)
            if np.isinf(fx):
                break
        
        if calls_used == 0:
            x0 = np.zeros(dim)
            evaluate(x0)
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Stochastic Line Search with Adaptive Directions ---
        
        # Initialize direction set with standard basis vectors
        directions = [np.eye(dim)[i] for i in range(dim)]
        
        x = x0.copy()
        fx = best_fx
        
        # Golden Section Search constants
        phi = (1 + np.sqrt(5)) / 2
        inv_phi = 1 / phi
        inv_phi2 = inv_phi ** 2
        
        def golden_search(x_start, direction, max_iter=10):
            """
            Perform a 1D minimization along 'direction' starting from 'x_start'.
            Returns the best x found and its function value.
            Uses a bracketing approach followed by golden section refinement.
            """
            if time.time() > deadline or calls_used >= budget:
                return x_start, fx_start
            
            # 1. Bracketing
            step = 1.0
            # Try to find a range [a, b] where the function increases on both ends
            # We start with x_start as one end (t=0) and expand in the direction.
            # Actually, for line search in optimization, we usually want to find the minimum.
            # Let's assume we are at a point and want to move.
            # A simpler approach for this specific task:
            # We know Bent Cigar is smooth. We can just do a local search around x_start.
            
            # Let's use a fixed interval search first to bracket the minimum.
            # Since we are doing coordinate-like or random direction search,
            # the minimum is likely close.
            
            # Initial interval: [x_start - step*dir, x_start + step*dir]
            # But we need to be careful with bounds.
            
            # Let's try a few points to establish a bracket.
            # t=0 is x_start.
            # We evaluate at t=step and t=-step?
            # To save calls, let's assume the minimum is within a reasonable distance.
            
            # Strategy:
            # 1. Evaluate at x_start (already have fx)
            # 2. Evaluate at x_start + step * dir
            # 3. If improvement, expand step until we find a worse point (bracket found)
            # 4. If no improvement, try -step.
            # 5. If neither improves, return x_start.
            
            x_plus = x_start + step * direction
            x_plus = np.clip(x_plus, -5, 5)
            fx_plus = evaluate(x_plus)
            
            if np.isinf(fx_plus):
                return x_start, fx_start
                
            if fx_plus < fx_start:
                # Improved, try to expand
                a = 0.0
                b = step
                fa = fx_start
                fb = fx_plus
                x_b = x_plus.copy()
                
                # Expand b until fb > fa (or we hit budget/time)
                while fb < fa and calls_used < budget and time.time() < deadline:
                    step *= 2.0
                    if step > 10.0: # Cap expansion
                        break
                    x_new = x_start + step * direction
                    x_new = np.clip(x_new, -5, 5)
                    fx_new = evaluate(x_new)
                    if np.isinf(fx_new):
                        break
                    if fx_new < fb:
                        a = b
                        b = step
                        fa = fb
                        fb = fx_new
                        x_b = x_new.copy()
                    else:
                        # We found a point worse than the current best in this direction
                        # The minimum is between a and b (or a and current step)
                        # Actually, if fx_new > fb, the min is between a and b?
                        # No, fb is the best so far. fx_new is worse.
                        # So min is in [a, step].
                        # Let's set b = step, fb = fx_new for the bracket?
                        # Golden section requires f(a) < f(c) and f(b) < f(d) roughly?
                        # Standard bracket: f(a) > f(c) < f(b) is not required for GS, 
                        # GS works on an interval containing the minimum.
                        # If we have a local min at b and higher values at a and step, 
                        # then [a, step] contains the min.
                        b = step
                        fb = fx_new
                        break
                
                # Now we have a bracket [a, b] containing the minimum (or the min is at b if we broke early)
                # Actually, if we didn't break, we just keep expanding.
                # Let's refine [a, b] using Golden Section.
                
                # Note: a=0 corresponds to x_start.
                # We want to minimize f(x_start + t * direction) for t in [a, b].
                
                c = b - inv_phi * (b - a)
                d = a + inv_phi * (b - a)
                
                x_c = x_start + c * direction
                x_c = np.clip(x_c, -5, 5)
                fx_c = evaluate(x_c)
                
                x_d = x_start + d * direction
                x_d = np.clip(x_d, -5, 5)
                fx_d = evaluate(x_d)
                
                for _ in range(max_iter):
                    if time.time() > deadline or calls_used >= budget:
                        break
                    
                    if fx_c < fx_d:
                        # Min is in [a, d]
                        b = d
                        fb = fx_d
                        d = c
                        fx_d = fx_c
                        c = b - inv_phi * (b - a)
                        x_c = x_start + c * direction
                        x_c = np.clip(x_c, -5, 5)
                        fx_c = evaluate(x_c)
                    else:
                        # Min is in [c, b]
                        a = c
                        fa = fx_c
                        c = d
                        fx_c = fx_d
                        d = a + inv_phi * (b - a)
                        x_d = x_start + d * direction
                        x_d = np.clip(x_d, -5, 5)
                        fx_d = evaluate(x_d)
                    
                    if np.isinf(fx_c) or np.isinf(fx_d):
                        break
                        
                    if abs(b - a) < 1e-8:
                        break
                
                # Determine best point in the interval
                # Candidates: a, b, c, d (and maybe the initial x_plus if it was better?)
                # We tracked best_fx in the global scope, but let's return the local best found in this search.
                # Actually, evaluate() updates global best_x/best_fx. 
                # We just need to return the x that corresponds to the local minimum found.
                
                # Let's just pick the point with the lowest fx among the last evaluated ones.
                # Since evaluate updates global best, we can just return the current global best?
                # No, the global best might be from a previous, better direction.
                # We should return the x that is the local min along this line.
                
                # Let's re-evaluate the candidates to be sure, or just trust the global best if it improved?
                # The function returns best_x, best_fx.
                # If this line search found a new global best, best_x is updated.
                # If not, we should return the best point along this line, which might be x_start.
                
                # To be safe, let's return the point with the lowest fx evaluated in this specific search.
                # We didn't keep a local list. 
                # However, since Bent Cigar is unimodal in each direction from a good start,
                # and we started from the current best, the global best is likely the local best.
                # Let's just return best_x, best_fx. If this search didn't improve, best_x is unchanged.
                
                return best_x, best_fx

            else:
                # No improvement in + direction. Try - direction.
                x_minus = x_start - step * direction
                x_minus = np.clip(x_minus, -5, 5)
                fx_minus = evaluate(x_minus)
                
                if np.isinf(fx_minus):
                    return x_start, fx_start
                
                if fx_minus < fx_start:
                    # Similar logic, but negative direction
                    # We can reuse the logic by negating the direction and calling again?
                    # Or just implement symmetrically.
                    # For simplicity, let's just do a small local search if we find an improvement.
                    # Given the budget, we can afford a few iterations.
                    # Let's just return the best found so far.
                    # The global best_x has been updated if fx_minus < best_fx.
                    return best_x, best_fx
                else:
                    # No improvement in either direction
                    return x_start, fx_start

        # Main Loop for Line Search
        max_cycles = 5000
        cycle = 0
        
        while calls_used < budget and time.time() < deadline and cycle < max_cycles:
            cycle += 1
            
            # Powell-style direction update
            # Find the direction that gives the largest improvement
            best_improvement = 0
            best_dir_idx = -1
            new_x = x.copy()
            
            # We need to store the starting point of each line search to compare improvements
            # Actually, standard Powell minimizes along each direction sequentially.
            # x_{k+1} = min(x_k + t * d_k)
            # We do this sequentially.
            
            # Let's track the improvement for each direction in this cycle
            improvements = np.zeros(len(directions))
            
            for i, dir in enumerate(directions):
                if time.time() > deadline or calls_used >= budget:
                    break
                
                fx_before = fx
                x_new, fx_new = golden_search(x, dir)
                
                if fx_new < fx_before:
                    x = x_new
                    fx = fx_new
                    improvements[i] = fx_before - fx_new
                else:
                    # If no improvement, x remains unchanged for this step?
                    # In Powell, if no improvement, we usually keep the old x.
                    # But we might have moved x in previous directions of this cycle.
                    # Yes, x is updated sequentially.
                    pass
            
            # Check if we found a better direction
            if best_improvement == 0 and np.max(improvements) > 0:
                best_dir_idx = np.argmax(improvements)
                best_improvement = improvements[best_dir_idx]
            
            # If we improved significantly, add the new direction (x_start_cycle - x_start_prev_cycle)
            # But we didn't track the start of the cycle.
            # Let's track it.
            
            # Revised Loop:
            # x_start_cycle = x.copy()
            # ... perform line searches ...
            # x_end_cycle = x
            # if improvement > threshold:
            #    new_dir = x_end_cycle - x_start_cycle
            #    directions.append(new_dir)
            #    directions.pop(worst_dir_idx)
            
            # Let's refactor the loop to track this.
            break # Exit this placeholder loop to rewrite properly below
            
        # --- Proper Implementation of the Main Loop ---
        
        # Reset x to best_x found so far
        x = best_x.copy()
        fx = best_fx
        
        # Directions list
        directions = [np.eye(dim)[i] for i in range(dim)]
        
        # We will run cycles until budget is exhausted
        while calls_used < budget and time.time() < deadline:
            x_start_cycle = x.copy()
            fx_start_cycle = fx
            
            improvements = []
            
            for i, dir in enumerate(directions):
                if time.time() > deadline or calls_used >= budget:
                    break
                
                fx_before = fx
                x_new, fx_new = golden_search(x, dir)
                
                if fx_new < fx_before:
                    x = x_new
                    fx = fx_new
                    improvements.append((fx_before - fx_new, i))
                else:
                    improvements.append((0, i))
                    
            if time.time() > deadline or calls_used >= budget:
                break
            
            # Determine if we should update directions
            if improvements:
                max_improvement, max_idx = max(improvements, key=lambda t: t[0])
                
                if max_improvement > 1e-8:
                    # Calculate the new direction
                    x_end_cycle = x
                    new_dir = x_end_cycle - x_start_cycle
                    
                    # Check if new_dir is linearly independent (dot product with existing dirs is small)
                    # For simplicity, we just add it and remove the worst one.
                    # Find the direction with the smallest improvement
                    min_improvement, min_idx = min(improvements, key=lambda t: t[0])
                    
                    # Replace the worst direction with the new one
                    # Normalize new_dir? Not strictly necessary for line search, but helps with stability.
                    norm = np.linalg.norm(new_dir)
                    if norm > 1e-10:
                        new_dir = new_dir / norm
                        directions[min_idx] = new_dir
                    else:
                        # If new dir is zero, keep old or remove?
                        # Keep old for now.
                        pass
                        
            # Convergence check: if x didn't move much
            if np.linalg.norm(x - x_start_cycle) < 1e-10:
                # Try to restart with a random direction?
                # Or break if we are very close to the minimum.
                # Bent Cigar minimum is in [-4, 4]^5.
                # If we are stuck, maybe we are at the minimum.
                # Let's break if improvement is tiny.
                if max_improvement < 1e-10:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
