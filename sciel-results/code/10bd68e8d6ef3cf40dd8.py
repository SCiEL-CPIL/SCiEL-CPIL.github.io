"""
Multi-start Nelder-Mead optimizer for BBOB f7 (Step ellipsoidal) in 5D.

Strategy: Nelder-Mead is a direct search method that does not rely on gradient
estimates or covariance adaptation. For the Step Ellipsoidal function, which has
a flat region and a sharp discontinuity (the "step"), Nelder-Mead can be effective
if the simplex is positioned near the step. The "multi-start" aspect uses the
first portion of the budget to race several diverse starting points, keeping the
best one, and then the remaining budget to aggressively refine that best point.
This avoids the slow "learning" phase of CMA-ES and aims for faster early progress.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Multi-start Nelder-Mead simplex method.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start racing
        # Use first 50% of budget to find a good starting point for refinement.
        # We will try 5 diverse starting points, allocating 10% of budget to each.
        num_starts = 5
        race_budget_per_start = int(budget * 0.5 / num_starts)
        
        # Diverse starting points: 0, and random points in [-4, 4]
        starts = [np.zeros(dim)]
        np.random.seed(42) # Fixed seed for reproducibility of starts? 
                           # The task says "hidden seeded shift", but our optimizer 
                           # can use its own randomness. However, for deterministic 
                           # behavior in this context, let's just use a few fixed 
                           # diverse points or random. Let's use random with a 
                           # seed derived from time to be "random" but reproducible 
                           # within a run? No, standard practice is just np.random.
                           # But wait, the evaluator might run multiple instances.
                           # Let's just use np.random.
        
        # Actually, let's use a mix of 0 and random points.
        # To ensure diversity, let's generate random points.
        for _ in range(num_starts - 1):
            pt = np.random.uniform(-4, 4, dim)
            starts.append(pt)

        # We need to be careful with the budget.
        # If we use 50% of budget for racing, we have 50% for refinement.
        # Let's refine the best point found in racing.
        
        # We will run a simplified Nelder-Mead for each start, limited by race_budget_per_start.
        # But standard Nelder-Mead needs a simplex (dim+1 points).
        # Setup:
        # Initial simplex: start + alpha * e_i for i=1..dim.
        # Alpha = 0.1 * box_size? Box is [-5,5], so size 10. Alpha = 0.5?
        # Let's use alpha = 0.5.
        
        current_best_x = np.zeros(dim)
        current_best_f = np.inf

        for start_pt in starts:
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            
            # Initialize simplex
            alpha = 0.5
            simplex = [start_pt.copy()]
            for i in range(dim):
                pt = start_pt.copy()
                pt[i] += alpha
                simplex.append(pt)
            
            # Evaluate initial simplex
            f_vals = []
            for pt in simplex:
                if calls_made >= budget:
                    break
                f_val = evaluate(pt)
                f_vals.append(f_val)
            
            if len(f_vals) < dim + 1:
                continue
                
            f_vals = np.array(f_vals)
            
            # Nelder-Mead loop
            max_iter = race_budget_per_start
            # Standard NM parameters
            alpha_reflect = 1.0
            gamma_expand = 2.0
            rho_contract = 0.5
            sigma_shrink = 0.5
            
            # We track the number of function calls in this specific NM run
            # to stop if we exceed the allocated race budget for this start.
            # Note: evaluate() increments global calls_made.
            # We need to check calls_made against a limit.
            start_calls = calls_made
            limit_calls = start_calls + race_budget_per_start
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_made >= limit_calls:
                    break
                if calls_made >= budget:
                    break
                    
                # Sort simplex by fitness
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = f_vals[order]
                
                # Compute centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Worst point
                x_worst = simplex[-1]
                f_worst = f_vals[-1]
                
                # Reflection
                x_reflect = centroid + alpha_reflect * (centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                
                if calls_made >= limit_calls: break
                f_reflect = evaluate(x_reflect)
                
                if f_reflect < f_vals[0]:
                    # Expansion
                    if calls_made >= limit_calls: break
                    x_expand = centroid + gamma_expand * (centroid - x_worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = evaluate(x_expand)
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_vals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_vals[-1] = f_reflect
                elif f_reflect < f_vals[-2]:
                    # Simple reflection
                    simplex[-1] = x_reflect
                    f_vals[-1] = f_reflect
                else:
                    # Contraction
                    if f_reflect < f_worst:
                        # Outside contraction
                        x_contract = centroid + rho_contract * (x_reflect - centroid)
                    else:
                        # Inside contraction
                        x_contract = centroid - rho_contract * (x_worst - centroid)
                    
                    x_contract = np.clip(x_contract, -5, 5)
                    if calls_made >= limit_calls: break
                    f_contract = evaluate(x_contract)
                    
                    if f_contract <= min(f_reflect, f_worst):
                        simplex[-1] = x_contract
                        f_vals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                        
                        if calls_made >= limit_calls: break
                        f_vals[i] = evaluate(simplex[i])
                        
            # Update global best after this start
            # We need to find the best point in the current simplex
            # But evaluate() already updated global best_x/best_f.
            # So current_best_x is effectively best_x.
            
        # Phase 2: Refinement
        # Use the remaining budget to refine the best point found so far.
        # We will run a longer Nelder-Mead starting from best_x.
        # Initial simplex around best_x.
        # We need to be careful not to re-evaluate points we already know?
        # The function is black-box, so we just call it.
        # However, re-evaluating best_x is a waste.
        # We can construct the simplex such that one vertex is best_x.
        
        if calls_made < budget:
            # Construct simplex
            # Vertex 0: best_x
            # Vertices 1..dim: best_x + small_delta * e_i
            # Small delta to ensure we are in the local neighborhood.
            # Let's estimate a good delta. 
            # If we converged in phase 1, the simplex size might be small.
            # Let's use a fixed small delta like 0.1 or 0.01.
            # Or use the spread of the last simplex from phase 1?
            # We didn't store it. Let's just use a heuristic.
            # If best_f is very good, use small delta.
            # Let's try delta = 0.1.
            
            delta = 0.1
            simplex = [best_x.copy()]
            for i in range(dim):
                pt = best_x.copy()
                pt[i] += delta
                simplex.append(pt)
            
            f_vals = []
            # Evaluate best_x again? No, we know it's best_f.
            # But the NM algorithm expects f_vals[0] to correspond to simplex[0].
            # If we don't call evaluate, we save a call.
            f_vals.append(best_f)
            for i in range(1, dim + 1):
                if calls_made >= budget:
                    break
                f_val = evaluate(simplex[i])
                f_vals.append(f_val)
            
            if len(f_vals) < dim + 1:
                return best_x, best_f

            f_vals = np.array(f_vals)
            
            # NM Loop for refinement
            # Budget remaining: budget - calls_made
            # We will run until budget is exhausted or time limit.
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                    
                # Sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = f_vals[order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = f_vals[-1]
                
                # Reflection
                x_reflect = centroid + alpha_reflect * (centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                
                if calls_made >= budget: break
                f_reflect = evaluate(x_reflect)
                
                if f_reflect < f_vals[0]:
                    if calls_made >= budget: break
                    x_expand = centroid + gamma_expand * (centroid - x_worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = evaluate(x_expand)
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_vals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_vals[-1] = f_reflect
                elif f_reflect < f_vals[-2]:
                    simplex[-1] = x_reflect
                    f_vals[-1] = f_reflect
                else:
                    if f_reflect < f_worst:
                        x_contract = centroid + rho_contract * (x_reflect - centroid)
                    else:
                        x_contract = centroid - rho_contract * (x_worst - centroid)
                    
                    x_contract = np.clip(x_contract, -5, 5)
                    if calls_made >= budget: break
                    f_contract = evaluate(x_contract)
                    
                    if f_contract <= min(f_reflect, f_worst):
                        simplex[-1] = x_contract
                        f_vals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            if calls_made >= budget: break
                            f_vals[i] = evaluate(simplex[i])
                        
                        if calls_made >= budget: break

        return best_x, best_f
# EVOLVE-BLOCK-END
