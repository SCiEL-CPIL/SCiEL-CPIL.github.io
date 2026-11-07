"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Powell's method (conjugate directions) for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget

        # Start at origin
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # Initial directions: standard basis
        directions = np.eye(dim)
        f_min = fx

        # Helper: line search along a direction from a point
        # Returns (new_point, new_f, min_point, min_f, calls_used)
        def line_search(x0, f0, d, remaining_budget, deadline):
            # Initial step size: try 1.0 first
            # We will do a small number of bisection-like or golden section steps
            # To save calls, we use a simple heuristic: try +/- 1, then refine
            # Actually, for Powell, we need to find the minimum along the line.
            
            # Strategy: 
            # 1. Evaluate at x0 + 1*d and x0 - 1*d
            # 2. If one is better, bracket the minimum and do 2-3 bisection steps
            # 3. If neither is better, the minimum is likely at x0 (or very close), return x0
            #    But Powell assumes we move to the min. If f(x0+1) > f0 and f(x0-1) > f0, 
            #    we might just stay or do a tiny step. 
            #    However, usually we can assume the min is between -1 and 1 if we started well.
            #    Let's use a more robust approach:
            #    Evaluate f(x0 + step*d) for step in [1, -1].
            #    If f(x0+1) < f0: search in [0, 1] (or [0, 2] if f(x0+2)<f(x0+1)?)
            #    If f(x0-1) < f0: search in [-1, 0]
            #    Else: return x0, f0 (0 additional calls if we assume x0 is local min along d)
            
            # To be safe and efficient, let's do a fixed number of evaluations.
            # Let's try:
            # 1. f_plus = f(x0 + 1.0 * d)
            # 2. f_minus = f(x0 - 1.0 * d)
            
            calls_local = 0
            best_x = x0
            best_f = f0
            
            if remaining_budget < 3:
                return best_x, best_f, calls_local

            x_plus = x0 + 1.0 * d
            x_plus = np.clip(x_plus, -5, 5)
            f_plus = func(x_plus)
            calls_local += 1
            if f_plus < best_f:
                best_x = x_plus
                best_f = f_plus

            if remaining_budget < calls_local + 1:
                return best_x, best_f, calls_local

            x_minus = x0 - 1.0 * d
            x_minus = np.clip(x_minus, -5, 5)
            f_minus = func(x_minus)
            calls_local += 1
            if f_minus < best_f:
                best_x = x_minus
                best_f = f_minus

            # If best is x0, we might still try to refine if we have budget, 
            # but for Powell, if f(x0) is better than neighbors, we assume min is at x0.
            # If best is x_plus, we know min is in [0, 1] (assuming convexity roughly) or [0, 2]?
            # Let's assume the minimum is between the two best points we found.
            
            # Identify the bracket
            # Points evaluated: x0 (0), x_plus (1), x_minus (-1)
            # Values: f0, f_plus, f_minus
            
            if best_x is x0:
                # If x0 is best, the min is likely near x0. 
                # We could do a smaller step, but let's just return x0 to save calls.
                # Or try step 0.5?
                if remaining_budget >= calls_local + 1:
                    x_half = x0 + 0.5 * d
                    x_half = np.clip(x_half, -5, 5)
                    f_half = func(x_half)
                    calls_local += 1
                    if f_half < best_f:
                        best_x = x_half
                        best_f = f_half
                return best_x, best_f, calls_local

            # If best is x_plus (at t=1) or x_minus (at t=-1)
            # We need to refine the minimum.
            # Let's use 2 bisection steps on the interval [0, 1] or [-1, 0]
            # Actually, the interval should be [0, 1] if x_plus is best, because f(0) > f(1) is false? 
            # Wait, if x_plus is best, then f_plus < f0 and f_plus < f_minus.
            # So the min is likely between 0 and 1? Not necessarily, could be >1.
            # But for Sphere, if we are close, 1 is a good step.
            
            # Let's do a simple bisection on [0, 1] if x_plus is best, or [-1, 0] if x_minus is best.
            # This assumes the min is within 1 unit.
            
            if best_x is x_plus:
                a, b = 0.0, 1.0
                fa, fb = f0, f_plus
            else: # best_x is x_minus
                a, b = -1.0, 0.0
                fa, fb = f_minus, f0
            
            # Bisection for 2 iterations
            for _ in range(2):
                if remaining_budget < calls_local + 1:
                    break
                m = (a + b) / 2.0
                x_m = x0 + m * d
                x_m = np.clip(x_m, -5, 5)
                f_m = func(x_m)
                calls_local += 1
                
                if f_m < fa:
                    b = m
                    fb = f_m
                else:
                    a = m
                    fa = f_m
                
                # Update best
                if f_m < best_f:
                    best_x = x_m
                    best_f = f_m
                # Also check if endpoint is better? We already have fa, fb.
                # Actually, the min is at the end of the interval where the value is lower.
                # After bisection, the min is either at a or b.
                
            # Final check of the two bracket endpoints
            # The current best_x might be the midpoint from the last iteration.
            # We should compare with the final a and b points if we haven't evaluated them recently?
            # We have fa and fb values.
            if fa < best_f:
                best_x = x0 + a * d
                best_x = np.clip(best_x, -5, 5)
                best_f = fa
            if fb < best_f:
                best_x = x0 + b * d
                best_x = np.clip(best_x, -5, 5)
                best_f = fb

            return best_x, best_f, calls_local

        # Main Powell Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            old_min = f_min
            delta = 0.0
            first_dir_index = 0
            min_dir_index = 0
            min_f = f_min

            # Cycle through all directions
            for i in range(dim):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                d = directions[i]
                new_x, new_f, calls_used = line_search(x, fx, d, budget - calls, deadline)
                calls += calls_used
                
                if calls >= budget:
                    break
                
                # Update current position
                x = new_x
                fx = new_f
                
                # Track the direction that yielded the largest improvement
                improvement = old_min - new_f
                if improvement > delta:
                    delta = improvement
                    min_dir_index = i
                    min_f = new_f
                    first_dir_index = i if i == 0 else first_dir_index
                    # Actually, we need to track the first direction's improvement too for the test
                    # Standard Powell:
                    # 1. Perform minimization along each direction d_i, updating x.
                    # 2. Let x_0 be the point before the cycle.
                    # 3. Let x_1 be the point after the cycle.
                    # 4. Let d_min be the direction that gave the largest decrease.
                    # 5. Compute x_2 = 2*x_1 - x_0 (reflection of x_0 across x_1)
                    # 6. Minimize along d_min from x_2 to get x_3? No, that's not quite right.
                    # 
                    # Correct Powell Step:
                    # 1. Start with x0.
                    # 2. For i=1 to N: minimize f(x + t*d_i) to get x_i.
                    #    Let x_0 be the starting point.
                    #    Let x_N be the ending point.
                    #    Let d_min be the direction corresponding to the largest decrease in f.
                    #    Let f_min be the value at the end of that specific minimization? 
                    #    No, usually it's the total decrease along that direction.
                    #    
                    #    Let's stick to a simpler version:
                    #    If the total decrease is significant, add a new direction.
                    #    The new direction is x_N - x_0.
                    #    Replace the direction that gave the smallest decrease (or the first one if the new dir is better).
                    
                    # For now, let's just track if we should update directions.
                    pass

            # After the cycle:
            # x is the new point.
            # We need to decide if we update the direction set.
            # Simple heuristic:
            # If we made good progress, replace the direction that gave the least improvement with the new direction (x - x_start_cycle).
            
            # We didn't track x_start_cycle in the loop above properly. Let's restart the loop logic with better tracking.
            # Actually, the loop above modifies x. We need the x at the start of the cycle.
            
            # Let's restructure the main loop to properly implement Powell's direction update.
            # This requires storing x_before_cycle.
            
            # I will rewrite the main loop structure to be cleaner.
            break # Break out of this flawed implementation to write the correct one below? No, I can't break out of the method.
            # I must implement it correctly in the first pass.
            # Let's assume the code above is a draft. I will replace the whole __call__ body with a correct implementation.

        # Since I cannot "rewrite" in place easily without seeing the final code, 
        # I will provide the correct implementation in the final code block.
        # The code above is just thinking.
        
        return x, fx

    # Correct Implementation
    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget

        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        directions = np.eye(dim)
        
        # To avoid the complexity of full Powell direction replacement which can be tricky with limited budget,
        # I will use a simplified Powell-like method:
        # 1. Cycle through directions.
        # 2. If total improvement is good, add a new direction (x_end - x_start) and remove the worst one.
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            x_start = x.copy()
            f_start = fx
            old_min = fx
            delta = 0.0
            min_dir_index = 0
            min_f = fx
            worst_dir_index = 0
            worst_improvement = float('inf')
            
            # We need to track the improvement per direction
            improvements = [0.0] * dim

            for i in range(dim):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                d = directions[i]
                # Line search
                # Use the line_search helper defined previously? 
                # I need to define it inside or make it a local function.
                
                # Let's inline a simple line search to keep it self-contained in the loop for clarity
                # or define it once.
                
                # I will define a local helper function here for clarity.
                def _line_search(x0, f0, d, rem_budget):
                    # Try +/- 1
                    c = 0
                    bx = x0
                    bf = f0
                    
                    if rem_budget < 1: return bx, bf, c
                    xp = np.clip(x0 + d, -5, 5)
                    fp = func(xp)
                    c += 1
                    if fp < bf: bx, bf = xp, fp
                    
                    if rem_budget < c + 1: return bx, bf, c
                    xm = np.clip(x0 - d, -5, 5)
                    fm = func(xm)
                    c += 1
                    if fm < bf: bx, bf = xm, bf
                    
                    # If best is x0, try 0.5
                    if bx is x0 and rem_budget >= c + 1:
                        xh = np.clip(x0 + 0.5*d, -5, 5)
                        fh = func(xh)
                        c += 1
                        if fh < bf: bx, bf = xh, fh
                        return bx, bf, c
                    
                    # If best is xp or xm, bisect 2 times
                    if bx is xp:
                        a, b, fa, fb = 0.0, 1.0, f0, fp
                    else:
                        a, b, fa, fb = -1.0, 0.0, fm, f0
                        
                    for _ in range(2):
                        if rem_budget < c + 1: break
                        m = (a+b)/2
                        xmid = np.clip(x0 + m*d, -5, 5)
                        fmid = func(xmid)
                        c += 1
                        if fmid < fa:
                            b, fb = m, fmid
                        else:
                            a, fa = m, fmid
                        if fmid < bf: bx, bf = xmid, fmid
                    
                    if fa < bf:
                        bx = np.clip(x0 + a*d, -5, 5)
                        bf = fa
                    if fb < bf:
                        bx = np.clip(x0 + b*d, -5, 5)
                        bf = fb
                    return bx, bf, c

                new_x, new_f, c_used = _line_search(x, fx, d, budget - calls)
                calls += c_used
                if calls >= budget: break
                
                x = new_x
                fx = new_f
                
                imp = f_start - fx # Improvement from start of cycle? Or from previous step?
                # Powell usually compares to the value before the step in this direction.
                # But for direction replacement, we often look at the total drop.
                # Let's use the drop from the value at the start of this specific direction's step.
                # I didn't store that. Let's approximate with f_start - fx for the whole cycle?
                # No, that's not right.
                
                # Let's just track if we improved at all.
                if fx < min_f:
                    min_f = fx
                    min_dir_index = i
                    delta = f_start - fx # Total improvement so far?
                    
                # For worst direction, we need the improvement *along that specific direction*.
                # Let's re-do the loop to store f_before_step.
                
                # This is getting complicated. Let's simplify:
                # Just use the standard Powell update:
                # 1. x0 = x
                # 2. for i in 1..N: x = min(x + t*d_i)
                # 3. if f(x) < f(x0) - epsilon:
                #    d_new = x - x0
                #    f_new = f(x)
                #    x2 = 2*x - x0
                #    f2 = min(x2 + t*d_new)
                #    if f2 < f(x):
                #       replace d_worst with d_new
                #       x = x2
                #       f = f2
                #    else:
                #       replace d_worst with d_new
                #       x = x
                #       f = f(x)
                #    else:
                #       replace d_worst with d_new
                #       x = x
                #       f = f(x)
                
                # I will implement this specific logic.
                pass

            # Since the inline logic is getting messy, I will write a clean version from scratch in the final block.
            break

        return x, fx

# I will provide the clean version in the final output.
# The thinking above confirms the strategy: Powell's method with direction replacement.
# Key benefits: Decoupled convergence, fast for Sphere.
# Key risks: Line search cost. Mitigation: Cheap line search (2-5 evals).
# Budget: 500 calls. Dim 5.
# Cycle: 5 directions * ~3 evals = 15 evals.
# ~33 cycles in 500 evals.
# Plus direction updates.
# This should be sufficient.

# Final Code Structure:
# 1. Init x=0, f=func(0).
# 2. dirs = I.
# 3. Loop:
#    a. x0 = x, f0 = f.
#    b. for i in 1..N:
#         xi = x + t*di (minimize)
#         x = xi, f = f(xi)
#         track worst direction (smallest drop)
#    c. if f < f0 - tol:
#         d_new = x - x0
#         x2 = 2*x - x0
#         f2 = minimize(x2 + t*d_new)
#         if f2 < f:
#             x = x2, f = f2
#         replace worst dir with d_new
#    d. check budget/time
# 4. Return x, f

# Line Search:
# - Try +/- 1
# - If better, bisect 2 times.
# - If not, try 0.5? Or just stay.
# - Cap at 5 evals.

# This is robust.

# Let's write the final code.

# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Powell's method (conjugate directions) for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget

        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        directions = np.eye(dim)
        
        # Helper for line search along direction d from x0
        def line_search(x0, f0, d, rem_budget):
            c = 0
            bx = x0
            bf = f0
            
            # Try +/- 1
            if rem_budget < 1:
                return bx, bf, c
            xp = np.clip(x0 + d, -5, 5)
            fp = func(xp)
            c += 1
            if fp < bf:
                bx, bf = xp, fp

            if rem_budget < c + 1:
                return bx, bf, c
            xm = np.clip(x0 - d, -5, 5)
            fm = func(xm)
            c += 1
            if fm < bf:
                bx, bf = xm, fm

            # If x0 is best, try 0.5
            if bx is x0:
                if rem_budget >= c + 1:
                    xh = np.clip(x0 + 0.5 * d, -5, 5)
                    fh = func(xh)
                    c += 1
                    if fh < bf:
                        bx, bf = xh, fh
                return bx, bf, c

            # If xp or xm is best, bisect
            if bx is xp:
                a, b, fa, fb = 0.0, 1.0, f0, fp
            else:
                a, b, fa, fb = -1.0, 0.0, fm, f0

            for _ in range(2):
                if rem_budget < c + 1:
                    break
                m = (a + b) / 2.0
                xmid = np.clip(x0 + m * d, -5, 5)
                fmid = func(xmid)
                c += 1
                if fmid < fa:
                    b, fb = m, fmid
                else:
                    a, fa = m, fmid
                if fmid < bf:
                    bx, bf = xmid, fmid

            # Check endpoints
            if fa < bf:
                bx = np.clip(x0 + a * d, -5, 5)
                bf = fa
            if fb < bf:
                bx = np.clip(x0 + b * d, -5, 5)
                bf = fb

            return bx, bf, c

        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break

            x0 = x.copy()
            f0 = fx
            f_start_cycle = fx
            worst_dir_idx = 0
            worst_drop = float('inf')

            # Cycle through directions
            for i in range(dim):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                d = directions[i]
                new_x, new_f, c_used = line_search(x, fx, d, budget - calls)
                calls += c_used
                if calls >= budget:
                    break

                drop = fx - new_f
                if drop < worst_drop:
                    worst_drop = drop
                    worst_dir_idx = i

                x = new_x
                fx = new_f

            # Direction Update
            if fx < f_start_cycle - 1e-10:
                d_new = x - x0
                # Check if new direction is worth adding
                # Evaluate at x2 = 2*x - x0
                if calls < budget:
                    x2 = 2 * x - x0
                    x2 = np.clip(x2, -5, 5)
                    f2 = func(x2)
                    calls += 1
                    if calls < budget:
                        # Minimize along d_new from x2
                        x3, f3, c_used = line_search(x2, f2, d_new, budget - calls)
                        calls += c_used
                        if calls < budget:
                            if f3 < fx:
                                x = x3
                                fx = f3
                            # Replace worst direction with d_new
                            directions[worst_dir_idx] = d_new

        return x, fx

# EVOLVE-BLOCK-END
