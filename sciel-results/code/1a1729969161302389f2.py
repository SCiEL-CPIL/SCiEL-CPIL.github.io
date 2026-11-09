"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Estimate the linear model parameters (gradient 'a' and offset 'b') 
       using finite differences from the origin.
    2. Generate all 32 vertices of the box [-5, 5]^5.
    3. Sort vertices by the predicted function value from the linear model.
    4. Evaluate vertices in order of predicted value.
    5. This ensures the global minimum (which is at a vertex for linear functions)
       is found very early, maximizing the AOCC score.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin
        calls_made = 0
        dim = self.dim
        budget = self.budget

        # 1. Start at origin to estimate gradient cleanly
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_fx = f0

        # 2. Estimate gradient using forward differences
        # f(x) = a^T x + b. f(0) = b.
        # f(h e_i) = a_i h + b.
        # a_i = (f(h e_i) - f(0)) / h
        h = 1.0
        a_est = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break
            
            xi = x0.copy()
            xi[i] += h
            fi = func(xi)
            calls_made += 1
            
            a_est[i] = (fi - f0) / h
            
            if fi < best_fx:
                best_fx = fi
                best_x = xi.copy()

        # 3. Generate all vertices
        vertices = []
        for mask in range(2 ** dim):
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(dim)])
            vertices.append(v)

        # 4. Predict function values at vertices and sort
        # Predicted f(v) = a_est^T v + f0
        predicted_values = []
        for v in vertices:
            pred_val = np.dot(a_est, v) + f0
            predicted_values.append((pred_val, v))
        
        # Sort by predicted value (ascending) to find the minimum first
        predicted_values.sort(key=lambda x: x[0])

        # 5. Evaluate vertices in sorted order
        for pred_val, v in predicted_values:
            if time.time() > deadline or calls_made >= budget:
                break
            
            fv = func(v)
            calls_made += 1
            
            if fv < best_fx:
                best_fx = fv
                best_x = v.copy()

        # 6. If budget remains, do a local search around the best found vertex.
        # This is a safeguard in case the function is not perfectly linear or
        # if there are numerical issues, but primarily it consumes remaining budget
        # without hurting the score (since we've likely already found the min).
        # For a linear function, the minimum is at a vertex, so this should be a no-op
        # in terms of finding a better value, but it ensures we use the time safely.
        
        # We implement a simple (1+1)-ES style local search.
        # Since the domain is bounded, we clip proposals.
        
        if calls_made < budget and time.time() < deadline:
            sigma = 0.5
            x_cur = best_x.copy()
            f_cur = best_fx
            
            # We will use the remaining budget for local search steps.
            # Each step costs 1 call.
            while calls_made < budget and time.time() < deadline:
                # Generate a perturbation
                step = np.random.randn(dim) * sigma
                x_new = x_cur + step
                x_new = np.clip(x_new, -5.0, 5.0)
                
                # Avoid duplicate calls if possible (though not strictly required, it saves budget)
                # But for simplicity and robustness against "stuck" states, we just call.
                f_new = func(x_new)
                calls_made += 1
                
                if f_new < f_cur:
                    f_cur = f_new
                    x_cur = x_new.copy()
                    best_fx = f_new
                    best_x = x_new.copy()
                    # Increase sigma slightly if making progress?
                    # Or keep it constant. For linear, we don't expect progress.
                    # Let's adapt sigma: if we move, we might be on a slope, increase step?
                    # No, for linear, we are at a corner. Any move inward increases f.
                    # So this loop should mostly fail to improve.
                    
                # Decay sigma if no improvement?
                # Simple adaptation:
                # If we haven't improved in a while, reduce sigma to avoid wasting calls?
                # Or just stop?
                # Since the score is based on the best value found, and we've likely found the global min,
                # these calls don't improve the score but don't hurt it either (n_i stays 0).
                # However, if the gradient estimate was bad and we are not at the true min,
                # this local search might help.
                
                # Let's implement a simple failure counter to stop if stuck.
                # But we don't have a counter here. Let's just run until budget or deadline.
                # Actually, running 400 random walks is wasteful if we are already optimal.
                # But it's safe.
                
                # Optimization: If we are at a vertex and the function is linear, 
                # any move into the interior increases the value.
                # So f_new > f_cur almost always.
                # We can break early if we detect we are stuck?
                # Let's add a simple check: if no improvement in last 50 steps, stop.
                # This requires tracking state.
                
                # Let's refine:
                # If we just found the min at a vertex, we are done.
                # The local search is only a fallback.
                # To be efficient, let's limit the local search to a smaller number of steps,
                # say 50, or stop if no improvement for 20 steps.
                
                # For now, let's just let it run. It's safe.
                
                # Actually, let's add a break condition if we are at a vertex and 
                # the estimated gradient points outwards (which it should if we are at the min).
                # If a_est[i] > 0 and x[i] == -5, moving right increases f.
                # If a_est[i] < 0 and x[i] == 5, moving left increases f.
                # So if we are at the "minimizing" vertex for our estimated gradient,
                # we know we are at the min of the linear model.
                
                # Let's check if best_x is the predicted minimizer.
                pred_min_vertex = np.where(a_est > 0, -5.0, 5.0)
                if np.allclose(best_x, pred_min_vertex):
                    # We are at the predicted minimum. 
                    # For a linear function, this is the global minimum.
                    # We can stop the local search.
                    break

        return best_x, best_fx

# EVOLVE-BLOCK-END
