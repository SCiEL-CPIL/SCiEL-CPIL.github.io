import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Evaluate Center
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        update_best(x0, f0)

        # 2. Estimate Gradient
        # Use a large step to get strong signal for sign detection
        step = 3.0
        grad = np.zeros(dim)
        for i in range(dim):
            if calls_made >= budget or time.time() > deadline:
                break
            xi = np.zeros(dim)
            xi[i] = step
            fi = func(xi)
            calls_made += 1
            update_best(xi, fi)
            grad[i] = (fi - f0) / step
            
            # If gradient is very small or ambiguous, check the other direction
            if abs(grad[i]) < 0.1 and calls_made < budget and time.time() < deadline:
                xi_neg = np.zeros(dim)
                xi_neg[i] = -step
                fi_neg = func(xi_neg)
                calls_made += 1
                update_best(xi_neg, fi_neg)
                # If fi_neg < f0, gradient should be negative
                if fi_neg < f0:
                    grad[i] = (f0 - fi_neg) / step  # negative

        # 3. Predict Optimal Vertex
        # For linear min: x_i = -5 if grad[i] > 0, else 5
        x_pred = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                x_pred[i] = -5.0
            else:
                x_pred[i] = 5.0
        
        # Also try the opposite vertex to be safe
        x_pred_alt = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                x_pred_alt[i] = 5.0
            else:
                x_pred_alt[i] = -5.0
        
        if calls_made < budget and time.time() < deadline:
            f_pred = func(x_pred)
            calls_made += 1
            update_best(x_pred, f_pred)
        
        # Evaluate the alternative vertex if we have budget
        if calls_made < budget and time.time() < deadline:
            f_pred_alt = func(x_pred_alt)
            calls_made += 1
            update_best(x_pred_alt, f_pred_alt)
            
        # 4. Verification/Refinement
        # For a truly linear function, the vertex is optimal. But let's verify
        # by checking if we're at a boundary and doing a lightweight refinement.
        
        # Check if best is at a vertex (all coordinates are ±5)
        is_vertex = np.all(np.abs(np.abs(best_x) - 5.0) < 1e-6)
        
        if is_vertex:
            # We found a vertex. For linear functions, this should be optimal.
            # Use remaining budget to verify by sampling a few points near the vertex
            # and confirming they're worse.
            x = best_x
            fx = best_f
            sigma = 0.1
            
            # Do a few verification steps with small perturbations
            max_verify_steps = min(50, budget - calls_made)
            for _ in range(max_verify_steps):
                if calls_made >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 0.5
                else:
                    sigma *= 0.7
                    
                if sigma < 1e-6:
                    break
        else:
            # Not at a vertex - do a more thorough local search
            x = best_x
            fx = best_f
            sigma = 1.0
            
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    
                if sigma < 1e-6:
                    break

        return best_x, best_f
