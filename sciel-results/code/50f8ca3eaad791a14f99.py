"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for Linear Slope function (BBOB f5).
    Strategy:
    1. Estimate gradient using center and axis points (d+1 calls).
    2. Predict the optimal vertex based on gradient signs.
    3. Evaluate predicted vertex and verify.
    4. If not at a vertex (quadratic component), refine with adaptive search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 1: Gradient estimation from center
        # Evaluate center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        calls_made += 1
        update_best(x_center, f_center)
        
        # Evaluate axis points to estimate gradient
        step = 1.0  # Larger step for better gradient estimation within [-5,5]
        grad_est = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break
            
            xi = np.zeros(dim)
            xi[i] = step
            fi = func(xi)
            calls_made += 1
            update_best(xi, fi)
            
            # Gradient component: (f(x+e_i*step) - f(x)) / step
            grad_est[i] = (fi - f_center) / step
        
        # Phase 2: Predict optimal vertex
        # For linear function: min is at vertex where x_i = -5 if grad[i] > 0, else +5
        x_vertex = np.zeros(dim)
        for i in range(dim):
            if grad_est[i] > 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 5.0
        
        # Evaluate predicted vertex
        if calls_made < budget and time.time() < deadline:
            f_vertex = func(x_vertex)
            calls_made += 1
            update_best(x_vertex, f_vertex)
        
        # Phase 3: Check if we're at the optimum or need refinement
        # For a pure linear function, we should be at the optimum now.
        # If the function has a quadratic component, the optimum might be interior.
        # Do a quick check: evaluate a point slightly inside from the vertex
        
        # If we used all budget or hit deadline, return
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 4: Adaptive refinement
        # If the vertex is the best, try moving slightly inward to check for interior minimum
        # Use a small step inward from the vertex
        
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        
        # (1+1)-ES refinement
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                update_best(x, fx)
                sigma *= 1.15  # Increase step on success
            else:
                sigma *= 0.85  # Decrease step on failure
            
            # Early termination if sigma is very small
            if sigma < 1e-8:
                break
            
            # If we've converged near a vertex and function is linear, we can stop
            # Check if we're very close to a vertex
            dist_to_vertex = 0
            for i in range(dim):
                if x[i] > 4.99:
                    dist_to_vertex += abs(x[i] - 5.0)
                elif x[i] < -4.99:
                    dist_to_vertex += abs(x[i] + 5.0)
                else:
                    dist_to_vertex += min(abs(x[i] - 5.0), abs(x[i] + 5.0))
            
            if dist_to_vertex < 0.01 and sigma < 0.05:
                # We're very close to a vertex and step is small
                # For linear function, this is likely the optimum
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
