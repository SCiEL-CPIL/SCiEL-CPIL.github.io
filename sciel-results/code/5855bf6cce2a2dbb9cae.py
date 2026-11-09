"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for Linear Slope function.
    Strategy:
    1. Estimate gradient using center and axis points (single-sided).
    2. Predict the optimal vertex based on gradient signs.
    3. Evaluate predicted vertex.
    4. Refine with (1+1)-ES if necessary.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        rng = np.random.default_rng(42)
        d = self.dim
        calls_made = 0
        
        # Phase 1: Estimate gradient using single-sided differences
        # Evaluate center point
        center = np.zeros(d)
        f_center = func(center)
        calls_made += 1
        
        # Evaluate points offset in each positive direction
        # For linear function f(x) = a^T x + b, f(center + h*e_i) - f(center) = a_i * h
        # So gradient component a_i = (f(center + h*e_i) - f(center)) / h
        h = 1.0  # Small step for gradient estimation
        grad_est = np.zeros(d)
        
        for i in range(d):
            if calls_made >= self.budget or time.time() > deadline:
                break
            point = center.copy()
            point[i] += h
            point = np.clip(point, -5, 5)
            f_point = func(point)
            calls_made += 1
            grad_est[i] = (f_point - f_center) / h
        
        # For a linear function, the minimum is at the vertex where
        # x_i = -5 if grad[i] > 0 (decreasing in that direction)
        # x_i =  5 if grad[i] < 0 (increasing in that direction)
        # x_i =  0 if grad[i] == 0 (doesn't matter)
        
        # Determine optimal vertex
        optimal_vertex = np.zeros(d)
        for i in range(d):
            if grad_est[i] > 0:
                optimal_vertex[i] = -5.0  # Move in negative direction
            elif grad_est[i] < 0:
                optimal_vertex[i] = 5.0   # Move in positive direction
            else:
                optimal_vertex[i] = 0.0   # Neutral
        
        # Evaluate the predicted optimal vertex
        f_opt = func(optimal_vertex)
        calls_made += 1
        
        best_x = optimal_vertex.copy()
        best_f = f_opt
        
        # Phase 2: Verify with a few nearby points and refine with (1+1)-ES
        # This handles cases where the function might not be perfectly linear
        # or where numerical issues affect gradient estimation
        
        # Try a few random points near the optimal vertex to verify
        for _ in range(min(5, max(0, self.budget - calls_made))):
            if time.time() > deadline:
                break
            y = np.clip(best_x + 0.5 * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
        
        # Initialize ES parameters
        x = best_x.copy()
        fx = best_f
        sigma = 0.3  # Start with moderate sigma
        
        # Run (1+1)-ES for remaining budget
        while calls_made < self.budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15  # Increase step size on success
            else:
                sigma *= 0.85  # Decrease step size on failure
                
            # Safety break if sigma becomes too small
            if sigma < 1e-6:
                break

        return x, fx
# EVOLVE-BLOCK-END
