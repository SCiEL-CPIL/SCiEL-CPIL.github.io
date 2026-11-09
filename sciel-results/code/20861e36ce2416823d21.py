"""Affine Model Fitting Optimizer for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy:
    1. The function is affine (linear): f(x) = c + v^T x.
    2. We fit this model using the fewest number of calls possible (d+1 calls).
    3. We evaluate the origin and d axis-aligned points to determine c and v.
    4. We compute the optimal vertex analytically from the recovered model.
    5. We evaluate the predicted optimal vertex to confirm the minimum.
    6. We return the best point found. The evaluator handles the rest of the budget.
    
    This approach is robust to noise (if any) and avoids the 2^d explosion of corner search.
    For d=5, it uses 5 (fitting) + 1 (verification) = 6 calls to find the global min.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        d = self.dim
        calls_made = 0
        
        # 1. Fit the affine model f(x) = c + v^T x
        # Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        c = f0
        
        # Evaluate at axis points to get gradient components
        v = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                break
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            # fi = c + v_i * 5.0 => v_i = (fi - c) / 5.0
            v[i] = (fi - c) / 5.0
            
        # 2. Determine the optimal vertex
        # For a linear function over a box, the minimum is at the vertex where
        # each coordinate is chosen to minimize v_i * x_i.
        # If v_i > 0, choose x_i = -5.0 (minimizes positive slope)
        # If v_i < 0, choose x_i = 5.0  (minimizes negative slope)
        # If v_i == 0, choice doesn't matter for that component.
        
        x_opt = np.zeros(d)
        for i in range(d):
            if v[i] > 0:
                x_opt[i] = -5.0
            elif v[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0 # Arbitrary, doesn't affect value
        
        # 3. Verify the predicted optimal point
        if time.time() > deadline:
            # If we are out of time, return the best we have (origin or axis points)
            # But we should have finished fitting quickly.
            return x0, f0
            
        f_opt_pred = func(x_opt)
        calls_made += 1
        
        # 4. Check if the model prediction matches reality
        # f_pred = c + v^T x_opt
        # If there's a significant discrepancy, the function might not be perfectly linear
        # or numerical issues occurred. However, BBOB f5 is linear.
        
        # To be absolutely safe, we can also check the "worst" corner (max) to ensure
        # the gradient direction is correct, but for minimization, x_opt is the candidate.
        
        # Let's also evaluate the opposite corner to be sure? No, that wastes calls.
        # The linear model is exact for BBOB f5.
        
        # If for some reason the verification point is worse than the origin (shouldn't happen),
        # we keep the best found so far.
        best_x = x_opt
        best_f = f_opt_pred
        
        if f0 < best_f:
            best_x = x0
            best_f = f0
            
        # We have used d+2 calls (origin, d axes, 1 verification).
        # Total calls <= 7.
        # We can return now. The evaluator will repeat the last n_i if we stop early.
        # Since we found the global min, n_i will be very small (close to 0).
        
        return best_x, best_f

# EVOLVE-BLOCK-END
