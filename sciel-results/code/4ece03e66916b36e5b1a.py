"""
Strategy: Exploit the linearity of f5.
1. Estimate the gradient using central differences (2d+1 calls).
2. Compute the exact step size to project the current point onto the boundary of the box in the direction of the negative gradient.
3. Move to that point.
4. Since f5 is linear, the minimum is on the boundary. This projection should land us at or very near the global minimum.
5. Spend remaining budget on minor refinements or verification to ensure robustness against estimation errors.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        rng = np.random.default_rng(42)
        
        # Start at the center of the box
        x = np.zeros(self.dim)
        fx = func(x)
        calls = 1
        
        # Step 1: Estimate gradient using central differences
        # We need 2 * dim additional calls
        # h should be small but not too small to avoid numerical issues
        h = 1e-4 
        grad = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
                
            x_plus = x.copy()
            x_plus[i] += h
            x_plus = np.clip(x_plus, -5, 5)
            f_plus = func(x_plus)
            calls += 1
            
            x_minus = x.copy()
            x_minus[i] -= h
            x_minus = np.clip(x_minus, -5, 5)
            f_minus = func(x_minus)
            calls += 1
            
            grad[i] = (f_plus - f_minus) / (2 * h)
            
            if calls >= self.budget:
                break
        
        # Step 2: Project onto the boundary in the direction of -grad
        # We want to maximize alpha such that x - alpha * grad is in [-5, 5]^d
        # x_i - alpha * grad_i >= -5  => alpha <= (x_i + 5) / grad_i (if grad_i > 0)
        # x_i - alpha * grad_i <= 5   => alpha >= (x_i - 5) / grad_i (if grad_i > 0)
        # 
        # Actually, we are moving in direction d = -grad.
        # x_new = x + alpha * d = x - alpha * grad
        # Constraints: -5 <= x_i - alpha * grad_i <= 5
        
        if np.any(grad != 0):
            alphas = []
            for i in range(self.dim):
                if grad[i] > 1e-12:
                    # x_i - alpha * grad_i >= -5  => alpha <= (x_i + 5) / grad_i
                    # x_i - alpha * grad_i <= 5   => alpha >= (x_i - 5) / grad_i (this is a lower bound, usually not binding for max step)
                    # We want the largest alpha that keeps us in bounds.
                    # The upper bound on alpha is (x_i + 5) / grad_i
                    alphas.append((x[i] + 5.0) / grad[i])
                elif grad[i] < -1e-12:
                    # x_i - alpha * grad_i >= -5  => alpha >= (x_i + 5) / grad_i (lower bound)
                    # x_i - alpha * grad_i <= 5   => alpha <= (x_i - 5) / grad_i
                    # We want the largest alpha. The upper bound is (x_i - 5) / grad_i
                    # Note: grad[i] is negative, so (x_i - 5) / grad[i] is positive if x_i < 5.
                    alphas.append((x[i] - 5.0) / grad[i])
                # else grad[i] is 0, no constraint from this dimension
            
            if alphas:
                alpha_max = min(alphas)
                # Ensure alpha is non-negative
                if alpha_max > 0:
                    x_new = x - alpha_max * grad
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1
                    
                    if f_new < fx:
                        x, fx = x_new, f_new
                    else:
                        # If projection didn't help, maybe gradient estimate was bad or we're already at min
                        # Try a smaller step or just stay
                        pass
        
        # Step 3: Refine with a simple local search for the remaining budget
        # Since we should be very close to the optimum, we can use a small step size random walk
        # or just verify. But to be safe against gradient estimation errors, we do a few (1+1)-ES steps
        # with a very small sigma.
        
        sigma = 1e-3
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.8
                
            # If we've converged, we can break early to save time (though not strictly necessary)
            # But we must make sure we don't exceed budget.
            
        return x, fx
