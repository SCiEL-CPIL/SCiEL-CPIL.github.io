"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Initial random sampling to estimate the linear model parameters (a, b).
    2. Use the estimated gradient to take large, informed steps towards the minimizer.
    3. Refine with local search.
    4. Deadline guard to ensure we finish within 300s (with margin).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        # Phase 1: Estimate linear model
        # f(x) ≈ a^T x + b
        # We need enough points to estimate 'a' and 'b'.
        # Let's use 2 * dim + 1 points for stability.
        n_samples = 2 * self.dim + 1
        X = np.random.default_rng(42).uniform(-4, 4, (n_samples, self.dim))
        Y = np.array([func(x) for x in X])
        
        # Add bias term for intercept
        X_bias = np.hstack([X, np.ones((n_samples, 1))])
        
        # Solve least squares: X_bias @ [a, b] = Y
        try:
            params, _, _, _ = np.linalg.lstsq(X_bias, Y, rcond=None)
            a = params[:self.dim]
            b = params[self.dim]
        except:
            # Fallback to random search if estimation fails
            return self._fallback(func, start_time, deadline)

        # Phase 2: Analytical minimization of the linear model
        # Minimize a^T x + b subject to -5 <= x_i <= 5
        # The minimum is at x_i = -5 if a_i > 0, and x_i = 5 if a_i < 0.
        x_best = np.zeros(self.dim)
        for i in range(self.dim):
            if a[i] > 0:
                x_best[i] = -5.0
            else:
                x_best[i] = 5.0
        
        f_best = func(x_best)
        
        # Phase 3: Local refinement using (1+1)-ES around x_best
        # The linear model might not be perfect, or the true minimum might be slightly off-boundary
        # due to estimation errors or non-linearities (though f5 is linear).
        # However, the "hidden shift, rotation and offset" implies the function is:
        # f(x) = offset + slope * (R @ (x - shift))
        # This is still linear in x. f(x) = c^T x + d.
        # So the analytical solution should be very close.
        # Let's do a quick local search to ensure we don't miss a slightly better point
        # if the estimation was noisy.
        
        sigma = 0.5
        x_current = x_best.copy()
        f_current = f_best
        
        calls_made = n_samples + 1
        budget_left = self.budget - calls_made
        
        # Simple local search
        for _ in range(min(budget_left, 100)): # Limit refinement steps to be safe
            if time.time() > deadline:
                break
            
            y = np.clip(x_current + sigma * np.random.default_rng().standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < f_current:
                x_current = y
                f_current = fy
                sigma *= 1.1
            else:
                sigma *= 0.9
                
            if sigma < 1e-4:
                break
        
        return x_current, f_current

    def _fallback(self, func, start_time, deadline):
        # Fallback (1+1)-ES if model estimation fails
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        calls = 1
        while calls < self.budget:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
        return x, fx
# EVOLVE-BLOCK-END
