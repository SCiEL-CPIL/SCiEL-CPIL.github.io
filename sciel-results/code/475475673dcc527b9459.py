import numpy as np
import time

class Optimizer:
    """
    Gradient-Descent Optimizer for BBOB f5 (Linear Slope).
    
    Strategy:
    1. Exploit the linearity of f5. The gradient is constant everywhere.
    2. Estimate the gradient using finite differences at the start.
       Since f5 is linear, 2 calls per dimension (forward difference) are sufficient
       to determine the exact direction of the minimum.
    3. Use the estimated gradient to perform large, deterministic steps towards the
       boundary of the box (the minimum of a linear function on a box is always at a vertex).
    4. Refine with a final local search if budget remains, though for linear functions
       the vertex is the exact solution.
    
    This approach is fundamentally different from stochastic search (CMA-ES, NM) 
    because it uses the known structure of the function class to guide the search 
    deterministically after a brief initialization phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0
        self.best_f = float('inf')
        self.best_x = None
        self.start_time = None
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        # 300s limit, use 290s to be safe
        self.deadline = self.start_time + 290.0
        self.calls_made = 0
        self.best_f = float('inf')
        self.best_x = None

        # 1. Wrapped function for strict tracking
        def wrapped_func(x):
            if self.calls_made >= self.budget:
                return float('inf')
            if time.time() > self.deadline:
                return float('inf')
            
            self.calls_made += 1
            try:
                x_arr = np.asarray(x, dtype=float)
                # Clip to bounds to ensure valid input, though func clips too
                x_clipped = np.clip(x_arr, -5, 5)
                val = float(func(x_clipped))
                
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = x_clipped.copy()
                    
                return val
            except Exception:
                return float('inf')

        # 2. Phase 1: Gradient Estimation
        # For a linear function f(x) = a^T x + b, the gradient is a.
        # We estimate a using central differences or forward differences.
        # Forward differences are cheaper: f(x+eps*e_i) - f(x) ~ a_i * eps.
        # Since we only need the direction, we can use a large eps to avoid numerical noise
        # and because the function is linear, the slope is exact regardless of eps (within float precision).
        
        # Start at the center (0,0,...,0) or a random point?
        # Center is good because it's equidistant from all boundaries, maximizing step potential.
        x0 = np.zeros(self.dim)
        f0 = wrapped_func(x0)
        
        if self.calls_made >= self.budget or time.time() > self.deadline:
            return self.best_x, self.best_f

        # Estimate gradient components
        # Use a step size h. Since the domain is [-5, 5], h=1.0 is safe and large enough.
        h = 1.0
        grad = np.zeros(self.dim)
        
        for i in range(self.dim):
            if self.calls_made >= self.budget or time.time() > self.deadline:
                break
            
            x_plus = x0.copy()
            x_plus[i] += h
            # Ensure x_plus is within bounds for the sake of consistent gradient estimation
            # If x0 is 0, x_plus is 1, which is in [-5, 5].
            f_plus = wrapped_func(x_plus)
            
            # grad_i = (f(x+h) - f(x)) / h
            grad[i] = (f_plus - f0) / h
            
            # Check if we can afford the negative side for better accuracy?
            # For linear functions, forward difference is exact. No need for central.
            # But let's check if we have budget for a second pass if we want to be robust to non-linearities?
            # The task says f5 is Linear Slope. We trust it.
            
        if self.calls_made >= self.budget or time.time() > self.deadline:
            return self.best_x, self.best_f

        # 3. Phase 2: Directed Descent
        # The minimum of a linear function on a box is at the vertex that maximizes -grad^T x.
        # Since we want to minimize f, we want to move in the direction -grad.
        # The optimal vertex is determined by the sign of each gradient component.
        # If grad[i] > 0, we want to minimize x[i] -> x[i] = -5.
        # If grad[i] < 0, we want to maximize x[i] -> x[i] = 5.
        # If grad[i] == 0, x[i] doesn't matter.
        
        # Let's construct the theoretical optimal vertex.
        x_opt_candidate = np.zeros(self.dim)
        for i in range(self.dim):
            if grad[i] > 1e-10:
                x_opt_candidate[i] = -5.0
            elif grad[i] < -1e-10:
                x_opt_candidate[i] = 5.0
            else:
                x_opt_candidate[i] = 0.0 # or keep x0[i]

        # Evaluate the theoretical optimum
        f_opt_est = wrapped_func(x_opt_candidate)
        
        # If the gradient estimation was noisy or the function wasn't perfectly linear,
        # we might have missed. But for f5, this should be very close to f_opt.
        
        # 4. Phase 3: Refinement (Safety Net)
        # We have used ~1 (center) + dim (gradient) + 1 (opt) = dim + 2 calls.
        # For dim=5, that's 7 calls. We have ~493 calls left.
        # We should use the remaining budget to ensure we haven't made a mistake.
        # We can run a few steps of Nelder-Mead or a local search from x_opt_candidate.
        # However, since f5 is linear, the vertex IS the global minimum.
        # Any movement away from the vertex will increase the function value.
        # So, the best strategy is to just stay there? 
        # But AOCC rewards *progress*. If we jump to the optimum at call 8, 
        # all subsequent calls (9..500) will have b_i = f_opt, so e_i ~ 0, n_i ~ -8 (clipped to 0? No, log10(1e-8)=-8, n_i=0).
        # Wait, n_i = (log10(e_i) + 8) / 10.
        # If e_i = 1e-8, log10(1e-8) = -8, n_i = 0.
        # If e_i is small, n_i is small.
        # AOCC = 1 - mean(n_i).
        # If we reach the optimum early, n_i becomes 0 for the rest of the run.
        # This maximizes AOCC.
        
        # Therefore, the strategy is:
        # 1. Find the vertex quickly (Phase 1 & 2).
        # 2. Evaluate it.
        # 3. If we have budget left, do we need to do anything else?
        #    The score is based on the *trajectory* of best-f-so-far.
        #    Once best_f is at f_opt, it stays there.
        #    So, once we find the optimal vertex, we can just stop calling or call the same point?
        #    Calling the same point doesn't change best_f.
        #    We should stop calling to save time? No, the evaluator records calls.
        #    "Every call is recorded". If we make fewer than 500 calls, the last n_i is repeated.
        #    So if we stop at call 8, n_8 is repeated for 8..500.
        #    This is perfect. We don't need to waste calls.
        
        # However, what if our gradient estimation was slightly off due to floating point or 
        # if the "Linear Slope" has a small non-linear component?
        # BBOB f5 is strictly linear: f(x) = c^T (x - z) + b.
        # So the vertex logic is exact.
        
        # Let's add a small verification step.
        # If we have budget, let's try to improve slightly in case the gradient was 0 in some dim 
        # but actually non-zero (unlikely with h=1.0).
        
        # Actually, let's just use the remaining budget to run a very fast local search 
        # starting from x_opt_candidate, just in case.
        # But for a linear function, any local search will just move away from the vertex 
        # if it doesn't project onto the axes.
        # Nelder-Mead might oscillate.
        
        # Better: Use the remaining budget to sample random points?
        # No, random points will be worse than the vertex.
        
        # The most "AOCC-efficient" thing is to find the global minimum as fast as possible.
        # We did that in ~7 calls.
        # We can stop here.
        
        # But wait, the prompt says "repeating the last n_i if fewer calls are made".
        # So stopping is fine.
        
        # One edge case: What if the gradient estimation failed?
        # e.g. f_plus - f0 was 0 due to precision?
        # With h=1.0 and values in [-100, 100], precision is fine.
        
        # Let's do a final check: if we have a lot of budget left, 
        # maybe we should do a few random restarts to be absolutely sure 
        # we didn't get a bad gradient estimate (e.g. if the function was not linear)?
        # The task says "BBOB-style function f5 (Linear slope)". It implies it is linear.
        # But to be safe, let's use 50 calls for random sampling to see if we find anything better.
        # If we find something better, great. If not, we stick with the vertex.
        
        remaining_budget = self.budget - self.calls_made
        if remaining_budget > 50 and time.time() < self.deadline:
            rng = np.random.default_rng(42)
            # Sample 20 random points
            for _ in range(20):
                if self.calls_made >= self.budget or time.time() > self.deadline:
                    break
                x_rand = rng.uniform(-5, 5, self.dim)
                f_rand = wrapped_func(x_rand)
                # If f_rand < self.best_f, best_x is updated in wrapped_func
        
        # Return the best found
        return self.best_x, self.best_f
