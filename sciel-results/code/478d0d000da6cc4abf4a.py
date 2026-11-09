"""Combined Multi-Start Powell + (1+1)-ES Optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Multi-Start Powell's Method with (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Use 285s limit to be safe within 330s
        self.deadline = self.start_time + 285.0

        def wrapped_func(x):
            # Check limits
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")

            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            # Update global best
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        def es_refine(x_center, remaining_budget):
            """(1+1)-ES refinement around x_center using remaining budget."""
            if remaining_budget < 10:
                return
            if time.time() > self.deadline:
                return
            
            rng = np.random.default_rng(42)
            x = x_center.copy()
            fx = self.best_f
            sigma = 0.5  # Start smaller for local refinement
            
            # Track progress to adapt sigma
            no_improve = 0
            max_no_improve = 20
            
            for _ in range(remaining_budget):
                if self.calls_made >= self.budget:
                    break
                if time.time() > self.deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.15  # Slightly more aggressive growth
                else:
                    no_improve += 1
                    sigma *= 0.9  # Decay sigma on failure
                    
                    # If stuck for too long, try a larger jump
                    if no_improve >= max_no_improve:
                        sigma *= 3.0
                        no_improve = 0
                        max_no_improve = min(max_no_improve + 10, 50)
                    
                # Stop if sigma becomes too small
                if sigma < 1e-7:
                    break

        # Multi-start configuration
        # Use diverse starting points, biased toward likely minimum region
        rng = np.random.default_rng(12345)
        
        starts = []
        # Start near origin (common for transformed Rosenbrock)
        starts.append(np.zeros(self.dim))
        # Start near (1,1,...,1) which is typical Rosenbrock minimum
        starts.append(np.ones(self.dim))
        # Random starts
        starts.append(rng.uniform(-3, 3, self.dim))
        starts.append(rng.uniform(-2, 2, self.dim))
        
        n_starts = len(starts)
        
        # Phase 1: Multi-start Powell's Method
        # Use more budget for Powell, keep 20% for ES
        powell_budget = int(self.budget * 0.8)
        
        for i, x0 in enumerate(starts):
            if self.calls_made >= powell_budget:
                break
            if time.time() > self.deadline:
                break

            # Define callback to stop SciPy if we hit limits
            def callback(xk):
                return self.calls_made >= powell_budget or time.time() > self.deadline

            try:
                # Use Powell's Method with more iterations for better convergence
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell',
                    options={
                        'maxiter': 50,  # More iterations for better convergence
                        'ftol': 1e-10,
                        'xtol': 1e-10,
                    },
                    callback=callback
                )
                
                # Update best if result is valid
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)

            except StopIteration:
                break
            except Exception:
                # If an error occurs, try next start if budget allows
                continue

        # Phase 2: (1+1)-ES refinement around the best point found
        if self.best_x is not None and self.calls_made < self.budget:
            remaining = self.budget - self.calls_made
            es_refine(self.best_x, remaining)

        # If no calls were made (shouldn't happen), return initial guess
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(np.clip(self.best_x, -5, 5))

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
