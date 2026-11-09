# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive (1+1)-ES with CMA-ES-like covariance adaptation and stall-based restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Parameters for the (1+1)-ES with covariance matrix adaptation
        window_size = 5
        successes = 0
        steps_in_window = 0
        stall_threshold = 30  # Steps without improvement before restarting
        max_sigma = 5.0       # Cap on sigma to prevent divergence
        min_sigma = 1e-5      # Floor on sigma
        
        # Initial state
        x = np.zeros(n)
        fx = wrapped_func(x)
        sigma = 1.5  # Initial step size
        
        # Covariance matrix initialization (identity)
        C = np.eye(n)
        # Step size adaptation parameters
        c_sigma = 1.0 / (n + 4)
        d_sigma = 1.0 + 2 * max(0, (np.sqrt(n) - 1) / (n + 3))
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Correlation matrix (start as identity)
        H = np.eye(n)
        
        # Success tracking for sigma adaptation
        steps_without_improvement = 0
        total_steps = 0
        
        while calls_made[0] < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate using current covariance structure
            # Sample from multivariate normal with covariance sigma^2 * C
            z = rng.standard_normal(n)
            # Transform using Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
                step = sigma * (L @ z)
            except np.linalg.LinAlgError:
                # If Cholesky fails, use diagonal approximation
                diag = np.sqrt(np.maximum(np.diag(C), 1e-10))
                step = sigma * (diag * z)
            
            y = np.clip(x + step, -5, 5)
            fy = wrapped_func(y)
            
            total_steps += 1
            
            if fy < fx:
                # Success: update mean
                x, fx = y, fy
                successes += 1
                steps_without_improvement = 0
                
                # Update covariance matrix using rank-1 update
                # This is a simplified version of CMA-ES update
                step_vec = step
                # Rank-1 update to C
                C = C + c_sigma * (np.outer(step_vec, step_vec) / chi_n - C)
                # Ensure positive semi-definite
                C = (C + C.T) / 2
                # Add small regularization for numerical stability
                C += 1e-10 * np.eye(n)
                
            else:
                steps_without_improvement += 1
                # Slight shrinkage of C on failure
                C = C * 0.999
            
            # Adapt step size based on 1/5 rule
            steps_in_window += 1
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > 1/5:
                    sigma = min(sigma * 1.05, max_sigma)
                else:
                    sigma = max(sigma * 0.95, min_sigma)
                successes = 0
                steps_in_window = 0
            
            # Stall detection: if no improvement for a while, restart
            if steps_without_improvement > stall_threshold:
                # Reset local variables for a new run
                # Try a few different restart strategies
                # 1. Random point in [-3, 3]
                # 2. Random point in [-5, 5]
                # 3. Perturb the best known point
                
                # We'll use a simple random restart in a slightly larger box
                # to allow escape from local minima, but biased towards center
                x = rng.uniform(-3, 3, n)
                fx = wrapped_func(x)
                sigma = 1.5
                steps_without_improvement = 0
                successes = 0
                steps_in_window = 0
                C = np.eye(n)  # Reset covariance

        return best_x, best_f
# EVOLVE-BLOCK-END
