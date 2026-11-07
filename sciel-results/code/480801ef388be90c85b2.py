"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free trust-region method with local quadratic surrogates."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin

        n = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.zeros(n)
        best_f = np.inf
        total_calls = 0

        # Initial point
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        total_calls += 1
        best_x, best_f = x.copy(), fx

        # Trust-region radius
        sigma = 2.0
        min_sigma = 1e-8
        max_sigma = 10.0

        # Build local quadratic model using coordinate and pairwise directions
        # We use a fixed set of 2n + n(n-1)/2 directions for full quadratic
        # For n=5: 10 + 10 = 20 directions, so 21 evaluations per model (center + 20)
        
        # For efficiency with budget 500, we'll use a subset of directions
        # Start with coordinate directions only (2n = 10 directions)
        # Then expand to include pairwise if budget allows

        def build_quadratic_model(center, fc, sigma_val):
            """Build quadratic model using finite differences."""
            nonlocal total_calls
            
            m = n  # number of directions to use
            # Use coordinate directions first
            directions = []
            for i in range(n):
                d = np.zeros(n)
                d[i] = 1.0
                directions.append(d)
            
            # If budget allows, add pairwise directions
            remaining = budget - total_calls
            if remaining > 2 * m + 10:
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.zeros(n)
                        d[i] = 1.0
                        d[j] = 1.0
                        directions.append(d)
                m = len(directions)
            
            # Limit number of directions based on remaining budget
            max_dir = min(m, max(2, (remaining - 20) // 1))
            directions = directions[:max_dir]
            m = len(directions)
            
            # Evaluate at center + m directions
            fvals = np.zeros(m)
            for k in range(m):
                x_trial = np.clip(center + sigma_val * directions[k], -5, 5)
                fvals[k] = func(x_trial)
                total_calls += 1
                if fvals[k] < best_f:
                    best_f = fvals[k]
                    best_x = x_trial.copy()
            
            # Build quadratic model: f(x) ≈ fc + g^T (x-center)/sigma + 0.5 * (x-center)^T H (x-center)/sigma^2
            # Using the directional derivatives
            # For direction d_k: f(center + sigma*d_k) - fc ≈ sigma * (g^T d_k) + 0.5 * sigma^2 * (d_k^T H d_k)
            
            # Solve for gradient and Hessian using least squares
            # A matrix where each row corresponds to a direction
            # We use the form: f_k - fc = sigma * g^T d_k + 0.5 * sigma^2 * d_k^T H d_k
            
            # Simplified: assume H is diagonal initially, then refine
            # For now, use only gradient estimation with central differences approximation
            # f(center + sigma*d) - fc ≈ sigma * g^T d + 0.5 * sigma^2 * d^T H d
            
            # Estimate gradient using forward differences
            grad = np.zeros(n)
            for k in range(min(n, m)):
                d = directions[k]
                # f(center + sigma*d) - fc ≈ sigma * g^T d + 0.5 * sigma^2 * d^T H d
                # Approximate as sigma * g^T d for gradient estimation
                grad += (fvals[k] - fc) / sigma * d / max(1, np.sum(d**2))
            
            # Normalize gradient estimate
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm * min(grad_norm, 1.0)
            
            return grad, fvals, directions

        # Main loop
        stagnation_count = 0
        max_stagnation = 30

        while total_calls < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Check for stagnation
            if stagnation_count > max_stagnation:
                # Restart with new random point
                x = rng.uniform(-5, 5, n)
                fx = func(x)
                total_calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 2.0
                stagnation_count = 0
                continue
            
            # Build local model
            if budget - total_calls < 5:
                break
            
            grad, fvals, directions = build_quadratic_model(x, fx, sigma)
            
            # Determine step direction
            # Move opposite to gradient
            if np.linalg.norm(grad) > 1e-12:
                step_dir = -grad
            else:
                step_dir = rng.standard_normal(n)
                step_dir /= np.linalg.norm(step_dir)
            
            # Try step
            x_trial = np.clip(x + sigma * step_dir, -5, 5)
            f_trial = func(x_trial)
            total_calls += 1
            
            if f_trial < fx:
                # Success
                x, fx = x_trial, f_trial
                sigma *= 1.3
                stagnation_count = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure
                sigma *= 0.7
                stagnation_count += 1
            
            # If sigma too small, try larger random perturbation
            if sigma < min_sigma:
                x = np.clip(x + rng.uniform(-1, 1, n), -5, 5)
                fx = func(x)
                total_calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 1.0
                stagnation_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
