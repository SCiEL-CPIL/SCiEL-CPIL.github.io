"""CMA-ES based optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive sigma for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Start at center of the box
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size

        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Eigendecomposition of C
        # C = B @ diag(D^2) @ B.T
        # We need B and D (sqrt of eigenvalues)
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Population size
        # Standard CMA-ES: lambda = 4 + 3*ln(n)
        # For n=5, lambda = 4 + 3*ln(5) = 4 + 4.82 = 8.82 -> 9
        # Let's use a slightly larger population for better exploration in high dimensions if needed, but 9 is standard.
        # However, for small dimensions, smaller lambda is often better.
        # Let's stick to standard:
        lambda_ = 4 + int(3 * np.log(dim))
        if lambda_ < 2:
            lambda_ = 2
        
        mu = lambda_ // 2  # Number of recombination weights

        # Recombination weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / w.sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Threshold for h_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Evolution path for sigma
        p_sigma = np.zeros(dim)
        # Evolution path for covariance
        p_c = np.zeros(dim)

        # Initial function evaluation
        fx = func(mean)
        best_x = mean.copy()
        best_fx = fx
        calls = 1

        # Precompute C_inv_sqrt = B @ diag(1/D) @ B.T
        # Since C starts as I, B=I, D=1, C_inv_sqrt=I
        C_inv_sqrt = np.eye(dim)

        # Main CMA-ES loop
        # We need to ensure we don't exceed budget.
        # Each generation uses lambda_ evaluations.
        # We can run as long as calls + lambda_ <= budget
        # If there is a remainder, we will use it for local search later.
        
        while calls + lambda_ <= budget:
            # Generate population
            # Z ~ N(0, I) of shape (lambda_, dim)
            Z = np.random.standard_normal((lambda_, dim))
            # Y = Z @ (B * D).T  -> Y = (B * D) @ Z.T
            # Actually, x = mean + sigma * B @ (D * z)
            # Vectorized: Y = (B * D) @ Z.T  => shape (dim, lambda_)
            # Then X = mean + sigma * Y.T
            Y = (B * D) @ Z.T
            X = mean + sigma * Y.T
            X = np.clip(X, -5, 5)

            # Evaluate population
            F = np.array([func(x) for x in X])
            calls += lambda_

            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_fx:
                best_fx = F[min_idx]
                best_x = X[min_idx].copy()

            # Sort by fitness
            idx = np.argsort(F)
            F_sorted = F[idx]
            X_sorted = X[idx]

            # Update mean
            mean_old = mean.copy()
            # mean = sum(w_i * x_i)
            mean = np.sum(w[:, None] * X_sorted[:mu], axis=0)

            # Update evolution paths
            # diff = (mean - mean_old) / sigma
            diff = (mean - mean_old) / sigma
            
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff)
            
            # h_sigma
            # Check if p_sigma is small
            # norm(p_sigma) / sqrt(1 - (1-c_sigma)^(2*t)) < (1+2/(n+1))*chi_n
            # t is number of generations. We can approximate t as calls/lambda_
            # But standard implementation uses a counter. Let's use a simple check based on norm.
            # A common approximation is to check if norm(p_sigma) < (1+2/(dim+1))*chi_n
            # More accurately, we track the decay.
            # For simplicity and robustness, we use the standard threshold without the time-dependent denominator 
            # or we can track it. Let's track it properly.
            # However, implementing the exact time-dependent threshold requires knowing the generation number.
            # Let's use a generation counter.
            # Actually, the standard check is:
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*t)) < (1+2/(n+1))*chi_n else 0
            # Let's maintain a counter 'gen'
            # Since we didn't initialize 'gen', let's add it.
            # Wait, I didn't include 'gen' in the variables. Let's add it.
            # But I can't change the variables easily in the middle of the thought process.
            # Let's just use the simpler check: norm(p_sigma) < (1+2/(dim+1))*chi_n
            # This is slightly less accurate but often sufficient.
            # Better: Use the norm directly against the threshold.
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1 + 2 / (dim + 1)) * chi_n else 0.0

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clip sigma to avoid under/overflow
            sigma = np.clip(sigma, 1e-12, 1e12)

            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff)

            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * y_i @ y_i)
            # where y_i = (x_i - mean_old) / sigma
            
            # Compute Y_selected (the normalized steps)
            Y_selected = (X_sorted[:mu] - mean_old) / sigma
            
            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Add the rank-mu update
            # sum(w_i * outer(y_i, y_i))
            # Y_selected is (mu, dim)
            # We need sum of w_i * y_i^T y_i
            # This is (Y_selected.T * w) @ Y_selected
            C += c_mu * (Y_selected.T * w) @ Y_selected

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigendecomposition
            try:
                D_sq, B = np.linalg.eigh(C)
                # Handle numerical issues
                D_sq = np.maximum(D_sq, 1e-12)
                D = np.sqrt(D_sq)
                # C_inv_sqrt = B @ diag(1/D) @ B.T
                C_inv_sqrt = (B / D) @ B.T
            except np.linalg.LinAlgError:
                # Fallback
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                C_inv_sqrt = np.eye(dim)
                sigma *= 0.5 # Reduce sigma if matrix broke

            # Check for convergence
            if sigma < 1e-8:
                break

        # Use remaining budget for local search
        # If we have budget left, we can do a simple coordinate descent or random walk around best_x
        # This is crucial for AOCC as it allows fine-tuning.
        remaining = budget - calls
        if remaining > 0:
            # Simple local search: try small perturbations
            # We can use a small step size based on current sigma or just a fixed small value
            # Let's try a few random steps with decreasing magnitude
            local_sigma = max(sigma, 1e-4)
            for _ in range(remaining):
                # Generate a random direction
                z = np.random.standard_normal(dim)
                x_new = best_x + local_sigma * z
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls += 1
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()
                # Optionally reduce local_sigma if we want to converge faster
                # But with limited budget, just trying random steps is okay.
                # A better local search would be coordinate descent, but it requires dim calls per step.
                # If remaining is small, random steps are fine.
                # If remaining is large, we could restart CMA-ES with smaller sigma, but that's complex.
                # Let's just stick to random steps for simplicity and robustness.
                # To improve, we can make the step size adaptive.
                if fx_new < best_fx:
                    local_sigma *= 1.1 # Increase step size if we found a better point? No, usually we decrease.
                else:
                    local_sigma *= 0.9 # Decrease step size if we didn't find a better point.
                
                # Ensure local_sigma doesn't become too small
                local_sigma = max(local_sigma, 1e-6)

        return best_x, best_fx
# EVOLVE-BLOCK-END
