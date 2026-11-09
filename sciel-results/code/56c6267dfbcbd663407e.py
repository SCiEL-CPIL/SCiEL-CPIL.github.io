"""Rewritten optimizer for BBOB f6 (Attractive sector) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Continuous optimizer using Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lambda_pop = int(4 + 3 * np.log(dim))
        # Learning rates
        mu = lambda_pop // 2  # Number of best points used for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Initial state
        # Start at the center of the box, or a random point if we want to be more general, 
        # but for BBOB f6, starting near the center is often good, or random.
        # The task says minimizer is in [-4, 4]^5.
        # Let's start at 0,0,0,0,0 as a robust default for BBOB, or random?
        # Random is safer against specific shifts if they are large, but 0 is the origin.
        # Given "hidden seeded shift", the minimum could be anywhere in [-4,4].
        # A random start in [-2, 2] might be better to avoid bias to 0 if shift is large.
        # However, CMA-ES is quite robust. Let's use a small random start.
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-2, 2, dim)
        sigma = 2.0 # Initial step size
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls_made = 0
        x_best = x_mean.copy()
        f_best = np.inf

        def eval_func(x):
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
            return f

        # Evaluate initial mean
        f_mean = eval_func(x_mean)
        
        # CMA-ES Main Loop
        while calls_made < budget:
            # Check if we have enough budget for a generation
            if calls_made + lambda_pop > budget:
                # If we can't fit a full population, we might do a final local polish
                # or just stop. CMA-ES works best with full populations.
                # Let's just break to avoid wasting calls on incomplete updates, 
                # or we can reduce lambda_pop for the last iteration.
                # For simplicity and robustness, we break if we can't do a full gen.
                break

            # Sample population
            Y = np.zeros((lambda_pop, dim))
            X = np.zeros((lambda_pop, dim))
            F = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                # Sample z ~ N(0, I)
                z = rng.standard_normal(dim)
                # Transform to x
                y = B @ (D * z)
                x_new = x_mean + sigma * y
                
                # Boundary handling: Reflection
                # If x_new is outside [-5, 5], reflect it
                # This is a common strategy to keep the distribution valid
                for j in range(dim):
                    if x_new[j] < -5:
                        x_new[j] = -5 + (-5 - x_new[j]) * 0.5 # Reflect with damping
                        # Or simple reflection: x_new[j] = -5 + (-5 - x_new[j])
                        # Damped reflection helps avoid sticking to boundary
                        x_new[j] = np.clip(x_new[j], -5, 5)
                    elif x_new[j] > 5:
                        x_new[j] = 5 - (x_new[j] - 5) * 0.5
                        x_new[j] = np.clip(x_new[j], -5, 5)
                
                Y[i] = z
                X[i] = x_new
                F[i] = eval_func(x_new)
                
                if calls_made >= budget:
                    break
            
            if calls_made >= budget:
                break

            # Sort by fitness
            idx = np.argsort(F)
            F = F[idx]
            X = X[idx]
            Y = Y[idx]

            # Update mean
            # Weighted sum of best mu points
            x_new_mean = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
            
            # Update p_sigma
            # h_sigma: Heaviside function to check if p_sigma is too small
            # Simplified: just update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * (np.sum(weights[:, np.newaxis] * Y[:mu], axis=0))))
            
            # Update p_c
            h_sigma = 1.0 # Simplified, usually calculated via norm
            # More accurate h_sigma:
            # chi_n is expected norm of N(0, I)
            # norm of p_sigma / (sigma * sqrt(1 - (1-c_sigma)^2))
            # Let's use a simpler check: if norm(p_sigma) < (1.4 + 1/dim) * chi_n
            # But we need to scale p_sigma by 1/sigma? No, p_sigma is in x-space?
            # Standard CMA-ES: p_sigma is in x-space? No, p_sigma is in y-space (standard normal space) scaled by sigma?
            # Actually, p_sigma is maintained in the x-space but updated with y-space samples?
            # Let's stick to the standard formulation:
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (1/sigma) * (x_new_mean - x_mean)
            # Wait, the standard update is:
            # p_sigma = (1-c_sigma)*p_sigma + h_sigma * sqrt(c_sigma*(2-c_sigma)*mu_eff) * (B * D * (y_bar))
            # where y_bar is the weighted average of the y-samples.
            # My code above: `B @ (D * (np.sum(...)))` is exactly B*D*y_bar.
            # So p_sigma is in the "transformed" space? No, p_sigma is usually in x-space?
            # Let's check the norm for h_sigma.
            # norm(p_sigma) / (sigma * chi_n)
            # If norm(p_sigma) < (1.4 + 1/dim) * sigma * chi_n, h_sigma = 1, else 0.
            # This prevents adaptation when the step size is too small or noise is high.
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 1.0/dim) * sigma * chi_n else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Update Covariance Matrix C
            # Rank-mu update
            v = (x_new_mean - x_mean) / sigma
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(weights_i * (v_i - v_bar) @ (v_i - v_bar))
            
            # Let's use the more stable form:
            # C = (1 - c_1 - c_mu) * C
            # C += c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # For rank-mu update:
            # We need the differences v_i = (x_i - x_old_mean) / sigma
            # Actually, standard CMA-ES uses v_i = (x_i - x_mean_old) / sigma
            # Then C += c_mu * sum(w_i * v_i @ v_i)
            # But this can be numerically unstable. 
            # Let's use the decomposition update:
            # B, D = eig(C)
            # Update C via B, D
            
            # Recompute eigendecomposition every step for simplicity and stability
            # This is O(N^3) but N=5, so it's very fast.
            D_old = D
            B_old = B
            C_old = C
            
            # Update C using the formula
            # C = (1 - c_1 - c_mu) * C_old
            C = (1 - c_1 - c_mu) * C_old
            
            # Add rank-1 term
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C_old)
            
            # Add rank-mu term
            # v_i = (x_i - x_mean_old) / sigma
            # Note: x_mean_old is the mean BEFORE this generation's update.
            # We have X sorted. The first mu are the best.
            # We need the y-samples corresponding to these?
            # Actually, the rank-mu update uses the vectors v_i = (x_i - x_mean_old) / sigma
            # Let's compute these.
            # x_mean_old is the x_mean from the start of the loop.
            
            # Wait, I need to store x_mean_old before updating x_mean.
            # I updated x_new_mean but haven't assigned it to x_mean yet.
            # So x_mean is still x_mean_old.
            
            for i in range(mu):
                v_i = (X[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(v_i, v_i)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0
            
            # Eigendecomposition
            # Since C is symmetric positive semi-definite, use eigh
            D_sq, B = np.linalg.eigh(C)
            # Clip small negative values
            D_sq = np.clip(D_sq, 1e-10, None)
            D = np.sqrt(D_sq)
            
            # Update step size sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (norm(p_sigma) / chi_n - 1))
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1.0))
            
            # Update mean
            x_mean = x_new_mean
            
            # Convergence check
            # If the range of f is very small, or sigma is very small
            if sigma < 1e-10:
                break
            if np.max(F[:mu]) - np.min(F[:mu]) < 1e-10:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
