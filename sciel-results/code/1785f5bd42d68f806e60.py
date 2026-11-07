"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim + 1))
        if lam < 2:
            lam = 2
        # Number of recombination weights
        mu = lam // 2
        
        # Weights for the top mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Equivalent population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Damping for h_sigma
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial step size
        sigma = 2.0
        
        # Initial mean
        # Start at the center of the box
        m = np.zeros(dim)
        
        # Initial covariance matrix
        C = np.eye(dim)
        
        # Cholesky decomposition of C
        try:
            C_root = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_root = np.eye(dim)
            
        # Path states
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        best_x = m
        best_f = func(m)
        calls = 1
        
        if calls >= budget:
            return best_x, best_f
            
        rng = np.random.default_rng(42)
        
        # Precompute expected norm of standard normal
        c_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
        
        gen = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough calls for a full generation
            # If not, we can do a partial generation or stop.
            # Stopping is safer to avoid invalid states.
            # However, if we have some calls left, we can use them for a (1+1) polish.
            # For simplicity, let's just run full generations.
            if calls + lam > budget:
                # Use remaining calls for a simple local search or just break
                # Let's just break to be safe and rely on the best found so far.
                # Or we can reduce lam? No, weights are fixed.
                # Let's just break.
                break
                
            gen += 1
            
            # Generate population
            N = rng.standard_normal((lam, dim))
            Z = N @ C_root.T
            X = m[None, :] + sigma * Z
            X = np.clip(X, -5, 5)
            
            # Evaluate fitness
            F = np.array([func(x) for x in X])
            calls += lam
            
            # Update best
            for i in range(lam):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i]
            
            # Sort indices by fitness
            idx = np.argsort(F)
            
            # Select top mu
            best_idx = idx[:mu]
            Y_best = X[best_idx]
            
            # Update mean
            m_w = np.dot(weights, Y_best)
            
            # Update path sigma
            delta_m = m_w - m
            # Transform to z-space
            # delta_z = C_root.T @ (delta_m / sigma)
            delta_z = C_root.T @ (delta_m / sigma)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_z
            
            # Update h_sigma
            # Check condition: ||p_sigma|| / sqrt(1 - (1 - c_sigma)^(2 * gen)) < (1.4 + 2/(dim+1)) * c_n
            # Approximate denominator
            # For large gen, (1 - c_sigma)^(2 * gen) -> 0
            # So sqrt(1 - ...) -> 1
            # Let's use the exact formula for the first few generations, then approximate.
            # Actually, the term is sqrt(1 - (1 - c_sigma)^(2 * gen))
            # Let's compute it.
            decay = (1 - c_sigma) ** (2 * gen)
            if decay < 1e-10:
                denom = 1.0
            else:
                denom = np.sqrt(1 - decay)
            
            threshold = (1.4 + 2/(dim+1)) * c_n
            
            if np.linalg.norm(p_sigma) / denom < threshold:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_z
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / c_n - 1))
            
            # Update C
            # Term 1: Decay
            # Term 2: Path covariance
            # Term 3: Rank-mu update
            
            # Rank-1 update
            # p_c @ p_c - C
            # Note: p_c is in z-space? 
            # The formula for C update in z-space is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (h_sigma - 1 + c_1) * C) + c_mu * sum(w_i * z_i z_i^T)
            # Where z_i = (y_i - m_w) / sigma
            
            # Let's compute the rank-mu term
            # Z_best = (Y_best - m_w[None, :]) / sigma
            Z_best = (Y_best - m_w[None, :]) / sigma
            
            # Sum of weighted outer products
            # We can do this efficiently:
            # W_diag = np.diag(weights)
            # C_mu = Z_best.T @ W_diag @ Z_best
            
            # This is (dim x mu) @ (mu x mu) @ (mu x dim)
            # Which is (dim x mu) @ (mu x dim)
            # Let's use np.einsum or matrix multiplication.
            # Z_best.T @ (weights[:, None] * Z_best)
            C_mu_term = Z_best.T @ (weights[:, None] * Z_best)
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (h_sigma - 1 + c_1) * C) + c_mu * C_mu_term
            
            # Simplify:
            # C = (1 - c_1 - c_mu + c_1 * (h_sigma - 1 + c_1)) * C + c_1 * (p_c @ p_c) + c_mu * C_mu_term
            # Coeff for C: 1 - c_1 - c_mu + c_1 * h_sigma - c_1 + c_1^2
            # = 1 - 2*c_1 - c_mu + c_1^2 + c_1 * h_sigma
            
            coeff_C = 1 - 2*c_1 - c_mu + c_1**2 + c_1 * h_sigma
            
            C = coeff_C * C + c_1 * np.outer(p_c, p_c) + c_mu * C_mu_term
            
            # Ensure positive definiteness
            # Add small identity if needed
            # Try Cholesky
            try:
                C_root = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add epsilon to diagonal
                C += 1e-8 * np.eye(dim)
                try:
                    C_root = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset to identity if it fails badly
                    C = np.eye(dim)
                    C_root = np.eye(dim)
                    p_c = np.zeros(dim)
                    p_sigma = np.zeros(dim)
                    sigma = 2.0
                    m = m_w # Keep the best mean

        return best_x, best_f
# EVOLVE-BLOCK-END
