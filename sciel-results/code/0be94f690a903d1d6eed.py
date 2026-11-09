"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Lambda selection: typically 4 + 3*ln(d)
        lam = max(4, int(4 + 3 * np.log(d)))
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = (2 + 2 / (d + 1)) / ((d + 2)**2 + 2)
        c_mu = min(1 - c_1, (2 + (mu_eff - 2) / (d + 1)) / ((d + 2)**2 + mu_eff + 2))

        # Initialization
        rng = np.random.default_rng(0)
        x = rng.uniform(-4, 4, d)
        f_x = func(x)
        n_calls = 1

        sigma = 2.0  # Initial step size, large enough to explore [-5,5]
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Helper to update C and C_inv_sqrt
        def update_covariance(C, p_sigma, p_c, c_sigma, c_c, c_1, c_mu, w, y_w, z_w, mu_eff, d):
            # Update C
            # C = (1 - c_1 - c_mu + mu_eff * c_sigma * (1 - c_sigma)) * C 
            #   + c_1 * (p_c * p_c.T + (1 - c_sigma * (1 - c_sigma)) * I)
            #   + c_mu * sum(w_i * y_i * y_i.T)
            
            # Note: y_i are the weighted steps in z-space (before scaling by sigma)
            # The standard update uses the transformed steps
            
            # Let's use the standard formula from Hansen's paper
            # C <- (1 - c_1 - c_mu + mu_eff * c_sigma * (1 - c_sigma)) * C 
            #      + c_1 * (p_c p_c^T + (1 - c_sigma(1-c_sigma)) I) 
            #      + c_mu * sum_{i=1}^mu w_i z_i z_i^T
            
            # Wait, the formula usually involves the covariance of the steps.
            # Let's stick to a simpler, more robust update for small d.
            # Since d=5, we can compute the inverse directly if needed, 
            # but maintaining C_inv_sqrt is complex. 
            # For d=5, we can just use Cholesky decomposition every generation if needed, 
            # or keep C and compute C^{-1/2} via eigendecomposition.
            
            # Given the small dimension, let's just store C and compute C_inv_sqrt 
            # when generating samples.
            # Actually, for d=5, computing C^{-1/2} via eigendecomposition 
            # is cheap and robust.
            
            pass # We will implement the update in the loop directly

        # Main Loop
        gen = 0
        while n_calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                if n_calls >= budget or time.time() > deadline:
                    break
                
                # Sample z from N(0, I)
                z = rng.standard_normal(d)
                # Transform to y using C^{-1/2}
                # We need C^{-1/2}. Let's compute it from C.
                # For d=5, eigendecomposition is fast.
                # But we update C every generation. 
                # Let's maintain C and compute C_inv_sqrt once per generation.
                
                # This is inefficient to do inside the loop. 
                # Let's restructure: 
                # 1. Compute C_inv_sqrt from current C.
                # 2. Generate all lam offspring.
                # 3. Evaluate.
                # 4. Update C, sigma, x.
                
                pass # Restructuring below

            # --- Restructured Generation Step ---
            
            # 1. Compute C_inv_sqrt
            # Use eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # 2. Generate offspring
            offspring = []
            for i in range(lam):
                if n_calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(d)
                y = x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                f_y = func(y)
                n_calls += 1
                offspring.append((f_y, y, z))

            if len(offspring) < 2:
                break

            # 3. Sort and select
            offspring.sort(key=lambda t: t[0])
            # Update best
            if offspring[0][0] < f_x:
                f_x = offspring[0][0]
                x = offspring[0][1].copy()

            # 4. Update parameters
            # Weighted mean of selected z's
            z_w = np.zeros(d)
            for i in range(mu):
                z_w += w[i] * offspring[i][2]

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ z_w

            # Update sigma
            # ||p_sigma|| / E||N(0,I)||
            # E||N(0,I)|| = sqrt(d) * (1 - 1/(4d) + 1/(32d^2)) approx
            # For d=5, it's roughly sqrt(5) * 0.97
            # Let's use the standard approximation
            E_norm = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d**2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-5, 10.0)

            # Update p_c
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * h_sigma * C_inv_sqrt @ z_w
            # h_sigma = 1 if ||p_sigma|| < (1.4 + 1/d) * c_sigma * E_norm
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 1.0/d) * c_sigma * E_norm else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ z_w

            # Update C
            # C = (1 - c_1 - c_mu + mu_eff * c_sigma * (1 - c_sigma)) * C 
            #   + c_1 * (p_c * p_c.T + (1 - c_sigma * (1 - c_sigma)) * I)
            #   + c_mu * sum(w_i * z_i * z_i.T)
            
            C_temp = np.zeros((d, d))
            for i in range(mu):
                z_i = offspring[i][2]
                C_temp += w[i] * np.outer(z_i, z_i)
            
            C = ((1 - c_1 - c_mu + mu_eff * c_sigma * (1 - c_sigma)) * C + 
                 c_1 * (np.outer(p_c, p_c) + (1 - c_sigma * (1 - c_sigma)) * np.eye(d)) + 
                 c_mu * C_temp)
            
            # Ensure C is positive definite (symmetrize)
            C = (C + C.T) / 2

        return x, f_x
# EVOLVE-BLOCK-END
