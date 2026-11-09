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
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Main Loop
        gen = 0
        while n_calls < budget:
            if time.time() > deadline:
                break

            # Compute C_inv_sqrt via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Generate offspring
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

            # Sort and select
            offspring.sort(key=lambda t: t[0])
            # Update best
            if offspring[0][0] < f_x:
                f_x = offspring[0][0]
                x = offspring[0][1].copy()

            # Update parameters
            # Weighted mean of selected z's
            z_w = np.zeros(d)
            for i in range(mu):
                z_w += w[i] * offspring[i][2]

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ z_w

            # Update sigma
            E_norm = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d**2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-5, 10.0)

            # Update p_c
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 1.0/d) * c_sigma * E_norm else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ z_w

            # Update C
            C_temp = np.zeros((d, d))
            for i in range(mu):
                z_i = offspring[i][2]
                C_temp += w[i] * np.outer(z_i, z_i)
            
            C = ((1 - c_1 - c_mu + mu_eff * c_sigma * (1 - c_sigma)) * C + 
                 c_1 * (np.outer(p_c, p_c) + (1 - c_sigma * (1 - c_sigma)) * np.eye(d)) + 
                 c_mu * C_temp)
            
            # Ensure C is positive definite (symmetrize)
            C = (C + C.T) / 2

            gen += 1

        return x, f_x
# EVOLVE-BLOCK-END
