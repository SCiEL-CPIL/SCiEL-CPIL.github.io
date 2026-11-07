"""Optimized program for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-stage optimizer for Büche-Rastrigin:
    1. Dense stratified Sobol search over first ~80 calls to find a promising basin.
    2. CMA-ES for the remaining budget for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_f = float('inf')
        evals_used = 0

        # --- Phase 1: Stratified Sobol Search ---
        # Use Sobol sequence for better space-filling than uniform random
        n_sobol = 64  # 2^6 points, good for 5D
        if budget < n_sobol + 20:
            n_sobol = max(1, budget // 4)

        # Generate Sobol points in [0,1]^d
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # n_sobol must be power of 2 for Sobol
        n_sobol = 2 ** int(np.ceil(np.log2(max(n_sobol, 2))))
        if n_sobol > budget - 10:
            n_sobol = 2 ** int(np.floor(np.log2(max(budget - 10, 2))))
        
        sobol_points = sampler.random(n_sobol)  # in [0,1)^d
        # Map to [-5, 5]^d
        sobol_points = sobol_points * 10.0 - 5.0

        # Evaluate Sobol points
        for i in range(n_sobol):
            if time.time() > deadline or evals_used >= budget:
                break
            x_trial = sobol_points[i]
            f_trial = func(x_trial)
            evals_used += 1
            if f_trial < best_f:
                best_f = f_trial
                best_x = x_trial.copy()

        # --- Phase 2: CMA-ES ---
        # Use scipy's differential_evolution? No, use a simple CMA-ES implementation
        # Actually, let's use a well-tuned (1+1)-ES with better adaptation,
        # or implement a minimal CMA-ES.
        
        # For robustness and speed, use a simplified CMA-ES or well-tuned (mu+lambda)-ES
        # Let's implement a compact CMA-ES
        
        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # CMA-ES parameters
        mu = max(1, (n_sobol // 2))  # number of parents
        lambda_ = 2 * mu  # population size
        c_sigma = (mu_eff + 2) / ((dim + 2)**2 + mu_eff)  # sigma adaptation rate
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)  # path cm rate
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)  # rank-one update
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))  # rank-mu update
        mu_eff = 1 / sum(1.0 / np.arange(1, mu + 1))  # effective number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        sigma = 2.0  # initial step size
        x_mean = best_x.copy() if best_x is not None else np.zeros(dim)
        f_mean = best_f
        
        # CMA-ES state
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # inverse square root of C
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Precompute eigendecomposition helper
        def compute_C_sqrt_inv(C_mat):
            """Compute inverse square root of covariance matrix."""
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.maximum(eigvals, 1e-20)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt_inv

        def update_C_sqrt_inv(C_mat):
            nonlocal C_inv, C_sqrt_inv
            C_inv = np.linalg.inv(C_mat)
            C_sqrt_inv = compute_C_sqrt_inv(C_mat)

        # Initial setup
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Track best
        overall_best_x = best_x.copy() if best_x is not None else x_mean.copy()
        overall_best_f = best_f

        # CMA-ES loop
        while evals_used < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            fitness = []
            for i in range(lambda_):
                if time.time() > deadline or evals_used >= budget:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                f_y = func(y)
                evals_used += 1
                offspring.append(y)
                fitness.append(f_y)

            if not offspring:
                break

            # Sort by fitness
            indices = np.argsort(fitness)
            offspring = [offspring[i] for i in indices]
            fitness = [fitness[i] for i in indices]

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[i]
            
            # Update path
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ ((new_mean - x_mean) / sigma))
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2/(dim+1)) * chi_n else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - x_mean)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 * dim) * C \
                + c_1 * np.outer(p_c, p_c) \
                + delta_h * np.outer(p_c, p_c) \
                - c_mu * sum(weights[i] * np.outer(offspring[i] - x_mean, offspring[i] - x_mean) for i in range(mu))
            
            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(1e-10, min(sigma, 100.0))

            # Update mean
            x_mean = new_mean
            f_mean = fitness[0]

            # Update best
            if fitness[0] < overall_best_f:
                overall_best_f = fitness[0]
                overall_best_x = offspring[0].copy()

            # Update C_sqrt_inv periodically (expensive)
            # Do it every few generations for speed
            if evals_used % (lambda_ * 3) < lambda_:
                try:
                    update_C_sqrt_inv(C)
                except np.linalg.LinAlgError:
                    # Reset if matrix becomes singular
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt_inv = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)

        # Final polish: a few local steps around the best point
        if evals_used < budget and time.time() < deadline:
            sigma_local = 0.05
            for _ in range(min(20, budget - evals_used)):
                if time.time() > deadline:
                    break
                y = np.clip(overall_best_x + sigma_local * rng.standard_normal(dim), -5, 5)
                f_y = func(y)
                evals_used += 1
                if f_y < overall_best_f:
                    overall_best_f = f_y
                    overall_best_x = y.copy()
                else:
                    sigma_local *= 0.9

        return overall_best_x, overall_best_f
# EVOLVE-BLOCK-END
