# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Weights
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        cs_path = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0

        # Helper to check time
        def time_exceeded():
            return time.time() > deadline

        # Initial evaluation of the mean
        f_mean = func(x_mean)
        calls += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()

        # Generation loop
        while calls < budget and not time_exceeded():
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C to identity
                C = np.eye(dim)
                L = np.eye(dim)
                pc = np.zeros(dim)
                cs_path = np.zeros(dim)

            population = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (L @ z)
                # Clip to box
                x_i = np.clip(x_i, -5.0, 5.0)
                population.append((x_i, None))
            
            # Evaluate population
            for i in range(lam):
                if calls >= budget or time_exceeded():
                    break
                x_i, _ = population[i]
                f_i = func(x_i)
                calls += 1
                population[i] = (x_i, f_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()

            # If we didn't evaluate the full population, break
            if calls >= budget or time_exceeded():
                break

            # Sort by fitness
            population.sort(key=lambda x: x[1])
            selected = [p[0] for p in population[:mu]]
            selected_f = [p[1] for p in population[:mu]]
            
            # Update mean
            x_old = x_mean.copy()
            x_mean_new = np.sum([weights[i] * selected[i] for i in range(mu)], axis=0)
            
            # Update path accumulators
            # Note: C^-1/2 is needed for the pc update in standard CMA-ES
            # For simplicity and robustness in this short budget, we use a simplified update
            # or ensure C is well-conditioned.
            
            # Inverse square root of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid division by zero
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            pc = (1 - cs) * pc + cs * C_inv_sqrt @ (x_mean_new - x_old) / sigma
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(cc * (2 - cc)))
            
            cs_path = (1 - cs) * cs_path + (np.sign(hsig) - 1 + cs) * (x_mean_new - x_old) / sigma
            
            # Update C
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum([weights[i] * np.outer(selected[i] - x_old, selected[i] - x_old) for i in range(mu)], axis=0)
            
            C = (1 + c1 * (2 - c1) * (cc * (2 - cc)) / (cs * (1 - cs)) - c1) * C \
                - c1 * rank_one \
                + c1 * cmu / c1 * rank_mu # This formula is slightly off, let's use the standard one
            
            # Standard CMA-ES update for C:
            # C = (1 + c1 * (2 - c1) * (cc * (2 - cc) / (cs * (1 - cs)) - c1)) * C - c1 * rank_one + cmu * rank_mu
            # Let's correct the coefficient.
            coeff = 1 + c1 * (2 - c1) * (cc * (2 - cc) / (cs * (1 - cs)) - c1)
            C = coeff * C - c1 * rank_one + cmu * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(cs / (dim + 3) * (np.linalg.norm(cs_path) / np.sqrt(1 - (1 - cs)**2) - 1))
            
            x_mean = x_mean_new
            # Clip mean to box? No, mean can be outside, but offspring are clipped.
            # However, keeping mean inside helps.
            x_mean = np.clip(x_mean, -5.0, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
