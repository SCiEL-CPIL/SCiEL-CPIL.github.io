import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        # CMA-ES Parameters
        # Population size
        lam = max(4, 4 + int(3 * np.log(dim)))
        # Number of parents
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (2 + 2.0 / (dim + 1)) / (dim + 4 + 2.0 / (dim + 1))
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        x_mean = np.zeros(dim)
        sigma = 1.5
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = None
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng()

        def evaluate(x):
            nonlocal calls, best_x, best_f
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Initial evaluation
        f_mean = evaluate(x_mean)
        best_f = min(best_f, f_mean)
        best_x = x_mean.copy()

        # Pre-compute D (Cholesky of C)
        D = np.linalg.cholesky(C)

        while calls < budget:
            if time.time() > deadline:
                break

            # Adjust population size if near budget
            current_lam = lam
            if calls + current_lam > budget:
                current_lam = budget - calls
            if current_lam == 0:
                break

            # Sample
            z = rng.standard_normal((current_lam, dim))
            # Whitening: y = D @ z
            y = np.einsum('ij,kj->ki', D, z)
            x_offspring = x_mean + sigma * y
            x_offspring = np.clip(x_offspring, -5, 5)

            # Evaluate
            f_offspring = np.array([evaluate(xi) for xi in x_offspring])

            # Sort
            idx = np.argsort(f_offspring)
            x_offspring = x_offspring[idx]
            f_offspring = f_offspring[idx]

            # Save old mean
            x_mean_old = x_mean.copy()

            # Update mean
            x_mean = np.sum(x_offspring[:mu] * weights[:, np.newaxis], axis=0)

            # Update step size
            # sigma = sigma * exp(c_sigma / chi_n * (||ps|| - chi_n))
            # But we need to update ps first
            
            # Update ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * D @ (x_mean - x_mean_old)
            # Note: (x_mean - x_mean_old) is the step in x-space.
            # We need to map it back to y-space: D^-1 @ (x_mean - x_mean_old)
            # Actually, the formula is:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * D^{-1} * (x_mean - x_mean_old) / sigma
            # Since x_mean - x_mean_old = sigma * D @ y_mean_weighted
            # D^{-1} * (sigma * D @ y_mean_weighted) / sigma = y_mean_weighted
            # So we can compute y_mean_weighted directly from the sorted offspring?
            # y_i = D^{-1} (x_i - x_mean_old) / sigma
            # But x_i were generated as x_mean_old + sigma * D * z_i
            # So y_i = z_i (if no clipping occurred).
            # Clipping complicates this.
            # Standard CMA-ES implementations often recompute the whitening or handle clipping carefully.
            # For simplicity and robustness against clipping effects, let's just use the standard update 
            # assuming the step is in the y-space.
            
            # Let's compute the weighted sum of the z-vectors (or y-vectors)
            # We have z. The selected z's are z[idx[:mu]]
            y_mean_weighted = np.sum(z[idx[:mu]] * weights[:, np.newaxis], axis=0)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean_weighted
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) - chi_n))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update covariance matrix
            # pc = (1 - c_c) * pc + c_c * (x_mean - x_mean_old) / sigma
            # Wait, pc is in x-space or y-space?
            # In standard CMA-ES, pc is in the "search distribution" space, which is y-space?
            # No, pc is usually updated as:
            # pc = (1 - c_c) * pc + c_c * (x_mean - x_mean_old) / sigma
            # And the rank-one update is:
            # C += c_1 * (pc * pc^T - (1 - (1 - c_c)^2) * C) + c_mu * sum(w_i * (y_i * y_i^T))
            # Where y_i = D^{-1} (x_i - x_mean_old) / sigma
            
            # Let's stick to the standard formulation where pc is in x-space scaled by sigma?
            # Actually, many implementations keep pc in the "unwhitened" space.
            # Let's use the formulation from Hansen's paper:
            # pc = (1 - c_c) pc + c_c / sigma (x_mean - x_mean_old)
            
            pc = (1 - c_c) * pc + c_c / sigma * (x_mean - x_mean_old)
            
            # Rank-one update term
            # c_1 * (pc pc^T - (1 - (1 - c_c)^2) C)
            
            # Rank-mu update term
            # c_mu * sum w_i (y_i y_i^T)
            # y_i = D^{-1} (x_i - x_mean_old) / sigma
            # Since x_i = x_mean_old + sigma D z_i, y_i = z_i (ignoring clipping)
            # If clipping occurred, z_i is not exactly D^{-1}(x_i - x_mean_old)/sigma.
            # For robustness, let's compute y_i explicitly.
            # y_i = np.linalg.solve(D, (x_i - x_mean_old) / sigma)
            
            y_selected = np.array([np.linalg.solve(D, (xi - x_mean_old) / sigma) for xi in x_offspring[:mu]])
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc^T + (1 - (1 - c_c)^2) * C) + c_mu * sum(w_i * y_i * y_i^T)
            # Note: The term (1 - (1 - c_c)^2) is often approximated or handled carefully.
            # Standard formula:
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - (1-c_c)^2) C) + c_mu sum w_i y_i y_i^T
            # This simplifies to:
            # C = (1 - c_1 - c_mu + c_1(1 - (1-c_c)^2)) C + c_1 pc pc^T + c_mu sum w_i y_i y_i^T
            # Let's just compute the terms.
            
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * y_selected[:, np.newaxis, :] * y_selected[:, np.newaxis, :].T, axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - (1 - c_c)**2) * C) + c_mu * rank_mu
            
            # Ensure positive definiteness
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update D
            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-10 * np.eye(dim)
                try:
                    D = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: reset C to identity
                    C = np.eye(dim)
                    D = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    sigma = 1.0

        return best_x, best_f
