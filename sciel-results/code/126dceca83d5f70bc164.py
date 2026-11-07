"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        n = self.dim
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(n))
        # Ensure we don't exceed budget in first generation
        if lam > self.budget:
            lam = self.budget
        
        # Initial mean: center of the box
        mean = np.zeros(n)
        # Initial sigma: small relative to box size
        # Box is [-5, 5], width 10.
        # Weierstrass has features at various scales.
        # A good initial sigma is often 0.25 * range or similar.
        # Let's start with 0.5.
        sigma = 0.5
        
        # Covariance matrix
        C = np.eye(n)
        # Inverse square root of C
        C_inv_sqrt = np.eye(n)
        # Eigenvalues
        eigvals = np.ones(n)
        
        # Adaptation parameters
        mu = lam // 2  # Number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (n + 4 - mu_eff)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # State variables
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        best_x = mean.copy()
        best_f = np.inf
        
        # Main loop
        while self._call_count < self.budget:
            # Generate population
            # We need to check if we have enough budget for a full generation
            # If not, we might generate a partial population or just stop
            # For simplicity and efficiency, if budget < lam, we just evaluate remaining points
            # But CMA-ES needs a full generation to update parameters.
            # So we should only start a generation if we have at least lam calls left.
            # However, the last few calls can be used for fine-tuning or just random sampling if CMA-ES is stuck.
            
            if self.budget - self._call_count < lam:
                # Use remaining budget for random local search or just stop
                # Let's do a simple random search around the best point
                for _ in range(self.budget - self._call_count):
                    x = best_x + np.random.normal(0, sigma * 0.1, n)
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    self._call_count += 1
                    if f < best_f:
                        best_f = f
                        best_x = x
                break

            # Sample candidates
            candidates = []
            for i in range(lam):
                z = np.random.normal(0, 1, n)
                # Transform z using C_inv_sqrt and sigma
                # x = mean + sigma * C_inv_sqrt @ z
                x = mean + sigma * (C_inv_sqrt @ z)
                x = np.clip(x, -5, 5)
                candidates.append(x)
            
            # Evaluate candidates
            fitness = np.zeros(lam)
            for i in range(lam):
                if self._call_count >= self.budget:
                    break
                fitness[i] = func(candidates[i])
                self._call_count += 1
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = candidates[i].copy()
            
            # If we ran out of budget mid-generation, we can't update CMA-ES parameters properly
            # But we have updated best_x. We can just stop.
            if self._call_count >= self.budget:
                break
                
            # Sort candidates by fitness
            indices = np.argsort(fitness)
            x_sorted = np.array([candidates[i] for i in indices])
            f_sorted = fitness[indices]
            
            # Update mean
            new_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update step size and covariance
            # Calculate normalized difference
            diff = (new_mean - mean) / sigma
            
            # Update ps
            ps = np.sqrt(1 - c_sigma * (2 - c_sigma)) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff)
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
            
            # Update C
            # Heff is the Hessian approximation
            # h_sigma = 1 if ||ps||^2 / chi_n^2 < (1 + 2/(mu_eff-1)) * (1-c_c) else 0
            # chi_n = sqrt(n) * (1 - 1/(4*n) + 1/(21*n^2))
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            h_sigma = 1 if (np.dot(ps, ps) / chi_n**2 < (1 + 2/(mu_eff-1)) * (1-c_c)) else 0
            
            # Rank one update
            # pc * pc^T
            # Rank mu update
            # Sum of weights * (x_i - mean_old) * (x_i - mean_old)^T
            # Note: x_sorted[:mu] are the best mu individuals
            
            # We need to be careful with the covariance update formula.
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc^T + (h_sigma - 1 + c_c) * C) + c_mu * sum(w_i * (x_i - mean_old)(x_i - mean_old)^T)
            
            # Let's compute the term sum(w_i * v_i v_i^T)
            # v_i = x_sorted[i] - mean (old mean)
            # Note: CMA-ES uses the old mean for the rank-mu update
            
            # However, implementing the full CMA-ES update with matrix operations
            # can be numerically tricky and slow if not done carefully.
            # A simpler, robust approach for small n is to use the update formulas directly.
            
            # Let's stick to the standard update.
            # We need C_inv_sqrt to be updated as well.
            
            # For now, let's simplify the update to ensure stability.
            # If the full CMA-ES update is too complex to implement correctly in a few lines,
            # we might fall back to a simpler strategy like (1+1)-ES or a simplified DE.
            # But CMA-ES is the goal.
            
            # Let's implement the update carefully.
            # 1. Compute the rank-mu update term
            # v = x_sorted[:mu] - mean
            # term = sum(weights[i] * outer(v[i], v[i]) for i in range(mu))
            
            v = x_sorted[:mu] - mean
            # Efficient computation of sum of weighted outer products
            # term = v.T @ np.diag(weights) @ v
            # But v is (mu, n), weights is (mu,)
            # term = (v * weights[:, None]).T @ v
            
            term_rank_mu = (v * weights[:, None]).T @ v
            
            # 2. Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (h_sigma - 1 + c_c) * C) + c_mu * term_rank_mu
            
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (h_sigma - 1 + c_c) * C)
            C += c_mu * term_rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update C_inv_sqrt
            # This requires eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues due to numerical errors
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update mean
            mean = new_mean
            
        return best_x, best_f
# EVOLVE-BLOCK-END
