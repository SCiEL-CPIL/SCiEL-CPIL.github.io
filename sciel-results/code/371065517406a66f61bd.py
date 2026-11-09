"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase: Halton space-filling exploration then CMA-ES local optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    @staticmethod
    def _halton_sequence(n, dim, base=None):
        """Generate n points in [0,1]^dim using Halton sequences."""
        if base is None:
            # First dim primes
            base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
        
        points = []
        for i in range(1, n + 1):
            point = []
            for d in range(dim):
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= base[d]
                    r += f * (i_val % base[d])
                    i_val //= base[d]
                point.append(r)
            points.append(point)
        return np.array(points)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin

        # Phase 1: Halton space-filling exploration
        # Use ~15% of budget for initial exploration
        n_explore = max(20, budget // 7)  # ~71 points for budget=500
        halton_points = self._halton_sequence(n_explore, dim)
        
        # Map from [0,1]^dim to [-5,5]^dim
        halton_mapped = halton_points * 10.0 - 5.0
        
        best_x = None
        best_fx = np.inf
        calls = 0
        
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = halton_mapped[i]
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no evaluations made
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1

        # Phase 2: CMA-ES local optimization from best Halton point
        remaining_budget = budget - calls
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_fx

        # CMA-ES parameters
        x = best_x.copy()
        sigma = 1.5  # Initial step size
        C = np.eye(dim)
        C_inv = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # CMA-ES adaptation rates
        mu = max(1, dim // 2)  # Number of rejections to consider
        lambda_ = max(4, 2 * dim)  # Population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Precompute square root of C for sampling
        def cholesky_with_fallback(A):
            try:
                return np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                # Add small regularization
                return np.linalg.cholesky(A + 1e-10 * np.eye(A.shape[0]))
        
        B = np.linalg.cholesky(C)
        D = np.sqrt(np.diag(B @ B.T))
        
        c_calls = 0
        no_improve = 0
        stagnation_limit = 3 * dim + 20
        
        while calls + c_calls < budget and time.time() < deadline:
            # Generate lambda_ offspring
            offspring = []
            for k in range(lambda_):
                if calls + c_calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                c_calls += 1
                offspring.append((fy, y))
            
            if not offspring:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            if offspring[0][0] < best_fx:
                best_fx = offspring[0][0]
                best_x = offspring[0][1].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Update mean
            y_w = np.zeros(dim)
            for k in range(min(mu, len(offspring))):
                y_w += weights[k] * offspring[k][1]
            x = y_w
            
            # Update step size (sigma)
            y_sigma = B.T @ (D @ (np.linalg.solve(B, y_w - x)))  # This is simplified
            # Actually, let's use a simpler update
            delta = y_w - x
            if np.linalg.norm(delta) > 1e-10:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(delta) / sigma - np.sqrt(dim)))
            
            # Update covariance matrix
            delta_mean = (y_w - x) / sigma
            pc = (1 - c_c) * pc + c_c * delta_mean
            rank_one = np.outer(pc, pc)
            
            rank_mu = np.zeros((dim, dim))
            for k in range(min(mu, len(offspring))):
                yk = (offspring[k][1] - x) / sigma
                rank_mu += weights[k] * np.outer(yk, yk)
            
            C = (1 + c_1 * c_c * (1 - c_c) * np.dot(pc, pc) - c_1) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            D = np.sqrt(np.diag(B @ B.T))
            
            # Check for convergence
            if sigma < 1e-6 or no_improve > stagnation_limit:
                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
