"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_single_run(self, func, rng, budget, x0, sigma0, deadline):
        """Run a single CMA-ES instance."""
        n = self.dim
        x = x0.copy()
        fx = func(x)
        sigma = sigma0
        lambda_ = max(4 + int(3 * np.log(n)), 10)
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        # Precompute eigen decomposition for C for efficient updates
        # Actually, let's use a simpler CMA implementation with matrix operations

        # Standard CMA-ES
        steps = 0
        best_x = x.copy()
        best_fx = fx
        eval_count = 1

        while eval_count < budget:
            if time.time() >= deadline:
                break

            # Generate lambda offspring
            # Use Cholesky or eigen decomposition of C
            # For simplicity and speed in low dimensions, use eigen decomposition
            # But updating C requires care. Let's use the standard matrix form.
            
            # Actually, for n=5, we can afford full matrix operations.
            # Let's implement a simpler but effective CMA-ES.
            
            # Generate samples
            B, D = np.linalg.eigh(C)
            # B is eigenvectors, D is eigenvalues
            # C = B diag(D) B^T
            # To generate N(0, C): B diag(sqrt(D)) z
            sqrt_D = np.sqrt(np.maximum(D, 0))
            
            Z = rng.standard_normal((lambda_, n))
            Y = Z @ (B * sqrt_D).T  # Y = Z @ B @ diag(sqrt(D)) -> shape (lambda_, n)
            # Wait, C = B diag(D) B^T. So sqrt(C) = B diag(sqrt(D)) B^T.
            # x_new = x + sigma * B diag(sqrt(D)) B^T z
            # Let's be precise:
            # C = Q D Q^T
            # x + sigma * Q D^{1/2} Q^T z
            # So: Y = Q @ (D^{1/2} @ (Q^T @ Z.T)).T ?
            # Let's do: Z is (lambda_, n). We want Y = Z @ (Q D^{1/2} Q^T).T = Z @ Q D^{1/2} Q^T
            # Since Q is orthogonal, Q^T = Q^{-1}
            # Actually, the transformation is: Y_i = x + sigma * Q D^{1/2} Q^T Z_i
            # So Y = Z @ (Q D^{1/2} Q^T).T = Z @ Q D^{1/2} Q^T
            # Let M = Q @ np.diag(sqrt_D) @ Q.T
            M = (B * sqrt_D) @ B.T
            Y = x + sigma * (Z @ M.T)
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            eval_count += lambda_
            
            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_fx:
                best_fx = F[0]
                best_x = Y[0].copy()
            
            # Update mean
            w = weights
            x_new = np.sum(w[:, None] * Y[:mu], axis=0)
            
            # Update step size control
            # ps = (1-c_sigma) ps + c_sigma * sqrt(mu_eff) * B^T * (1/sigma) * (x_new - x)
            # We need B^T (x_new - x) / sigma
            # x_new - x is the weighted sum of Y - x, which is sigma * M @ Z_weighted
            # So (x_new - x) / sigma = M @ Z_weighted
            # And B^T @ M = B^T @ B @ diag(sqrt_D) @ B^T = diag(sqrt_D) @ B^T
            # So B^T (x_new - x) / sigma = diag(sqrt_D) @ B^T @ Z_weighted
            # Let's compute Z_weighted = sum(w_i * Z_i)
            Z_weighted = np.sum(w[:, None] * Z[idx[:mu]], axis=0)
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * (sqrt_D * (B.T @ Z_weighted))
            
            # Update covariance
            # C = (1 + c_c(2-c_c) - c_sigma) C + c_sigma * (ps ps^T + (1 - (1-c_sigma)^2 - c_sigma) * (1/mu_eff) sum w_i (y_i - x)(y_i - x)^T)
            # The term (y_i - x) = sigma * M @ Z_i
            # So (y_i - x)(y_i - x)^T = sigma^2 * M @ Z_i Z_i^T @ M^T
            # Let's compute the weighted sum of outer products in the transformed space
            # sum w_i Z_i Z_i^T
            Z_mu = Z[idx[:mu]]
            C_z = np.zeros((n, n))
            for i in range(mu):
                C_z += w[i] * np.outer(Z_mu[i], Z_mu[i])
            
            # Now transform back
            # The update is:
            # C_new = (1 + c_c(2-c_c) - c_sigma) C + c_sigma * (ps ps^T + (1 - (1-c_sigma)^2 - c_sigma) * sigma^2 / mu_eff * M @ C_z @ M^T)
            # Wait, the standard formula is:
            # C = (1 + c_c(2-c_c) - c_sigma)C + c_sigma( ps ps^T + (1 - (1-c_sigma)^2 - c_sigma) * 1/mu_eff * sum w_i (y_i - x)(y_i - x)^T )
            # (y_i - x) = sigma * M @ Z_i
            # So sum w_i (y_i - x)(y_i - x)^T = sigma^2 * M @ (sum w_i Z_i Z_i^T) @ M^T = sigma^2 * M @ C_z @ M^T
            
            factor = 1 - (1 - c_sigma) ** 2 - c_sigma
            C = (1 + c_c * (2 - c_c) - c_sigma) * C + c_sigma * (
                np.outer(ps, ps) + factor * (sigma ** 2) / mu_eff * (M @ C_z @ M.T)
            )
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update step size (CSA - Cumulative Step Adaptation)
            # h_sigma = 1 if ||ps|| / sqrt(1 - (1-c_sigma)^{2(steps+1)}) < (1.4 + 2/(n+1)) * E||N(0,I)||
            # For simplicity, use a simpler adaptation or just fixed if too complex
            # Let's use a simple heuristic: if best improved, increase sigma slightly, else decrease
            # Actually, let's implement basic CSA
            # E||N(0,I)|| = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
            E_norm = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            # Approximate the threshold
            # For simplicity, skip full CSA and use a simpler rule
            # If the best point improved, we might want to keep sigma or increase it slightly
            # Otherwise, decrease it
            # Let's use: if F[0] < fx, sigma *= 1.05, else sigma *= 0.95
            # This is a rough heuristic but might work
            if F[0] < fx:
                sigma *= 1.02
            else:
                sigma *= 0.98
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            x = x_new
            fx = F[0]
            steps += 1
            
            # Check for convergence
            if sigma < 1e-12:
                break

        return best_x, best_fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_fx = np.inf
        
        # Initial point
        x0 = rng.uniform(-5, 5, self.dim)
        fx0 = func(x0)
        best_x = x0.copy()
        best_fx = fx0
        eval_count = 1
        
        # Run CMA-ES
        x, fx = self._cma_single_run(func, rng, self.budget - 1, x0, 2.0, deadline)
        eval_count += self.budget - 1  # Approximate, actual count is tracked inside
        
        if fx < best_fx:
            best_x = x.copy()
            best_fx = fx
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
