"""Optimizer for BBOB-style f22 using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES for 5D multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size lambda
        lam = max(4 + int(3 * np.log(dim)), 10)  # Heuristic for lambda
        # Initial sigma
        sigma = 2.0  # Initial step size, roughly covering the box [-5, 5]
        
        # Mean vector, initialized to center of the box
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Inverse of C
        C_inv = np.eye(dim)
        
        # Eigenvalues and eigenvectors of C
        eigval, eigvec = np.linalg.eigh(C)
        
        # Step size adaptation (CSA) parameters
        cs = (lam + 2) / (lam + 5 + 2 * dim)
        ds = 1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1)) - 1) + cs
        hs = (1 + 0.3 * ds) ** (1 / (lam * cs)) - 1
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Covariance adaptation parameters
        mu = lam // 2  # Number of best solutions used for update
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        c_c = (4 + mu_eff / lam) / (dim + 4 + 2 * mu_eff / lam)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track best solution
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Initial population sampling
        # We will evaluate points one by one to strictly respect the budget
        # and update best immediately for AOCC scoring benefit.
        
        # Helper to sample and evaluate
        def sample_and_eval():
            nonlocal mean, sigma, C, C_inv, eigval, eigvec, p_sigma, p_c, calls_made, best_x, best_f
            # Sample from multivariate normal
            # z ~ N(0, I)
            z = rng.standard_normal(dim)
            # y = B * D * z
            # We maintain eigval, eigvec such that C = eigvec @ diag(eigval) @ eigvec.T
            # So sqrt(C) = eigvec @ diag(sqrt(eigval)) @ eigvec.T
            # y = sigma * sqrt(C) @ z
            # Efficiently: y = sigma * (eigvec * sqrt(eigval)) @ (eigvec.T @ z)
            # But for small dim, direct multiplication is fine.
            # Let's use the decomposition: y = sigma * eigvec @ (sqrt(eigval) * (eigvec.T @ z))
            # Actually, simpler: y = sigma * np.diag(eigvec * np.sqrt(eigval)) @ z
            # Wait, C = V D V^T. sqrt(C) = V sqrt(D) V^T.
            # y = sigma * V sqrt(D) V^T z.
            
            # Let's just compute the square root directly or use the decomposition.
            # For dim=5, computing eigendecomposition every step is cheap.
            # But we can update C incrementally.
            # Standard CMA-ES updates C, then updates eigendecomposition.
            
            # For simplicity and robustness in this short budget, we will recompute eigendecomposition
            # only when necessary or just use the matrix directly if we update C.
            # However, sampling requires sqrt(C).
            # Let's stick to the standard algorithm steps.
            
            # 1. Sample y
            # We need B and D from C.
            # We maintain eigval, eigvec.
            # y = sigma * B * D * z where B = eigvec, D = diag(sqrt(eigval))
            # y = sigma * (eigvec * np.sqrt(eigval)) @ (eigvec.T @ z)
            # Let's do:
            y = sigma * (eigvec * np.sqrt(eigval)) @ (eigvec.T @ z)
            
            x = mean + y
            
            # Boundary handling: clip to [-5, 5]
            x = np.clip(x, -5, 5)
            
            f = func(x)
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return x, f

        # Initialize population
        # We need to evaluate `lam` points per generation.
        # Total budget is 500.
        # If lam=10, we can do 50 generations.
        
        # To maximize AOCC, we should evaluate points sequentially and update best.
        
        # Main Loop
        while calls_made < budget:
            # 1. Sample population
            x_pop = np.zeros((lam, dim))
            f_pop = np.zeros(lam)
            
            for i in range(lam):
                if calls_made >= budget:
                    break
                x_pop[i], f_pop[i] = sample_and_eval()
            
            if calls_made >= budget:
                break
            
            # 2. Sort population
            idx = np.argsort(f_pop)
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
            
            # 3. Update mean
            mean_old = mean.copy()
            mean = np.sum(w[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # 4. Update step size (sigma)
            # h_sigma check
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv @ (mean - mean_old) / sigma
            p_sigma_norm = np.linalg.norm(p_sigma)
            
            if p_sigma_norm / chi_n < (1.4 + 2 / (dim + 1)) * hs:
                sigma *= np.exp(cs / ds * (p_sigma_norm / chi_n - 1))
            
            # 5. Update covariance matrix
            # p_c update
            h_c = (np.linalg.norm(p_c) / chi_n < (1.4 + 2 / (dim + 1)) * hs)
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma
            
            # Rank-1 update
            z_diff = (mean - mean_old) / sigma
            C = (1 + c_1 * c_mu * (1 - np.sum(w**2)) + c_1 * h_c * c_c * (2 - c_c) - c_1) * C \
                + c_1 * np.outer(z_diff, z_diff)
            
            # Rank-mu update
            y_sorted = (x_sorted[:mu] - mean_old) / sigma
            C += c_mu * np.sum(w[:, np.newaxis, np.newaxis] * y_sorted[:, np.newaxis, :] * y_sorted[:, :, np.newaxis], axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update eigendecomposition
            eigval, eigvec = np.linalg.eigh(C)
            # Clip eigenvalues to avoid negative values due to numerical errors
            eigval = np.clip(eigval, 1e-10, None)
            
            # Update C_inv
            C_inv = np.linalg.inv(C)

        return best_x, best_f
# EVOLVE-BLOCK-END
