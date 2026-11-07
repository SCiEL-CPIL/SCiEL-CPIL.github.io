"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-restart exploration and CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        call_count = 0
        
        # Phase 1: Global exploration with random restarts + short CMA-ES
        n_restarts = 3
        calls_per_restart = 60
        
        for r in range(n_restarts):
            if time.time() >= deadline or call_count >= budget:
                break
            
            # Initialize with random point
            if r == 0:
                x = rng.uniform(-5, 5, d)
            else:
                # Gradually shrink the range toward center
                span = 10.0 * (0.6 ** r)
                x = rng.uniform(-span / 2, span / 2, d)
            
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Short CMA-ES run for local refinement
            if call_count + 20 >= budget or time.time() >= deadline:
                break
            
            sigma = 1.0
            x_mean = x.copy()
            C = np.eye(d)
            N = 6  # Small population for speed
            mu = N // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 0.5) / (d + 1.5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
            c_c = (4 - mu_eff + c_sigma) / (d + 4 + c_sigma)
            c_cov = (2 * (mu_eff - 2 + 1/mu_eff)) / ((d + 1.3)**2 + mu_eff) + c_c * (2 - c_c - 2/(d+1.3))
            c_1 = 2 / ((d + 1.3)**2 + N)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((d + 2)**2 + N))
            
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            chi_lambda = np.sqrt(d) * (1 - 1/(2*d) + 1/(8*d**2))
            
            for _ in range(min(20, budget - call_count)):
                if time.time() >= deadline or call_count >= budget:
                    break
                
                # Sample
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-6)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C)
                
                Y = rng.standard_normal((N, d))
                X = x_mean[None, :] + sigma * (Y @ L.T)
                
                F = np.zeros(N)
                for i in range(N):
                    if call_count >= budget:
                        break
                    F[i] = func(X[i])
                    call_count += 1
                
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]
                
                Y_sorted = (X - x_mean[None, :]) / sigma
                x_new = x_mean + sigma * np.dot(weights, Y_sorted[:mu])
                
                L_inv = np.linalg.inv(L)
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (L_inv @ (x_new - x_mean) / sigma)
                
                threshold = (1.4 + 2/(d+1)) * chi_lambda
                h_sigma = 1.0 if np.linalg.norm(p_sigma) < threshold else 0.0
                
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1))
                p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * (x_new - x_mean) / sigma
                
                delta_c = (1 - h_sigma) * c_c * (1 - c_c)
                C = (1 - c_cov - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_c * C)
                
                Y_mu = Y_sorted[:mu]
                C += c_mu * ((Y_mu * weights[:, None]).T @ Y_mu)
                
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-6:
                    C += (1e-6 - np.min(eigvals)) * np.eye(d)
                
                x_mean = x_new
                
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = X[0].copy()
        
        # Phase 2: Full CMA-ES from the best point found
        if best_x is not None and call_count < budget and time.time() < deadline:
            x_mean = best_x.copy()
            fx_mean = best_f
            sigma = 0.5
            C = np.eye(d)
            N = 10
            mu = N // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 0.5) / (d + 1.5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
            c_c = (4 - mu_eff + c_sigma) / (d + 4 + c_sigma)
            c_cov = (2 * (mu_eff - 2 + 1/mu_eff)) / ((d + 1.3)**2 + mu_eff) + c_c * (2 - c_c - 2/(d+1.3))
            c_1 = 2 / ((d + 1.3)**2 + N)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((d + 2)**2 + N))
            
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            chi_lambda = np.sqrt(d) * (1 - 1/(2*d) + 1/(8*d**2))
            
            while call_count < budget and time.time() < deadline:
                # Sample
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-6)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C)
                
                Y = rng.standard_normal((N, d))
                X = x_mean[None, :] + sigma * (Y @ L.T)
                
                F = np.zeros(N)
                for i in range(N):
                    if call_count >= budget:
                        break
                    F[i] = func(X[i])
                    call_count += 1
                
                if np.all(np.isnan(F)):
                    break
                
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]
                
                Y_sorted = (X - x_mean[None, :]) / sigma
                x_new = x_mean + sigma * np.dot(weights, Y_sorted[:mu])
                
                L_inv = np.linalg.inv(L)
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (L_inv @ (x_new - x_mean) / sigma)
                
                threshold = (1.4 + 2/(d+1)) * chi_lambda
                h_sigma = 1.0 if np.linalg.norm(p_sigma) < threshold else 0.0
                
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1))
                p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * (x_new - x_mean) / sigma
                
                delta_c = (1 - h_sigma) * c_c * (1 - c_c)
                C = (1 - c_cov - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_c * C)
                
                Y_mu = Y_sorted[:mu]
                C += c_mu * ((Y_mu * weights[:, None]).T @ Y_mu)
                
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-6:
                    C += (1e-6 - np.min(eigvals)) * np.eye(d)
                
                x_mean = x_new
                
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = X[0].copy()
        
        # Phase 3: Fine-tuning with (1+1)-ES
        if best_x is not None and call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            
            while call_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                call_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87
        
        return best_x, best_f
# EVOLVE-BLOCK-END
