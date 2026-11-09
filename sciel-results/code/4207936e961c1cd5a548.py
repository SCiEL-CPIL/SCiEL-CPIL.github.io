import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        N = self.budget
        
        # CMA-ES Parameters
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        x_mean = np.zeros(d)
        sigma = 1.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        calls = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls += 1
        
        start_time = time.time()
        deadline = start_time + 290
        gen = 0
        
        while calls + lambda_ <= N and time.time() < deadline:
            gen += 1
            
            # Compute Eigen decomposition of C for sampling
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                break
            
            # Sample population
            pop = np.zeros((lambda_, d))
            for i in range(lambda_):
                z = np.random.randn(d)
                pop[i] = x_mean + sigma * (L @ z)
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate population
            f_pop = np.array([func(x) for x in pop])
            calls += lambda_
            
            # Sort by fitness
            order = np.argsort(f_pop)
            pop = pop[order]
            f_pop = f_pop[order]
            
            # Update best
            if f_pop[0] < best_f:
                best_f = f_pop[0]
                best_x = pop[0].copy()
            
            # Update mean
            x_mean_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (x_mean_new - x_mean))
            
            # Update pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * gen)) < (1 + 2 / (d + 1)) * chi_n)
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            
            # Update C
            diff = pop[:mu] - x_mean
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc) \
                + c_mu / sigma**2 * np.einsum('i,ij,ik->jk', weights, diff, diff)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x_mean = x_mean_new
            
            # Convergence check
            if sigma < 1e-12:
                break

        return None
