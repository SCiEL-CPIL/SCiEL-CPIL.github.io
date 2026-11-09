import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        N = self.budget
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(d))
        if lam < 4: lam = 4
        if lam > 20: lam = 20
        
        cmu = min(2, lam // 2)
        c1 = (2 + (2 / (d + 1))**2) / ((d + 2)**2 + 2)
        cmu_eff = cmu * c1
        c_sigma = (cmu_eff + 2) / (d + cmu_eff + 5)
        dsigma = 1 + 2 * max(0, np.sqrt((cmu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + cmu_eff / c_sigma) / (d + 4 + 2 * cmu_eff / c_sigma)
        c_cov = (2 + (2 / (d + 1))**2) / ((d + 2)**2 + 2 + 2 * cmu_eff / c_sigma)
        mu_sigma = 2 / (cmu_eff + 2)
        weights = np.log(cmu + 0.5) - np.log(np.arange(1, cmu + 1))
        weights = weights / np.sum(weights)
        mu_w = np.sum(weights)
        chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        rng = np.random.default_rng(42)
        
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        start_time = time.time()
        deadline = start_time + 280
        
        num_generations = N // lam
        
        for gen in range(num_generations):
            if time.time() > deadline:
                break
            
            # 1. Sample
            # Use eigendecomposition of C for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            Z = rng.standard_normal((lam, d))
            Z_scaled = Z * np.sqrt(eigvals)
            X = x_mean + sigma * (Z_scaled @ eigvecs.T)
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            F = np.array([func(x) for x in X])
            calls += lam
            
            # 3. Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # 4. Update Mean
            x_mean_new = np.sum(X[:cmu] * weights.reshape(-1, 1), axis=0)
            
            # 5. Update Step Size
            z = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * cmu_eff) * z
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) 
                       < (1.4 + 2 / (d + 1)) * chi_n)
            
            # 6. Update Covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * cmu_eff) * z
            
            diff = (X[:cmu] - x_mean) / sigma
            rank_mu_term = np.sum(weights.reshape(-1, 1, 1) * diff[:, :, None] * diff[:, None, :], axis=0)
            
            C = (1 + c1 * (c_c * (2 - c_c) - cmu_eff) + cmu_eff) * C \
                + c1 * np.outer(p_c, p_c) \
                - cmu_eff * rank_mu_term
                
            C = (C + C.T) / 2
            eigvals_check, _ = np.linalg.eigh(C)
            if np.min(eigvals_check) < 1e-12:
                C = C + 1e-12 * np.eye(d)
                
            sigma *= np.exp((1 / dsigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            x_mean = x_mean_new

        return best_x, best_f
