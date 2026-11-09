import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: GP Surrogate for Global Exploration
        n_gp = 100
        
        X_gp = []
        Y_gp = []
        
        # Initial random points
        n_init = 10
        for _ in range(n_init):
            if calls >= budget or time.time() >= deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            fx = eval_point(x)
            X_gp.append(x)
            Y_gp.append(fx)
            
        X_gp = np.array(X_gp)
        Y_gp = np.array(Y_gp)
        
        # GP Parameters
        sigma_f = 1.0
        l = 1.5
        sigma_n = 1e-4
        
        def gp_predict(X_new, X_train, Y_train):
            N_train = len(X_train)
            N_new = len(X_new)
            
            diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            K_train = sigma_f ** 2 * np.exp(-0.5 * sq_dist / (l ** 2)) + sigma_n * np.eye(N_train)
            
            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            except np.linalg.LinAlgError:
                K_train += 1e-6 * np.eye(N_train)
                try:
                    L = np.linalg.cholesky(K_train)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                except np.linalg.LinAlgError:
                    return np.full(N_new, np.mean(Y_train)), np.full(N_new, sigma_f ** 2)

            diff_new = X_new[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            sq_dist_new = np.sum(diff_new ** 2, axis=2)
            K_new = sigma_f ** 2 * np.exp(-0.5 * sq_dist_new / (l ** 2))
            
            mean = K_new @ alpha
            
            K_new_L_inv = np.linalg.solve(L, K_new.T)
            var = sigma_f ** 2 - np.sum(K_new_L_inv ** 2, axis=1)
            var = np.clip(var, 0, None)
            
            return mean, var

        # Sequential EI
        for _ in range(n_gp - n_init):
            if calls >= budget or time.time() >= deadline:
                break
            
            n_cand = 500
            X_cand = np.random.uniform(-5, 5, (n_cand, dim))
            
            mean, var = gp_predict(X_cand, X_gp, Y_gp)
            std = np.sqrt(var)
            
            f_best = np.min(Y_gp)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (f_best - mean) / std
                z = np.clip(z, -10, 10)
                
                Phi = norm.cdf(z)
                phi = norm.pdf(z)
                
                EI = (f_best - mean) * Phi + std * phi
                EI = np.where(std > 1e-8, EI, np.where(mean < f_best, f_best - mean, 0))
            
            idx_best = np.argmax(EI)
            x_next = X_cand[idx_best]
            
            fx_next = eval_point(x_next)
            
            X_gp.append(x_next)
            Y_gp.append(fx_next)
            
            X_gp = np.array(X_gp)
            Y_gp = np.array(Y_gp)

        # Phase 2: CMA-ES for Local Refinement
        m = best_x.copy()
        
        lam = 10
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1) + 0.5) - np.log(mu + 0.5)
        w = w / np.sum(w)
        mu_eff = np.sum(w) ** 2 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = 4 / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        sigma = 0.5
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        while calls < budget:
            if time.time() >= deadline:
                break
            
            Z = np.random.standard_normal((lam, dim))
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            Y = Z @ C_sqrt.T
            X = m + sigma * Y
            X = np.clip(X, -5, 5)
            
            F = np.array([eval_point(x) for x in X])
            
            idx_sort = np.argsort(F)
            X_sorted = X[idx_sort]
            F_sorted = F[idx_sort]
            
            m_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            c_1_mu = c_1 + c_mu
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            delta = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(c_1_mu * (2 - c_1_mu)) * (C_inv_sqrt @ delta)
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (25 * dim ** 2))
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            p_c = (1 - c_c) * p_c + c_c * delta
            Y_sorted = (X_sorted[:mu] - m) / sigma
            C_new = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C_new += c_mu * w[i] * np.outer(Y_sorted[i], Y_sorted[i])
            
            C_new = (C_new + C_new.T) / 2
            m = m_new
            C = C_new
            
        return best_x, best_f
