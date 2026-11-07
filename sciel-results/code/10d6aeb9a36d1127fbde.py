"""CMA-ES based optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is a state-of-the-art derivative-free optimizer that adapts the
    covariance matrix of a multivariate normal distribution to the local
    landscape structure. This is particularly effective for rotated/shifted
    landscapes like BBOB functions where axis-aligned methods struggle.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, use 300s to be safe
        deadline = start_time + 300
        
        n = self.dim
        budget = self.budget
        
        # Split budget: 80% CMA-ES, 20% Local Refinement
        cma_budget = int(budget * 0.8)
        local_budget = budget - cma_budget
        
        # CMA-ES Parameters
        sigma = 2.0
        lambda_ = 12 
        
        # Initial mean
        mean = np.zeros(n)
        C = np.eye(n)
        
        # Step sizes for parameters
        cmu = 1.0 / (4 + np.exp(n / 20))
        ccs = (4 + n / lambda_) / (n + 4 + 2 * n / lambda_)
        dsigma = 1 + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (n + 1)) - 1) + ccs
        csigma = (1 + ccs) / (n + dsigma)
        cc = (2 + 2 * np.sqrt((lambda_ / 2 - 1) / (n + 1))) / (n + 2)
        
        # State variables
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Path variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        def clip(x):
            return np.clip(x, -5, 5)
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = clip(x)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx
        
        # Phase 1: CMA-ES
        f_mean = evaluate(mean)
        
        while calls < cma_budget and time.time() < deadline:
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
            
            Z = np.random.randn(lambda_, n)
            X = mean + sigma * (Z @ L.T)
            X = np.clip(X, -5, 5)
            
            F = np.array([evaluate(x) for x in X])
            
            if calls >= cma_budget:
                break
            
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            mu = max(1, int(lambda_ / 2))
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            if np.isinf(F_sorted[0]):
                break
            
            X_top = X_sorted[:mu]
            mean_new = np.sum(w[:, np.newaxis] * X_top, axis=0)
            
            v = (mean_new - mean) / sigma
            try:
                y = np.linalg.solve(L, v)
            except np.linalg.LinAlgError:
                y = v
            
            p_sigma = (1 - csigma) * p_sigma + csigma * np.sqrt(mu_eff) * y
            
            E_norm = np.sqrt(n) * (1 - 1.0/(4.0*n))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(csigma / dsigma * (norm_p_sigma / E_norm - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                C_inv_half_v = C_inv_half @ v
            except np.linalg.LinAlgError:
                C_inv_half_v = v
            
            p_c = (1 - cc) * p_c + cc * np.sqrt(mu_eff) * C_inv_half_v
            
            Y = (X_top - mean) / sigma
            try:
                Y_std = np.linalg.solve(L, Y.T).T
            except np.linalg.LinAlgError:
                Y_std = Y
            
            c_mu_base = 2.0 * (mu_eff - 2.0 + 1.0/mu_eff) / ((n + 2.0)**2 + mu_eff)
            c_mu = c_mu_base * (1.0 - csigma / dsigma * (norm_p_sigma**2 - 1.0))
            c_mu = np.clip(c_mu, 0.0, 2.0 / mu_eff)
            
            outer_sum = Y_std.T @ (w[:, np.newaxis] * Y_std)
            C = (1 - c_mu) * C + c_mu * outer_sum
            C = (C + C.T) / 2.0
            mean = mean_new
            
            if sigma < 1e-8:
                break

        # Phase 2: Local Refinement (Nelder-Mead)
        if best_x is not None and calls < budget and time.time() < deadline:
            # Initialize Nelder-Mead simplex
            x0 = best_x.copy()
            f0 = best_f
            alpha = 0.5 # Step size for simplex initialization
            
            # Create simplex
            simplex = [x0.copy()]
            fvals = [f0]
            
            for i in range(n):
                xi = x0.copy()
                xi[i] += alpha if xi[i] < 4.5 else -alpha # Keep inside box
                simplex.append(xi)
                fi = evaluate(xi)
                fvals.append(fi)
                
            # Nelder-Mead parameters
            alpha_n = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_n = 0.5
            
            # Main NM loop
            while calls < budget and time.time() < deadline:
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break
                    
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha_n * (centroid - simplex[-1])
                x_r = clip(x_r)
                f_r = evaluate(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = clip(x_e)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = clip(x_c)
                    f_c = evaluate(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma_n * (simplex[i] - simplex[0])
                            simplex[i] = clip(simplex[i])
                            fvals[i] = evaluate(simplex[i])
                            
                # Update best
                min_f = np.min(fvals)
                if min_f < best_f:
                    best_f = min_f
                    best_x = simplex[np.argmin(fvals)].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
