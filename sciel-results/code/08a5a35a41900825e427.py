import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def track_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return best_f
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # CMA-ES Parameters
        # For dim=5, lambda=10 is standard. 
        # We use a slightly larger population if budget allows, but 10 is safe.
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Initial parameters
        # Start near center, as minimizer is in [-4, 4]^5
        # Initial sigma should cover the box reasonably well
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Precompute inverse square root for sampling? 
        # We will update C and use np.linalg.cholesky or similar for sampling.
        # For dim=5, full matrix ops are cheap.
        
        # Control parameters
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim+1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim+2)**2 + mueff))
        
        # Step size adaptation parameters
        damps = 1 + 2 * max(0, np.sqrt((mueff-1)/(dim+1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        hsig = 0
        
        # History for stagnation detection
        last_improvement_call = 0
        best_f_history = []
        
        # Main Loop
        while calls_used < budget and time.time() < deadline - 2.0:
            
            # Sample population
            pop = np.zeros((lam, dim))
            # Use Cholesky decomposition for C^0.5
            # Ensure C is positive definite
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD, add small jitter
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(dim)
            
            for i in range(lam):
                if calls_used >= budget:
                    break
                z = np.random.randn(dim)
                pop[i] = x_mean + sigma * (L @ z)
                # Clip to bounds? 
                # CMA-ES usually handles bounds via resampling or mapping.
                # Here we simply clip, which is a simple but effective bound handling for small dim.
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate population
            fit = np.zeros(lam)
            for i in range(lam):
                if calls_used >= budget:
                    break
                fit[i] = track_call(pop[i])
            
            # Sort population
            idx = np.argsort(fit)
            pop = pop[idx]
            fit = fit[idx]
            
            # Update mean
            w = weights[:mu]
            x_new = np.sum(w[:, None] * pop[:mu], axis=0)
            
            # Update sigma (CSA)
            # Calculate step size
            step = np.linalg.norm(x_new - x_mean)
            if step > 0:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / step * (x_mean - x_new)
            ps_norm = np.linalg.norm(ps)
            hsig = (ps_norm / np.sqrt(1 - (1 - cs)**2) / chiN) < (1.4 + 2/(dim+1))
            
            sigma = sigma * np.exp(cs/damps * (ps_norm/chiN - 1))
            
            # Update pc (CMA-ES)
            if hsig:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) / step * (x_mean - x_new)
            else:
                pc = (1 - cc) * pc
            
            # Update C
            # Rank-1 update
            y = (x_mean - x_new) / sigma
            C = (1 + c1 * hsig * (cc - cc + 1) - c1) * C + (c1 + c1 * hsig * (cc - cc + 1)) * np.outer(y, y)
            
            # Rank-mu update
            y_norm = np.linalg.norm(y)
            if y_norm > 0:
                # We need the normalized differences of the selected individuals
                # v_i = (x_i - x_old) / sigma
                # Note: x_old is the previous mean, which is x_mean before update.
                # We stored pop[:mu] which are the best individuals.
                # We need to calculate v relative to the OLD mean.
                # x_mean was updated to x_new above? No, we used x_mean in calculation.
                # Let's be careful. 
                # x_new is the new mean. x_mean is the old mean.
                # y = (x_mean - x_new) / sigma
                # v_i = (pop[i] - x_mean) / sigma
                
                # Calculate the rank-mu term
                # sum_{i=1}^mu w_i * (v_i - y) (v_i - y)^T ? 
                # Actually, the formula is:
                # C = (1 - c1 - cmu) * C + c1 * y y^T + cmu * sum w_i v_i v_i^T
                # where v_i = (x_i - x_mean_old) / sigma
                
                # We need to recompute v_i for the selected individuals
                v_terms = []
                for i in range(mu):
                    v_i = (pop[i] - x_mean) / sigma
                    v_terms.append(v_i)
                
                # Weighted sum of outer products
                rank_mu_term = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu_term += w[i] * np.outer(v_terms[i], v_terms[i])
                
                C = (1 - c1 - cmu) * C + c1 * np.outer(y, y) + cmu * rank_mu_term
                
            # Update mean
            x_mean = x_new
            
            # Ensure C is symmetric and PD
            C = (C + C.T) / 2
            # Check for PD
            try:
                np.linalg.cholesky(C)
            except:
                C += 1e-5 * np.eye(dim)
                
            # Stagnation check
            if fit[0] < best_f_history[-1] if best_f_history else True:
                last_improvement_call = calls_used
                best_f_history.append(fit[0])
            else:
                if calls_used - last_improvement_call > 5 * lam:
                    # Restart: Perturb sigma and C
                    sigma *= 2.0
                    C = 0.5 * C + 0.5 * np.eye(dim)
                    last_improvement_call = calls_used
                    best_f_history = [fit[0]]

        return best_x, best_f
