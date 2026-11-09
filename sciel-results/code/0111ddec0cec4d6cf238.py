import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 290.0  # Safety margin
        n = self.dim
        budget = self.budget
        
        # Helper to evaluate and track
        def evaluate(x):
            if self.calls >= budget or time.time() > deadline:
                return self.best_f
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x_clipped.copy()
            return val

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Initial step size
        sigma = 2.0 # Start reasonably wide for [-5,5]
        
        # Initial mean
        # Random start
        rng = np.random.default_rng(42)
        mean = rng.uniform(-5, 5, n)
        # Evaluate initial mean
        evaluate(mean)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Evolution paths
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Update parameters
        c_sigma = (mueff + 2) / ((n + 2) * (mueff + 0.5))
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # Chi-squared distribution approximation for sigma update
        chi_n = n * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        max_iter = 1000 # Hard cap on generations, will be limited by budget
        stagnation_count = 0
        best_f_history = []
        
        for gen in range(max_iter):
            if self.calls >= budget or time.time() > deadline:
                break
                
            # Generate offspring
            z = np.zeros((lam, n))
            for i in range(lam):
                # Sample from multivariate normal
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset or jitter
                    C = C + 1e-6 * np.eye(n)
                    L = np.linalg.cholesky(C)
                
                z[i] = mean + sigma * (L @ rng.standard_normal(n))
            
            # Evaluate offspring
            fvals = np.zeros(lam)
            for i in range(lam):
                fvals[i] = evaluate(z[i])
                if self.calls >= budget or time.time() > deadline:
                    break
            
            # Sort offspring by fitness
            idx = np.argsort(fvals)
            z = z[idx]
            fvals = fvals[idx]
            
            # Update mean
            new_mean = np.sum(weights[:, None] * z[:mu], axis=0)
            
            # Update sigma
            # Calculate norm of (new_mean - mean)
            delta_mean = new_mean - mean
            norm_delta = np.linalg.norm(delta_mean)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * delta_mean
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(chi_n) - 1))
            # Limit sigma
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(chi_n))
            pc = (1 - c_c) * pc + float(hsig) * np.sqrt(c_c * (2 - c_c) * mueff) * delta_mean / sigma
            
            # Update covariance C
            delta_pc = (hsig - 1) * c_c * (2 - c_c) * pc
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + \
                c_1 * (pc[:, None] * pc[None, :] + delta_pc * pc[:, None] + delta_pc * pc[None, :] + (1 - delta_pc) * np.outer(pc, pc)) + \
                c_mu * np.sum(weights[:mu, None, None] * (z[:mu, :, None] - new_mean[:, None]) * (z[:mu, None, :] - new_mean[None, :]), axis=0)
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-9 * np.eye(n)
            
            # Update mean
            mean = new_mean
            
            # Check stagnation
            if len(best_f_history) > 10:
                recent = best_f_history[-10:]
                if max(recent) - min(recent) < 1e-8:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
            best_f_history.append(self.best_f)
            
            if stagnation_count > 20:
                break

        # Local refinement with Nelder-Mead
        if self.calls < budget - 10 and time.time() < deadline - 10:
            x0 = self.best_x.copy()
            local_calls = [0]
            max_local_calls = min(50, budget - self.calls)
            
            def local_wrapper(x):
                if local_calls[0] >= max_local_calls or self.calls >= budget or time.time() > deadline:
                    return self.best_f
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                self.calls += 1
                local_calls[0] += 1
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = x_clipped.copy()
                return val
            
            try:
                # Nelder-Mead is robust for polishing
                res = scipy.optimize.minimize(
                    local_wrapper,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12}
                )
            except Exception:
                pass

        return self.best_x, self.best_f
