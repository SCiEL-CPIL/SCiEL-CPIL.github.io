import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # CMA-ES Implementation
        # Parameters
        lambda_pop = max(4, int(np.ceil(3 + 3 * np.log(dim))))  # Population size
        mu = lambda_pop // 2  # Number of best individuals to update weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Step size
        sigma = 1.0  # Initial step size, tuned for [-5,5] box
        
        # Mean
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        C_eigen = np.linalg.eigh(C)
        C_eigenvalues = C_eigen[0]
        C_eigenvectors = C_eigen[1]
        
        # Inverse square root of C
        C_inv_sqrt = np.zeros((dim, dim))
        for i in range(dim):
            C_inv_sqrt[i, i] = 1.0 / np.sqrt(C_eigenvalues[i])
        C_inv_sqrt = C_eigenvectors @ C_inv_sqrt @ C_eigenvectors.T
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population evaluation
        def sample_population():
            nonlocal mean, sigma, C_inv_sqrt
            ys = rng.standard_normal((lambda_pop, dim))
            # Transform to x-space
            # x = mean + sigma * C^{1/2} y
            # Since we maintain C_inv_sqrt, we need C_sqrt
            # For simplicity in this lightweight implementation, we update C via rank-mu update
            # and use Cholesky or eigen for sampling if needed, but standard CMA-ES uses C_inv_sqrt for updates.
            # Actually, sampling requires C^{1/2}. 
            # Let's stick to a simpler variant: use the current C to generate samples.
            # We need C^{1/2}. We can compute it from eigenvalues/vectors.
            C_sqrt = C_eigenvectors @ np.diag(np.sqrt(np.maximum(C_eigenvalues, 1e-12))) @ C_eigenvectors.T
            xs = mean + sigma * (C_sqrt @ ys.T).T
            return xs

        # Initial population
        xs = sample_population()
        fs = np.zeros(lambda_pop)
        for i in range(lambda_pop):
            f = call_f(xs[i])
            if f is None:
                break
            fs[i] = f
        
        # Main loop
        while n_calls < budget - lambda_pop and time.time() < deadline - 5:
            if time.time() > deadline - 10:
                break
            
            # Sort by fitness
            idx = np.argsort(fs)
            xs = xs[idx]
            fs = fs[idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
            
            # Update mean
            mean = np.sum(weights[:, np.newaxis] * xs[:mu], axis=0)
            
            # Update covariance
            y = (mean - mean)  # This is 0, wait. 
            # Standard update:
            # y = (mean - old_mean) / sigma
            # We need old_mean. Let's store it.
            # Actually, let's restart the loop structure to be cleaner.
            
            pass # This logic is getting messy inline. Let's restructure.
            
            break # Break to rewrite properly

        # Rewriting CMA-ES loop properly
        
        # Reset state for clean loop
        mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        C_eigen = np.linalg.eigh(C)
        C_eigenvalues = C_eigen[0]
        C_eigenvectors = C_eigen[1]
        C_inv_sqrt = np.zeros((dim, dim))
        for i in range(dim):
            C_inv_sqrt[i, i] = 1.0 / np.sqrt(C_eigenvalues[i])
        C_inv_sqrt = C_eigenvectors @ C_inv_sqrt @ C_eigenvectors.T
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population
        def get_samples():
            C_sqrt = C_eigenvectors @ np.diag(np.sqrt(np.maximum(C_eigenvalues, 1e-12))) @ C_eigenvectors.T
            ys = rng.standard_normal((lambda_pop, dim))
            return mean + sigma * (C_sqrt @ ys.T).T

        xs = get_samples()
        fs = np.zeros(lambda_pop)
        for i in range(lambda_pop):
            f = call_f(xs[i])
            if f is None:
                return best_x, best_f
            fs[i] = f

        while n_calls < budget - lambda_pop and time.time() < deadline - 5:
            
            # 1. Sort
            idx = np.argsort(fs)
            xs = xs[idx]
            fs = fs[idx]
            
            # 2. Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                
            # 3. Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * xs[:mu], axis=0)
            
            # 4. Update path p_sigma
            y_sigma = (mean - old_mean) / (sigma * np.sqrt(1 - (1 - 2 / mueff) * (1 - np.exp(-2 / c_sigma))))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ y_sigma
            
            # 5. Update h_sigma (check if p_sigma is not too large)
            hs = np.sqrt((1 - np.exp(-2 / c_sigma)) * (1 - np.exp(-4 / c_sigma))) * (dim + 2) / dim
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (n_calls / lambda_pop + 1)))) < (1.4 + 2 / (dim + 1)) * hs else 0
            
            # 6. Update path p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * C_inv_sqrt @ (mean - old_mean) / sigma
            
            # 7. Update C
            # Rank-1 update
            C = (1 + c_1 * h_sigma * (h_sigma - 1) - c_1) * C + c_1 * np.outer(p_c, p_c)
            # Rank-mu update
            for i in range(mu):
                y_i = (xs[i] - old_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # 8. Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            
            # 9. Update C eigen decomposition
            C_eigen = np.linalg.eigh(C)
            C_eigenvalues = C_eigen[0]
            C_eigenvectors = C_eigen[1]
            
            # Ensure positive definiteness
            C_eigenvalues = np.maximum(C_eigenvalues, 1e-12)
            
            # Update C_inv_sqrt
            C_inv_sqrt = np.zeros((dim, dim))
            for i in range(dim):
                C_inv_sqrt[i, i] = 1.0 / np.sqrt(C_eigenvalues[i])
            C_inv_sqrt = C_eigenvectors @ C_inv_sqrt @ C_eigenvectors.T
            
            # 10. Sample new population
            xs = get_samples()
            fs = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                f = call_f(xs[i])
                if f is None:
                    return best_x, best_f
                fs[i] = f
                
        # Final polish with (1+1)-ES
        remaining = budget - n_calls
        if remaining > 5 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma_poly = 0.1
            for _ in range(remaining):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma_poly * rng.standard_normal(dim), -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma_poly *= 1.1
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                else:
                    sigma_poly *= 0.9
                    sigma_poly = max(sigma_poly, 1e-4)

        return best_x, best_f
