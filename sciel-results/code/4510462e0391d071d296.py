import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315  # 15s safety margin from 330s limit
        calls = 0
        max_calls = self.budget
        best_x = None
        best_f = np.inf
        dim = self.dim

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time Limit")
            
            calls += 1
            x = np.asarray(x, dtype=float)
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # CMA-ES Parameters
        # Population size: 4 + floor(3 * ln(dim)) is standard, but for dim=5, let's use a slightly larger pop for robustness
        # lambda = 4 + 3*ln(5) ~ 9. Let's use 10 for even number and stability.
        # Actually, for dim=5, lambda=10 is reasonable.
        lambda_ = 10
        
        # Initial standard deviation
        # Box is [-5, 5]. Center 0.
        # Standard deviation should cover the box reasonably well.
        # sigma = 1.0 is a good start for [-5,5] (covers ~99.7% within +/-3).
        sigma = 1.0
        
        # Mean
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Step size control (CSA - Cumulative Step Adaptation)
        # Parameters from Hansen's CMA-ES
        mu = lambda_ // 2  # Number of best individuals used for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Eigen decomposition cache (update every 1-5 generations)
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)
        
        # Generation counter
        gen = 0
        
        # Initial population evaluation
        # We need to generate lambda samples from N(m, sigma^2 * C)
        
        def sample_cma(m, sigma, C, n_samples):
            # Cholesky decomposition of C
            # Since C is updated via rank-1/rank-mu updates, it remains symmetric positive definite
            # but numerical errors might creep in. We can use eigendecomposition for sampling if needed,
            # or Cholesky. Cholesky is faster.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigendecomposition if Cholesky fails
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip negative eigenvalues
                eigvals = np.clip(eigvals, 1e-10, None)
                L = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                # Actually, for sampling x = m + sigma * L @ z, where z ~ N(0, I)
                # If C = V D V^T, then L can be V D^0.5
                # So x = m + sigma * V D^0.5 z
                # Let's stick to the standard transformation
                pass
            
            z = np.random.randn(n_samples, dim)
            # Transform z to standard normal with covariance C
            # x = m + sigma * L @ z
            # Note: L is lower triangular such that L @ L.T = C
            # So (L @ z) has covariance L @ I @ L.T = C
            samples = m + sigma * (z @ L.T)
            return samples

        # Initial population
        pop = sample_cma(m, sigma, C, lambda_)
        pop_fitness = np.full(lambda_, np.inf)
        
        for i in range(lambda_):
            try:
                pop_fitness[i] = wrapper(pop[i])
            except (ValueError, TimeoutError):
                break

        # If we couldn't evaluate the initial population, return best so far
        if np.any(np.isinf(pop_fitness)):
            if best_x is not None:
                return best_x, best_f
            else:
                return np.zeros(dim), np.inf

        # Sort population by fitness
        # We will maintain the population sorted or re-sort each generation
        # For CMA-ES, we select the best mu individuals
        
        # Main Loop
        while True:
            # Check time and budget
            if calls >= max_calls or time.time() > deadline:
                break
            
            # Check convergence
            # If sigma is too small or best fitness hasn't improved significantly
            # Simple convergence criterion: sigma < 1e-12 or max-min fitness < 1e-12
            # But we want to use the full budget if possible.
            # Let's just run until budget/time is up, as CMA-ES is robust.
            
            # 1. Sort population by fitness
            idx = np.argsort(pop_fitness)
            pop = pop[idx]
            pop_fitness = pop_fitness[idx]
            
            # 2. Select best mu individuals
            y_best = pop[:mu]
            
            # 3. Compute weighted mean (new mean)
            m_old = m
            m = np.sum(weights[:, np.newaxis] * y_best, axis=0)
            
            # 4. Update step size control (CSA)
            # h_sigma condition
            # E[||N(0, I)||] = sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim^2))
            # Approximation: sqrt(2 * (dim + 1) / (dim + 2)) * ... 
            # Standard formula:
            # phi = sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim^2))
            # h_sigma = 1 if ||p_sigma|| / (1 - (1-c_sigma)^{gen+1}) < (1.4 + 2/(dim+1)) * phi
            # We need to track the norm of the path p_sigma
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{-1/2} * (m - m_old) / sigma
            
            # Compute C^{-1/2}
            # We have eigvecs and eigvals from previous update (or initial)
            # It's better to update eigendecomposition periodically.
            # Let's update eigendecomposition every 5 generations for efficiency
            if gen % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
            
            C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            
            delta_m = (m - m_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ delta_m)
            
            # h_sigma
            phi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
            # The denominator is 1 - (1 - c_sigma)^(gen+1)
            # But we can approximate with 1 - (1 - c_sigma) if gen is large, or just use the standard check
            # Standard check: ||p_sigma|| < (1.4 + 2/(dim+1)) * phi
            # This check is usually done on the normalized path.
            # Let's use the simpler criterion:
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2/(dim+1)) * phi else 0.0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / phi - 1))
            # Wait, the formula is sigma = sigma * exp(c_sigma * (||p_sigma||/phi - 1))
            # Let's correct:
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / phi - 1))
            
            # 5. Update covariance matrix
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (m - m_old) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_m
            
            # Rank-1 update
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * I)
            # Rank-mu update
            # C += c_mu * sum(wi * yi * yi.T) where yi = (yi - m_old)/sigma
            
            # Let's compute the rank-mu term
            y_scaled = (y_best - m_old) / sigma
            # C_mu = sum(wi * yi * yi.T)
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                C_mu += weights[i] * np.outer(y_scaled[i], y_scaled[i])
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * np.eye(dim)) + c_mu * C_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # 6. Generate new population
            # Check if sigma is too small
            if sigma < 1e-12:
                # Converged
                break
                
            # Sample new population
            try:
                pop = sample_cma(m, sigma, C, lambda_)
            except:
                # Fallback if Cholesky fails
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                z = np.random.randn(lambda_, dim)
                pop = m + sigma * (z @ L.T)
            
            # Evaluate new population
            pop_fitness = np.full(lambda_, np.inf)
            for i in range(lambda_):
                try:
                    pop_fitness[i] = wrapper(pop[i])
                except (ValueError, TimeoutError):
                    break
            
            # If we ran out of budget during evaluation, break
            if calls >= max_calls or time.time() > deadline:
                break
            
            gen += 1

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except:
                best_f = np.inf

        return best_x, best_f
