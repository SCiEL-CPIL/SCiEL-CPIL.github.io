import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Katsuura function (f23) on [-5, 5]^5.
    Uses a population-based approach with covariance adaptation to handle
    the high multimodality of the function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize CMA-ES parameters
        # Initial center: random point in the box
        m = rng.uniform(-4, 4, dim)
        # Initial sigma: large enough to cover the box
        sigma = 2.0
        
        # Population size
        lambda_ = 20
        mu = lambda_ // 2
        
        # Weights (standard weights)
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Variance-effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = np.sqrt((1 + c_sigma) * (2 - c_sigma) * (mu_eff + 2) / (dim + 2))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim**2))
        
        # First evaluation
        evaluate(m)
        
        while calls_made < budget and remaining_time() > 2.0:
            # Generate population
            # Cholesky decomposition of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(dim)
                B = np.eye(dim)
            
            # Generate N(0, I)
            z = rng.standard_normal((lambda_, dim))
            
            # Transform to N(m, sigma^2 * C)
            pop = np.array([m + sigma * (B @ z[i]) for i in range(lambda_)])
            
            # Evaluate population
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if calls_made >= budget or remaining_time() < 1.0:
                    break
                fvals[i] = evaluate(pop[i])
            
            # If we ran out of budget, break
            if calls_made >= budget or remaining_time() < 1.0:
                break
            
            # Sort population by fitness
            order = np.argsort(fvals)
            selected_pop = pop[order]
            selected_fvals = fvals[order]
            
            # Update mean
            m_new = np.sum(weights[:, None] * selected_pop[:mu], axis=0)
            
            # Update covariance matrix
            # Difference matrix
            diff = selected_pop[:mu] - m
            # Weighted outer product
            C_new = c_1 * (1 + (1 - c_1) * c_c * (1 - c_c)) * C + \
                    c_mu * np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            
            # Update path states
            # Need B_inv for ps update
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                B_inv = np.eye(dim)
            
            ps = (1 - c_sigma) * ps + c_sigma * cs * (B_inv @ (m_new - m) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2 / (dim + 1)) * chi_n
            
            pc = (1 - c_c) * pc + hsig * c_c * (m_new - m) / sigma
            
            # Update C with rank-one correction
            if hsig:
                C_new += (1 + (1 - c_1) * c_c * (1 - c_c) - c_mu) * np.outer(pc, pc)
            
            # Ensure C is symmetric and positive semi-definite
            C_new = (C_new + C_new.T) / 2
            # Add small regularization to ensure PD
            C_new += 1e-9 * np.eye(dim)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update state
            m = m_new
            C = C_new
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-4, 10.0)
            
            # If sigma becomes too small, we might be stuck. 
            # For Katsuura, this is likely. We could restart, but let's see if CMA-ES handles it.
            # A simple restart strategy: if sigma < 1e-3, reset C and sigma but keep m.
            if sigma < 1e-3:
                C = np.eye(dim)
                sigma = 0.1
                pc = np.zeros(dim)
                ps = np.zeros(dim)

        return best_x, best_f
