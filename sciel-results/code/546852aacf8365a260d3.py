"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation 
    optimized for BBOB f14 (Different powers).
    
    f14 has an anisotropic landscape due to different powers in each dimension.
    CMA-ES adapts a full covariance matrix to align the search distribution 
    with the principal axes of the function, making it highly effective here.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Initial sigma: 2.0 is a good guess for [-5,5] box
        sigma = 2.0
        # Population size: standard is 4 + floor(3 * ln(N))
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2  # Weights top mu individuals
        
        # Weights for rank-mu update
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # State variables
        mean = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Initial evaluation to set a baseline
        # We start with a few random points to get a decent initial mean
        # and to ensure we have a valid best_x immediately.
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Phase 1: Quick random initialization to find a reasonable starting point
        # This helps if the global minimum is far from the origin.
        n_init = min(10, budget)
        for _ in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is not None:
            mean = best_x.copy()
            # Estimate initial sigma based on distance from origin or just keep 2.0
            # If best_x is far, maybe sigma should be larger, but 2.0 is safe for [-5,5]
        
        if best_x is None:
            # Fallback if no valid call was made (should not happen)
            return np.zeros(dim), np.inf

        # Pre-compute Cholesky decomposition of C
        try:
            B = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B = np.eye(dim)

        def sample_from_cma():
            z = rng.standard_normal(dim)
            return mean + sigma * B @ z

        # Main CMA-ES Loop
        while calls_made < budget and time.time() < deadline:
            # Generate population
            pop = []
            for _ in range(lambda_):
                if calls_made >= budget or time.time() > deadline:
                    break
                x = sample_from_cma()
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                pop.append((fx, x))
            
            if not pop:
                break

            # Sort by fitness
            pop.sort(key=lambda p: p[0])
            
            # Update best
            if pop[0][0] < best_f:
                best_f = pop[0][0]
                best_x = pop[0][1].copy()

            # Select top mu
            selected = pop[:mu]
            
            # Weighted mean of selected
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * selected[i][1]
            
            # Update step-size control (sigma)
            y_old = (new_mean - mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * y_old
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0) # Clip to prevent numerical issues
            
            # Update covariance matrix
            # Rank-one update
            if np.linalg.norm(ps) < (1.4 + 0.5 / c_sigma) * chi_n:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_old
            
            # Rank-mu update
            y_weighted = np.zeros((dim, mu))
            for i in range(mu):
                y_weighted[:, i] = (selected[i][1] - mean) / sigma
            
            C += c_1 * (np.outer(pc, pc) + (1 - h_sigma * (1 - c_1)) * np.eye(dim) - C)
            
            # Rank-mu part
            # C += c_mu * sum(w_i * (y_i - y_bar) (y_i - y_bar)^T)
            # Simplified: use weighted outer products
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(y_weighted[:, i], y_weighted[:, i])
                
            # Ensure positive definiteness (simple repair)
            C = (C + C.T) / 2
            # Add small diagonal if needed
            C += 1e-12 * np.eye(dim)
            
            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Update Cholesky decomposition
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-positive definite, reset to identity (or damped)
                C = np.eye(dim)
                B = np.eye(dim)
                # Reset pc, ps to avoid bias
                pc = np.zeros(dim)
                ps = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
