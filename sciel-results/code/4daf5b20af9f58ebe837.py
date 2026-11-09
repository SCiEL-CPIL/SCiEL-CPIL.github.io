"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with LHS initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Latin Hypercube Sampling for initial spread
        # This provides a good initial guess and covers the space well.
        n_lhs = 50
        if budget <= n_lhs:
            # Fallback if budget is very small
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            return x, fx

        # Generate LHS points
        # For each dimension, divide the range into n_lhs bins and pick one point per bin
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Create bins
            bins = np.linspace(-5, 5, n_lhs + 1)
            # Randomly select one point from each bin
            for i in range(n_lhs):
                lhs_points[i, d] = rng.uniform(bins[i], bins[i + 1])
        
        # Evaluate LHS points
        best_x = None
        best_f = np.inf
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            fx = func(lhs_points[i])
            if fx < best_f:
                best_f = fx
                best_x = lhs_points[i].copy()
        
        calls_made = n_lhs
        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        # Phase 2: CMA-ES
        # Initialize CMA-ES parameters
        x_mean = best_x
        f_mean = best_f
        sigma = 1.0
        lambda_ = 4 + int(3 * np.log(dim))  # Population size
        mu = lambda_ // 2  # Number of parents
        
        # Weighting
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        D = 1.0  # Determinant of C^(1/2)
        
        # Precompute C^(1/2)
        # For efficiency, we can use Cholesky or Eigen decomposition
        # Since dim=5, Eigen is fast enough
        
        def get_chol(C):
            # Use Cholesky if possible, else Eigen
            try:
                return np.linalg.cholesky(C)
            except:
                vals, vecs = np.linalg.eigh(C)
                vals = np.clip(vals, 1e-12, None)
                return vecs @ np.diag(np.sqrt(vals))
        
        C_sqrt = get_chol(C)
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # CMA-ES Loop
        while calls_made < budget and time.time() < deadline:
            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                pop[i] = x_mean + sigma * (C_sqrt @ z)
                # Clip to bounds
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate population
            pop_f = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                pop_f[i] = func(pop[i])
                calls_made += 1
            
            # Sort by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update evolution paths
            # C_inv_sqrt = np.linalg.inv(C_sqrt)
            C_inv_sqrt = np.linalg.inv(C_sqrt)
            y_mean = (x_mean_new - x_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ y_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // lambda_))) <
                    (1 + 2 / (dim + 1)) * np.sqrt(mu_eff))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y_mean
            
            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            delta = delta_hsig * (C - C_sqrt @ np.outer(pc, pc) @ C_sqrt.T)  # Approximation, better to use full formula
            # Actually, standard update:
            # C = (1 - c1 - cmu + delta_hsig) * C + c1 * (pc @ pc.T + delta_hsig * C) + cmu * sum(wi * yi @ yi.T)
            
            # Let's stick to the standard update formula
            C_new = (1 - c1 - cmu + delta_hsig) * C
            C_new += c1 * (pc @ pc.T + delta_hsig * C)
            
            # Sum of outer products of selected steps
            sum_yi_yiT = np.zeros((dim, dim))
            for i in range(mu):
                yi = (pop[i] - x_mean) / sigma
                sum_yi_yiT += weights[i] * np.outer(yi, yi)
            
            C_new += cmu * sum_yi_yiT
            
            # Ensure symmetry and positive definiteness
            C_new = (C_new + C_new.T) / 2
            # Regularize if needed
            C = C_new
            
            # Update sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            # dsigma is the normalization constant for the norm of a standard normal
            dsigma = 1.0  # Approximation, usually close to 1 for small dim
            
            # Update C_sqrt
            C_sqrt = get_chol(C)
            
            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()
                
            x_mean = x_mean_new

        return best_x, best_f
# EVOLVE-BLOCK-END
