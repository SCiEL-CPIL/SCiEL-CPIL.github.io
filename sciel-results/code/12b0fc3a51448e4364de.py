"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Initialize population size
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))
        
        # CMA-ES parameters
        mu = lambda_pop // 2  # number of recombination weights
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Step size control parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 1.7) + c_sigma) / ((dim + 2) * (2 + mueff / (dim + 1.7) - c_sigma))
        c_mu = (mueff / (dim + 2) - c_1) * c_sigma * (2 - c_sigma)
        
        # Initial step size
        sigma = 2.0
        cc = c_c
        cs = c_sigma
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initial mean
        m = rng.uniform(-5, 5, dim)
        fm = func(m)
        
        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for square root
        D, B = np.linalg.eigh(C)
        D = np.sqrt(D)
        # C^{-1/2}
        C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # History for best value tracking
        best_f = fm
        best_x = m.copy()
        
        calls_made = 1
        while calls_made < budget:
            # Generate population
            population = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = m + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                population.append(y)
            
            # Evaluate population
            fitness = []
            for y in population:
                fy = func(y)
                fitness.append(fy)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            # Sort by fitness
            indices = np.argsort(fitness)
            population = [population[i] for i in indices]
            fitness = [fitness[i] for i in indices]
            
            # Check if budget is exhausted
            if calls_made >= budget:
                break
            
            # Mean of selected points
            m_old = m.copy()
            m = sum(weights[i] * population[i] for i in range(mu))
            
            # Update evolution path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (m - m_old) / sigma
            
            # h_sigma condition
            hs = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made))) / chiN < 1.4 + 2 / (dim + 1)
            
            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + hs * np.sqrt(c_c * (2 - c_c) * mueff) * C_inv_sqrt @ (m - m_old)
            
            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(weights[i] * np.outer(population[i] - m_old, population[i] - m_old) for i in range(mu))
            C = (1 + c_sigma * c_mu * (2 - c_mu) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive semi-definiteness and recompute sqrt
            try:
                D, B = np.linalg.eigh(C)
                D = np.clip(D, 1e-12, None)
                D = np.sqrt(D)
                C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
            except:
                C = np.eye(dim)
                D, B = np.linalg.eigh(C)
                D = np.sqrt(D)
                C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
            
            # Termination criteria
            if sigma < 1e-12:
                break
            if np.all(np.abs(m) < 1e-12) and sigma < 1e-6:
                break
            
            # Check if population has converged
            if np.ptp([np.linalg.norm(y - m) for y in population[:mu]]) < 1e-10:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
