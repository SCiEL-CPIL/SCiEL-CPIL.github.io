"""
CMA-ES optimizer for BBOB f7 (Step ellipsoidal) in 5D.

Strategy: CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a
state-of-the-art derivative-free optimizer for black-box continuous optimization.
It adapts a multivariate normal distribution to the shape of the objective function.
For BBOB f7 (Step Ellipsoidal), which involves rotation and scaling, CMA-ES is
highly effective because it learns the covariance structure of the landscape,
allowing it to navigate the rotated ellipsoidal contours efficiently.

The implementation uses a standard CMA-ES with adaptive step size and covariance
matrix updates, along with restarts to handle potential local minima or flat regions
typical of step-like functions.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    CMA-ES is well-suited for non-smooth, rotated, and scaled functions like the
    step ellipsoidal. It maintains a population of samples drawn from a multivariate
    normal distribution, adapting both the mean and the covariance matrix based on
    the fitness of the selected individuals.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(N))
        lam = max(4 + int(3 * np.log(dim)), 10)
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size (box is [-5, 5])
        C = np.eye(dim)  # Covariance matrix
        pc = np.zeros(dim)  # Evolution path for covariance
        ps = np.zeros(dim)  # Evolution path for step size
        
        # Learning rates
        mu = lam // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Hessian update parameters
        h_sigma = (1 + 2 / max(dim - 1, 1) + c_sigma) * chi_n
        h_c = (1 + 2 / max(dim - 1, 1) + c_c) * chi_n

        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial population
        # We evaluate the initial mean first to have a baseline
        # But standard CMA-ES starts with a population.
        # Let's initialize population around 0.
        
        # Main loop
        gen = 0
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Sample population
            # Use Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(dim)
                L = np.eye(dim)
            
            z = np.random.randn(lam, dim)
            pop = x_mean + sigma * (z @ L.T)
            
            # Clip to box [-5, 5]
            pop = np.clip(pop, -5, 5)
            
            # Evaluate population
            fitness = np.empty(lam)
            for i in range(lam):
                if calls_made >= budget:
                    break
                fitness[i] = evaluate(pop[i])
            
            if calls_made >= budget:
                break
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Update mean
            x_new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update step size (sigma)
            # Inverse square root of det(C) is needed for the path
            # Approximation: use the determinant of C
            # More efficient: update ps using the inverse of the square root of C
            # For simplicity and robustness in 5D, we can compute the inverse square root
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C).T)
            except:
                C_inv_sqrt = np.eye(dim)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new_mean - x_mean) / sigma
            
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * h_sigma:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x_new_mean - x_mean) / sigma
            
            delta_h = 1 if np.linalg.norm(pc) / chi_n < (1.4 + 2 / (dim + 1)) * h_c else 0
            
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = C_inv_sqrt @ (pop[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small identity if needed for numerical stability
            C += 1e-12 * np.eye(dim)
            
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)
            
            gen += 1
            
            # Restart if sigma becomes too small or too large, or if we stagnate
            if sigma < 1e-8 or sigma > 100:
                # Restart around current best
                x_mean = best_x.copy()
                sigma = 1.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                # Reset some budget if we just restarted? No, budget is global.
                # But we need to make sure we don't loop forever on a dead end.
                # The time guard and budget guard handle this.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
