"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with multi-start for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        global_best_x = np.zeros(dim)
        global_best_f = float('inf')
        calls_used = 0
        
        def update_best(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()
        
        def time_up():
            return time.time() > deadline
        
        def budget_up():
            return calls_used >= total_budget

        # CMA-ES implementation
        def cmaes(x0, sigma0, max_iter):
            nonlocal calls_used, global_best_x, global_best_f
            
            # CMA-ES parameters
            mu = max(2, dim // 2)  # number of parents
            lambda_ = 2 * mu  # population size
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            x = x0.copy()
            sigma = sigma0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            step_size_history = []
            
            for it in range(max_iter):
                if budget_up() or time_up():
                    break
                
                # Generate lambda_ offspring
                offspring = []
                fitness = []
                
                for k in range(lambda_):
                    if budget_up() or time_up():
                        break
                    
                    # Standard normal
                    z = np.random.standard_normal(dim)
                    # Transform using C = B * diag(D^2) * B^T
                    # sqrt(C) = B * diag(D)
                    y = B @ (D * z)
                    x_new = x + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    
                    f_new = func(x_new)
                    calls_used += 1
                    update_best(x_new, f_new)
                    
                    offspring.append(x_new)
                    fitness.append(f_new)
                
                if len(offspring) < mu:
                    break
                
                # Sort by fitness
                idx = np.argsort(fitness)
                offspring = [offspring[i] for i in idx]
                fitness = [fitness[i] for i in idx]
                
                # Update mean
                x_old = x.copy()
                for i in range(mu):
                    x += weights[i] * (offspring[i] - x_old)
                
                # Update covariance matrix
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_sigma) * c_sigma) * np.eye(dim)) + c_mu * sum(weights * (y @ y.T))
                # This is expensive; we'll use a simplified update or skip some iterations
                # For efficiency, let's use a simpler approach: just update the mean and sigma
                # Actually, let's do a proper CMA-ES but with limited updates
                
                # Update evolution paths
                ys = [(offspring[i] - x_old) / sigma for i in range(mu)]
                pc = (1 - c_c) * pc + c_c / np.sqrt(c_c * (2 - c_c) * mueff) * (B @ (D * (weights @ np.array(ys))))
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B @ (D * (weights @ np.array(ys))))
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)
                
                # Update covariance matrix (simplified for speed)
                # We'll update C periodically
                if it % 5 == 0:
                    C = (1 - c_1 - c_mu) * C
                    C += c_1 * (pc @ pc.T + (1 - (1 - c_sigma) * c_sigma) * np.eye(dim))
                    for i in range(mu):
                        C += c_mu * weights[i] * np.outer(ys[i], ys[i])
                    
                    # Ensure positive semi-definite
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    
                    # Update B and D
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-10))
                    B = B.T  # eigh returns eigenvectors as columns, we need rows for our transformation
        
        # Multi-start CMA-ES
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(3):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Shuffle starts
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]
        
        # Run CMA-ES from each start
        for x0 in starts:
            if budget_up() or time_up():
                break
            
            remaining = total_budget - calls_used
            max_iter = max(10, remaining // 10)
            sigma0 = 2.0
            
            cmaes(x0, sigma0, max_iter)
            
            # If we have budget left, restart from the best point found
            if not budget_up() and not time_up():
                remaining = total_budget - calls_used
                if remaining > 50:
                    max_iter = max(10, remaining // 10)
                    cmaes(global_best_x, 1.0, max_iter)
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
