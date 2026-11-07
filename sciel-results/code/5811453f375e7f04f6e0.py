"""Optimizer for BBOB-style f15 (Rastrigin rotated) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Strategy: CMA-ES with multiple restarts
        # CMA-ES adapts covariance matrix to handle rotated landscapes better than (1+1)-ES
        # We use multiple restarts with decreasing budgets to explore diverse basins
        
        # Determine number of restarts based on budget
        # First restart gets a larger portion to find a good basin
        # Subsequent restarts get smaller portions for refinement/exploration
        n_restarts = 4
        # Budget allocation: 50%, 25%, 15%, 10% approximately
        restart_budgets = [
            int(budget * 0.5),
            int(budget * 0.25),
            int(budget * 0.15),
            int(budget * 0.1)
        ]
        # Ensure sum doesn't exceed budget
        total_alloc = sum(restart_budgets)
        if total_alloc > budget:
            restart_budgets[-1] = budget - sum(restart_budgets[:-1])
        
        # Initial points for each restart
        init_points = [np.zeros(dim)]
        for i in range(n_restarts - 1):
            init_points.append(rng.uniform(-4, 4, dim))

        for r in range(n_restarts):
            if time.time() > deadline or calls_made >= budget:
                break
            
            restart_budget = restart_budgets[r]
            if restart_budget < 10:
                continue
            
            x0 = init_points[r].copy()
            fx0 = func(x0)
            calls_made += 1
            
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            
            # CMA-ES parameters
            sigma0 = 2.0  # Initial step size
            pop_size = 12  # Population size (lambda)
            mu = pop_size // 2  # Number of best solutions for recombination
            
            # CMA-ES state
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)  # Covariance matrix
            C_inv_sqrt = np.eye(dim)  # Precomputed for efficiency
            
            # Weight vectors for recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / weights.sum()
            
            # Adaptation rates
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            
            # Path accumulators
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Generate initial population
            def generate_population(mean, sigma, C):
                # Compute sqrt of C for sampling
                # Use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                
                # Generate normal samples
                z = rng.standard_normal((pop_size, dim))
                # Transform to C distribution
                samples = mean + sigma * (z @ L.T)
                return samples, L
            
            # Main CMA-ES loop
            max_iter = restart_budget // pop_size
            for iteration in range(max_iter):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Generate population
                samples, L = generate_population(mean, sigma, C)
                
                # Evaluate population
                fvals = []
                for i in range(pop_size):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    x_i = np.clip(samples[i], -5, 5)
                    f_i = func(x_i)
                    calls_made += 1
                    fvals.append((f_i, i))
                
                if len(fvals) < pop_size:
                    break
                
                # Sort by fitness
                fvals.sort(key=lambda x: x[0])
                selected_indices = [idx for _, idx in fvals[:mu]]
                selected_samples = samples[selected_indices]
                selected_fvals = [f for f, _ in fvals[:mu]]
                
                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * selected_samples, axis=0)
                
                # Update path sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (new_mean - mean)
                
                # Update path C
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * L @ ((new_mean - mean) / sigma)
                
                # Update covariance
                outer = np.outer(p_c, p_c)
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * np.trace(C) / dim else 0.0
                C = (1 + c1 * h_sigma * (1 - h_sigma) + cmu) * C - c1 * outer
                # Add rank-mu update
                diff = (selected_samples - mean) / sigma
                C += cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->jk', weights, diff, diff), axis=0)
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                # Add small regularization if needed
                C += 1e-10 * np.eye(dim)
                
                # Update sigma
                sigma = sigma * np.exp(cs / (dim + 3) * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
                
                # Update mean
                mean = new_mean
                
                # Track best
                if selected_fvals[0] < best_f:
                    best_f = selected_fvals[0]
                    best_x = np.clip(selected_samples[0], -5, 5).copy()
            
            # After CMA-ES, do a local refinement with smaller steps
            if time.time() < deadline and calls_made < budget:
                x = best_x.copy() if best_x is not None else mean.copy()
                fx = best_f
                sigma_local = 0.5
                remaining = budget - calls_made
                for _ in range(min(remaining, 20)):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    y = x + sigma_local * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma_local *= 0.8

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
