"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive budget allocation and multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        rng = np.random.RandomState(42)

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def cma_es(x0, f0, max_iter=200, pop_size=None):
            nonlocal calls_made, best_x, best_f
            
            if pop_size is None:
                pop_size = max(4, int(4 + 3 * np.log(dim)))
            
            # CMA-ES parameters
            sigma = 0.5
            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + 4 + dim / (dim + 4))
            cmu = min(0.5, (2 * (dim - 2) + 1 / (dim + 2)) / ((dim + 2) * (dim + 2) + 2))
            
            # Initialize
            mean = x0.copy()
            best_local_x = x0.copy()
            best_local_f = f0
            
            # Covariance matrix
            C = np.eye(dim)
            sqrtC = np.eye(dim)
            
            # Step sizes
            dsigma = 1 + 2 * max(0, cmu - 0.5) + cmu
            dlambda = 1 + 2 * max(0, cmu - 0.5) + cmu
            
            # Evolution paths
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            
            # Chi-squared correction
            chi_mu = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Initial population
            indices = np.arange(pop_size)
            half = pop_size // 2
            
            # Evaluate initial population
            pop_x = np.zeros((pop_size, dim))
            pop_f = np.zeros(pop_size)
            
            # First point is the mean
            pop_x[0] = mean
            pop_f[0] = f0
            
            for i in range(1, pop_size):
                z = rng.standard_normal(dim)
                pop_x[i] = mean + sigma * (sqrtC @ z)
                pop_f[i] = safe_call(pop_x[i])
            
            # Main loop
            for it in range(max_iter):
                if calls_made >= budget - 10 or time.time() > deadline:
                    break
                
                # Sort by function value
                order = np.argsort(pop_f)
                pop_x = pop_x[order]
                pop_f = pop_f[order]
                
                # Update best
                if pop_f[0] < best_local_f:
                    best_local_f = pop_f[0]
                    best_local_x = pop_x[0].copy()
                
                # Check convergence
                if np.std(pop_f[:half]) < 1e-12 * (1 + abs(pop_f[0])):
                    break
                
                # Weighted recombination
                weights = np.zeros(pop_size)
                weights[:half] = np.log(half + 0.5) - np.log(np.arange(1, half + 1))
                weights /= weights.sum()
                mueff = 1 / np.sum(weights ** 2)
                
                # Update mean
                old_mean = mean.copy()
                mean = np.sum(weights[:, np.newaxis] * pop_x, axis=0)
                
                # Update evolution paths
                # Standard deviation of the weighted sample
                diff = (mean - old_mean) / (sigma * np.sqrt(np.sum(weights ** 2)))
                ps = (1 - dsigma) * ps + np.sqrt(dsigma * (2 - dsigma) * mueff / dim) * diff
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - dsigma) ** (2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * chi_mu
                pc = (1 - cmu) * pc + hsig * cmu * (diff / np.linalg.norm(diff) * np.sqrt(mueff) if np.linalg.norm(diff) > 0 else np.zeros(dim))
                
                # Update covariance matrix
                C = (1 + cmu * (2 - cmu) * (1 - hsig) - cmu) * C
                rank_mu = np.outer(mean - old_mean, mean - old_mean) / np.sum(weights ** 2)
                rank_one = np.outer(diff, diff)
                C += cmu * (rank_mu + (1 - hsig) * rank_one)
                
                # Update sigma
                sigma *= np.exp((dsigma / 5 * dim) * (np.linalg.norm(ps) / chi_mu - 1))
                sigma = np.clip(sigma, 1e-10, 100)
                
                # Update sqrt(C)
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.clip(eigenvalues, 1e-12, None)
                    sqrtC = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
                except:
                    sqrtC = np.eye(dim)
                
                # Generate new population
                for i in range(1, pop_size):
                    z = rng.standard_normal(dim)
                    pop_x[i] = mean + sigma * (sqrtC @ z)
                    pop_f[i] = safe_call(pop_x[i])
                
                # Keep the best point
                pop_x[0] = best_local_x
                pop_f[0] = best_local_f
            
            return best_local_x, best_local_f

        # Multi-start CMA-ES
        # Phase 1: Quick exploration with multiple starts
        num_starts = 6
        starts = []
        start_fvals = []
        
        # Center
        starts.append(np.zeros(dim))
        
        # Diverse random points
        for i in range(num_starts - 1):
            x = rng.uniform(-4, 4, dim)
            starts.append(x)
        
        # Evaluate all starts
        for x in starts:
            f = safe_call(x)
            start_fvals.append(f)
        
        # Sort starts by function value
        start_order = np.argsort(start_fvals)
        
        # Phase 2: Run CMA-ES from the best few starts
        # Allocate budget based on remaining calls
        remaining = budget - calls_made
        if remaining > 50:
            # Run CMA-ES from the best start
            best_start = starts[start_order[0]]
            best_f_start = start_fvals[start_order[0]]
            best_x, best_f = cma_es(best_start, best_f_start, max_iter=150)
        
        # Phase 3: If budget remains and we're not close to optimum, try second best start
        if best_f > 1 and calls_made < budget - 50:
            second_start = starts[start_order[1]]
            second_f = start_fvals[start_order[1]]
            if second_f < best_f:
                second_x, second_f = cma_es(second_start, second_f, max_iter=100)
                if second_f < best_f:
                    best_x = second_x
                    best_f = second_f
        
        return best_x, best_f
# EVOLVE-BLOCK-END
