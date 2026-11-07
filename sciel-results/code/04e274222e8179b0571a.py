"""Surrogate-guided CMA-ES hybrid for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-to-local strategy: random exploration then CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_fx = np.inf
        calls_used = 0

        def time_ok():
            return time.time() < deadline - 3.0 and calls_used < budget

        def safe_call(x):
            nonlocal calls_used
            if not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            try:
                fx = func(x)
                fx = float(fx)
            except Exception:
                return None
            calls_used += 1
            return x.copy(), fx

        # Phase 1: Global exploration - 50 uniform random samples
        n_global = min(50, budget)
        candidates = []
        for _ in range(n_global):
            x = rng.uniform(-5, 5, dim)
            result = safe_call(x)
            if result is None:
                break
            x_c, fx = result
            candidates.append((fx, x_c))

        if not candidates:
            result = safe_call(np.zeros(dim))
            if result:
                return result[0], result[1]
            return np.zeros(dim), 0.0

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])
        best_x = candidates[0][1].copy()
        best_fx = candidates[0][0]

        # Phase 2: CMA-ES from the best point found
        # Initialize CMA-ES with parameters tuned for 5D
        x = best_x.copy()
        sigma = 1.0  # initial step size
        pop_size = 4 + int(3 * np.log(dim))  # population size for dim=5: ~8
        pop_size = max(4, min(pop_size, budget - calls_used - 10))
        
        # CMA-ES state
        C = np.eye(dim)  # covariance matrix
        ps = np.zeros(dim)  # evolution path for sigma
        pc = np.zeros(dim)  # evolution path for C
        cs = 1.0 / (pop_size + 3)  # sigma adaptation time constant
        ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
        cmu = 2 / ((dim + 1)**2 + pop_size)  # covariance adaptation time constant
        cc = (4 + dim / pop_size) / (dim + 4 + 2 * pop_size / dim)  # path time constant
        c1 = cmu  # rank-1 update weight
        cmu_cov = 2 / ((dim + 1.7)**2 + pop_size)  # rank-mu update weight
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        # Generate initial population
        def generate_population(x, sigma, C, pop_size, rng):
            B, d = np.linalg.eigh(C)
            B = B * np.sqrt(np.maximum(d, 1e-10))
            zs = rng.standard_normal((pop_size, dim))
            return x + sigma * (zs @ B.T)

        # Sort indices for ranking
        def sort_indices(fitnesses):
            return np.argsort(fitnesses)

        # CMA-ES main loop
        while time_ok() and calls_used < budget:
            # Generate population
            zs = generate_population(x, sigma, C, pop_size, rng)
            zs = np.clip(zs, -5, 5)
            
            # Evaluate population
            fitnesses = []
            for i in range(pop_size):
                result = safe_call(zs[i])
                if result is None:
                    fitnesses.append(np.inf)
                else:
                    fitnesses.append(result[1])
            
            fitnesses = np.array(fitnesses)
            
            # Sort
            sorted_idx = sort_indices(fitnesses)
            sorted_zs = zs[sorted_idx]
            sorted_fitnesses = fitnesses[sorted_idx]
            
            # Update best
            if sorted_fitnesses[0] < best_fx:
                best_fx = sorted_fitnesses[0]
                best_x = sorted_zs[0].copy()
            
            # Update mean
            weights = np.ones(pop_size) / pop_size
            x = np.sum(sorted_zs * weights[:, np.newaxis], axis=0)
            
            # Update evolution paths
            diff = (x - best_x) / sigma if sigma > 1e-10 else np.zeros(dim)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * pop_size) * diff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n) < (1.4 + 2 / (dim + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * pop_size) * diff
            
            # Update covariance
            delta_hs = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(min(3, pop_size)):  # use top 3 for rank-mu
                wi = weights[sorted_idx[i]]
                vi = sorted_zs[i] - best_x
                rank_mu += wi * np.outer(vi, vi)
            
            C = (1 + c1 * delta_hs - c1 - cmu_cov) * C + c1 * rank_one + cmu_cov * rank_mu
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(1e-8, min(sigma, 10.0))
            
            # Check for convergence
            if np.linalg.norm(sigma * np.sqrt(np.diag(C))) < 1e-8:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
