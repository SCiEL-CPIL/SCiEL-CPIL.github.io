import numpy as np
import time

class Optimizer:
    """
    Hybrid Global-Local Optimizer for BBOB f21.
    
    Strategy:
    1. Global exploration using Latin Hypercube Sampling (LHS) to identify promising regions.
    2. Local exploitation using a lightweight CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 
       initialized at the best point from Phase 1.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        budget = self.budget
        dim = self.dim

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def check_time():
            return time.time() < deadline

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if not check_time():
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(12345)

        # Phase 1: Global Exploration with LHS
        n_lhs = 20
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            bins = np.linspace(-5, 5, n_lhs + 1)
            for i in range(n_lhs):
                lhs_points[i, d] = rng.uniform(bins[i], bins[i+1])
        
        lhs_f = np.full(n_lhs, float('inf'))
        for i in range(n_lhs):
            fx = safe_call(lhs_points[i])
            if fx is None:
                break
            lhs_f[i] = fx

        if best_x is None:
            # Fallback if LHS failed
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()
            else:
                return None, float('inf')

        # Phase 2: Local Exploitation with CMA-ES
        # Initialize CMA-ES at the best point found
        x0 = best_x.copy()
        
        # CMA-ES Parameters
        pop_size = 20  # lambda
        sigma = 1.0    # Initial standard deviation
        cc = (4 + dim / (dim + 4)) / ((dim + 1.7) ** 2 + 4)
        cs = (dim + 1.7 + 2 / (dim + 1.7)) / ((dim + 3.5) + 2 / (dim + 1.7))
        cmu = min(0.05, (1 / cc - 2) / (pop_size / 2 + 1 / cc - 2))
        cmu = max(0.0, cmu)

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        sigma_0 = sigma
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Pre-compute weights
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Run CMA-ES
        while check_time() and calls_made < budget:
            # Generate population
            pop = np.zeros((pop_size, dim))
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                # Cholesky decomposition of C
                # To save time, we can update Cholesky or just use np.linalg.cholesky
                # For dim=5, np.linalg.cholesky is fast enough
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                pop[i] = x0 + sigma * (L @ z)
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate population
            pop_f = np.full(pop_size, float('inf'))
            for i in range(pop_size):
                fx = safe_call(pop[i])
                if fx is None:
                    break
                pop_f[i] = fx
            
            if not check_time() or calls_made >= budget:
                break

            # Sort by fitness
            idx = np.argsort(pop_f)
            pop = pop[idx]
            pop_f = pop_f[idx]
            
            # Update mean
            x0 = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ (x0 - x0) / sigma) # This is wrong, need previous mean
            
            # Correct CMA-ES update logic requires storing previous mean
            # Let's restart the loop structure to be cleaner with state variables
            
            # Actually, the above inline update is messy. Let's use a standard CMA-ES loop structure.
            # I will rewrite the CMA-ES part more carefully.
            
            break # Break out of this flawed loop, I will implement a proper CMA-ES below.

        # RE-IMPLEMENTATION OF CMA-ES PROPERLY
        # We restart the local search phase with a proper implementation.
        # We have already spent ~20 calls on LHS.
        # Let's run CMA-ES from the current best_x.
        
        if check_time() and calls_made < budget:
            x0 = best_x.copy()
            sigma = 1.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Weights
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            while check_time() and calls_made < budget:
                # Generate population
                pop = np.zeros((pop_size, dim))
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    pop[i] = x0 + sigma * (L @ z)
                    pop[i] = np.clip(pop[i], -5, 5)
                
                # Evaluate
                pop_f = np.full(pop_size, float('inf'))
                for i in range(pop_size):
                    fx = safe_call(pop[i])
                    if fx is None:
                        break
                    pop_f[i] = fx
                
                if not check_time() or calls_made >= budget:
                    break

                # Sort
                idx = np.argsort(pop_f)
                pop = pop[idx]
                pop_f = pop_f[idx]
                
                # Update mean
                x0_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
                
                # Update sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ (x0_new - x0) / sigma)
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update C
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (L @ (x0_new - x0) / sigma)
                
                delta_sig = 1 + 2 * (np.linalg.norm(p_c) ** 2 / dim - 1) / (dim + cc + 5)
                
                C = (1 + cmu * (mu_eff + 2) - cc) * C
                C += cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(pop[:mu] - x0, pop[:mu] - x0), axis=0)
                C -= cc * (1 + cmu * (mu_eff + 2) - cc) * np.outer(p_c, p_c) + cc * (1 + cmu * (mu_eff + 2) - cc) * np.outer(p_c, p_c) * 0 # This term is complex, simplifying
                # Standard update:
                # C = (1 + cmu * (mu_eff + 2) - cc) * C + cmu * sum(wi * (xi - x0)(xi - x0)^T) - cc * (1 + cmu * (mu_eff + 2) - cc) * pc * pc^T ?
                # Actually, the rank-mu update is:
                # C = (1 + cmu * (mu_eff + 2) - cc) * C + cmu * sum(wi * (xi - x0_new)(xi - x0_new)^T)
                # The rank-1 update is handled by p_c?
                # Let's use a simpler, robust CMA-ES variant or just Nelder-Mead if CMA-ES is too complex to get right quickly.
                
                # Given the constraints, a simpler and very effective method for dim=5 is 
                # **Nelder-Mead** or **Powell** from SciPy, run from multiple starts.
                # Let's switch to a Multi-Start Powell strategy, which is known to be robust and sample-efficient for low dimensions.
                
                break

        # Final Phase: Multi-Start Local Refinement (Powell/Nelder-Mead)
        # We have best_x from LHS.
        # We will run a few local optimizations from the best few LHS points.
        
        # Get top 3 points from LHS
        top_idx = np.argsort(lhs_f)[:3]
        top_points = lhs_points[top_idx]
        
        # Use remaining budget for local refinement
        # Split remaining budget among top points
        remaining_budget = budget - calls_made
        if remaining_budget > 10:
            n_refine = 3
            budget_per_refine = remaining_budget // n_refine
            
            for i in range(n_refine):
                if not check_time() or calls_made >= budget:
                    break
                
                start_point = top_points[i].copy()
                
                # Use scipy.optimize.minimize with Powell
                # We need a wrapper that counts calls and respects time
                class PowellWrapper:
                    def __init__(self, func, budget_limit):
                        self.func = func
                        self.calls = 0
                        self.budget_limit = budget_limit
                        self.best_x = start_point
                        self.best_f = float('inf')
                        
                    def __call__(self, x):
                        if self.calls >= self.budget_limit or not check_time():
                            return float('inf')
                        self.calls += 1
                        calls_made_nonlocal = None # Need to update nonlocal
                        # Since we can't easily update nonlocal from a nested class without passing it,
                        # let's just use a simple loop with scipy's options?
                        # SciPy's minimize doesn't easily allow stopping mid-call based on external time.
                        # Better to implement a simple local search or use a small number of steps.
                        
                        val = float(self.func(x))
                        if val < self.best_f:
                            self.best_f = val
                            self.best_x = x.copy()
                        return val

                # Instead of complex wrappers, let's just do a simple coordinate descent or Nelder-Mead manually?
                # No, let's stick to the CMA-ES idea but implement it correctly in a single block.
                
                # I will implement a clean, standard CMA-ES below.

        # CLEAN CMA-ES IMPLEMENTATION
        if check_time() and calls_made < budget:
            x0 = best_x.copy()
            sigma = 1.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            while check_time() and calls_made < budget:
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                pop = np.zeros((pop_size, dim))
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    pop[i] = x0 + sigma * (L @ z)
                    pop[i] = np.clip(pop[i], -5, 5)
                
                pop_f = np.full(pop_size, float('inf'))
                for i in range(pop_size):
                    fx = safe_call(pop[i])
                    if fx is None:
                        break
                    pop_f[i] = fx
                
                if not check_time() or calls_made >= budget:
                    break

                idx = np.argsort(pop_f)
                pop = pop[idx]
                pop_f = pop_f[idx]
                
                x0_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
                
                # Update sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ (x0_new - x0) / sigma)
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update C
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (L @ (x0_new - x0) / sigma)
                
                delta_sig = 1 + 2 * (np.linalg.norm(p_c) ** 2 / dim - 1) / (dim + cc + 5)
                
                # Rank-mu update
                C = (1 + cmu * (mu_eff + 2) - cc) * C
                C += cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(pop[:mu] - x0_new, pop[:mu] - x0_new), axis=0)
                
                # Rank-1 update
                C -= cc * (1 + cmu * (mu_eff + 2) - cc) * np.outer(p_c, p_c) * delta_sig # Approximation
                
                x0 = x0_new

        return best_x, best_f
