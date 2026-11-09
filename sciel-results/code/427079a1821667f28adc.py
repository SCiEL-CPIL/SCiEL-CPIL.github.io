import numpy as np
import time
import math

class Optimizer:
    """
    CMA-ES optimizer with periodicity-aware restarts for Rastrigin rotated (f15).
    
    Strategy:
    1. Run CMA-ES in short bursts.
    2. Track global best.
    3. On restart, re-center CMA-ES on the global best with a reset covariance 
       and moderate sigma to escape local minima.
    4. Use a small fraction of budget for "periodic jumps" (size ~1.0) to check 
       adjacent basins in the Rastrigin landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        # Global tracking
        best_x = None
        best_f = np.inf
        n_calls = 0
        rng = np.random.default_rng(42)

        def safe_eval(x):
            nonlocal best_x, best_f, n_calls
            if n_calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            
            n_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES Parameters
        # Standard settings for small dimensions
        lam = 10  # Population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Phase 1: Initial broad exploration
        # Start at center or a random point? Center is usually safe for [-5,5] if shift is small, 
        # but random is better for unknown shift. Let's start at 0 but with large sigma.
        mean = np.zeros(dim)
        sigma = 2.5  # Large initial sigma to cover the box
        
        # Run CMA-ES for the first chunk
        calls_per_chunk = 100
        max_chunks = 4 # 400 calls total for CMA-ES, leaving 100 for polish/jumps
        
        for chunk_idx in range(max_chunks):
            if n_calls >= budget or time.time() > deadline - 10:
                break
            
            # Initialize or Reset CMA-ES state
            if chunk_idx == 0:
                mean = np.zeros(dim)
                sigma = 2.5
            else:
                # Restart logic: Center on global best, reset covariance
                if best_x is not None:
                    mean = best_x.copy()
                else:
                    mean = np.zeros(dim)
                
                # Reset sigma to a moderate value to explore around the best
                # 1.0 is roughly the period of Rastrigin, so this allows jumping to neighbors
                sigma = 1.2 
                
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Run CMA-ES for this chunk
            chunk_start_calls = n_calls
            while n_calls < budget and (n_calls - chunk_start_calls) < calls_per_chunk:
                if time.time() > deadline - 5:
                    break
                
                # Generate population
                pop = np.zeros((lam, dim))
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    # If Cholesky fails, reset covariance and continue
                    C = np.eye(dim)
                    L = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    sigma = max(sigma, 0.5)
                    continue

                for i in range(lam):
                    z = rng.standard_normal(dim)
                    pop[i] = mean + sigma * (L @ z)
                    
                # Evaluate population
                fvals = np.zeros(lam)
                for i in range(lam):
                    fvals[i] = safe_eval(pop[i])
                    if n_calls >= budget:
                        break

                if n_calls >= budget:
                    break

                # Check if we have enough valid evaluations
                valid_mask = np.isfinite(fvals)
                if np.sum(valid_mask) < mu:
                    break

                # Sort by fitness (only valid ones matter for selection, but we need to handle inf)
                # Replace inf with a large number for sorting
                fvals_for_sort = np.where(valid_mask, fvals, 1e10)
                order = np.argsort(fvals_for_sort)
                
                # Select top mu
                y = np.zeros((mu, dim))
                for i in range(mu):
                    y[i] = pop[order[i]]
                    # If the best point was invalid, this logic is tricky, 
                    # but safe_eval returns inf for invalid, so they sort to bottom.
                    
                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * y, axis=0)
                
                # Update paths
                v = (new_mean - mean) / sigma
                
                # Compute C^{-1/2}
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                ps = (1 - c_sigma) * ps + c_sigma / math.sqrt(c_sigma) * (C_inv_sqrt @ v)
                
                norm_ps = np.linalg.norm(ps)
                # Approximate t_gen for h_sigma calculation
                t_gen = (n_calls - chunk_start_calls) / lam
                h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma)**(2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)
                
                pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * v
                
                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                delta = (1 + delta_h) * (1 - c_1)
                
                rank1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_norm = y[i] - mean
                    rank_mu += weights[i] * np.outer(y_norm, y_norm)
                    
                C = delta * C + c_1 * (rank1 - C) + c_mu * rank_mu / (sigma**2)
                C = (C + C.T) / 2
                
                # Update sigma
                sigma = sigma * math.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                # Clamp sigma to prevent explosion or vanishing
                sigma = np.clip(sigma, 1e-4, 5.0)
                
                mean = new_mean

        # Phase 2: Periodicity-aware Jumps
        # Rastrigin has minima every 2 units. If we are stuck, jumping by ~1 unit 
        # in random directions might find a better basin.
        # Use remaining budget for this.
        if best_x is not None and n_calls < budget and time.time() < deadline - 5:
            jump_sigma = 1.0 # Half period
            remaining = budget - n_calls
            # Do a limited number of jumps
            max_jumps = min(remaining, 50)
            for _ in range(max_jumps):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                
                # Random direction
                dir_vec = rng.standard_normal(dim)
                dir_vec /= np.linalg.norm(dir_vec)
                
                # Try a few distances around 1.0
                for dist in [0.8, 1.0, 1.2]:
                    if n_calls >= budget or time.time() > deadline - 5:
                        break
                    candidate = best_x + dist * dir_vec
                    candidate = np.clip(candidate, -5.0, 5.0)
                    f_val = safe_eval(candidate)
                    # If we found something better, it updates best_x/best_f automatically
                    # We don't restart CMA-ES here, just probe.
        
        # Phase 3: Final Polish
        # Local search around the best found with very small steps
        if best_x is not None and n_calls < budget and time.time() < deadline - 5:
            polish_sigma = 0.1
            remaining = budget - n_calls
            max_polish = min(remaining, 50)
            for _ in range(max_polish):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                y = best_x + polish_sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                safe_eval(y)
                # If best_f improved, best_x is updated.
                # We could adapt polish_sigma, but fixed small is safer for time.

        if best_x is None:
            # Fallback if no calls were made or all failed
            best_x = np.zeros(dim)
            # One last desperate call if possible
            if n_calls < budget:
                safe_eval(best_x)
            else:
                best_f = np.inf

        return best_x, best_f
