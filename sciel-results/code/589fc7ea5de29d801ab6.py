import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val is None or np.isnan(f_val) or np.isinf(f_val):
                    return None
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration to find a good starting basin
        # f13 has a global minimum, but local minima can be tricky.
        # Random sampling helps locate the basin.
        n_global = 15
        init_points = []
        init_fvals = []
        for _ in range(n_global):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is not None:
                init_points.append(x)
                init_fvals.append(f_val)
        
        if best_x is None:
            # Fallback if no valid calls were made (should not happen)
            x0 = np.zeros(dim)
            safe_call(x0)
            return best_x, best_f

        # Use the best point from global exploration as the initial mean for CMA-ES
        # If we have multiple points, we could use a weighted average, but the best is usually sufficient
        # for CMA-ES to start adapting.
        mu = best_x.copy()
        
        # Initialize CMA-ES parameters
        # Lambda: number of offspring per generation
        # For dim=5, lambda = 4 + floor(3 * ln(5)) = 4 + 4 = 8. Let's use 10 for better exploration.
        lam = 10 
        # Initial step size sigma
        # The box is [-5, 5], width 10. A reasonable initial sigma is 1.0 or 2.0.
        # Since we started from a random point in [-5,5], 1.0 is conservative, 2.0 is aggressive.
        # CMA-ES adapts sigma, so 1.0 is a safe start.
        sigma = 1.0
        C = np.eye(dim)  # Covariance matrix
        # Pre-compute Cholesky factor for sampling
        # C = L L^T
        L = np.linalg.cholesky(C)
        
        # Weights for recombination
        # Mu = number of parents to use for recombination
        mu_eff = int(lam / 2)
        # Weights w_i for the top mu_eff individuals
        # Standard weights: w_i = log(mu_eff/2 + 1) - log(i) for i=1..mu_eff
        weights = np.zeros(mu_eff)
        for i in range(mu_eff):
            weights[i] = np.log(mu_eff / 2.0 + 1.0) - np.log(i + 1.0)
        weights /= np.sum(weights)
        
        # Variance of the selection weights
        mu_eff_inv = 1.0 / (np.sum(weights) ** 2)
        
        # Adaptation parameters
        c_sigma = (mu_eff_inv + 2) / (dim + mu_eff_inv + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff_inv - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff_inv / dim) / (dim + 4 + 2 * mu_eff_inv / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff_inv)
        c_mu = min(1 - c_1, 2 * (mu_eff_inv - 2 + 1 / mu_eff_inv) / ((dim + 2) ** 2 + mu_eff_inv))
        
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Max iterations for CMA-ES
        # We have budget - calls remaining.
        # Each iteration uses lam calls.
        max_iter_cma = (budget - calls) // lam
        if max_iter_cma < 1:
             # If not enough budget for even one CMA-ES generation, just return best
             return best_x, best_f

        for gen in range(max_iter_cma):
            if calls >= budget or time.time() > deadline:
                break

            # 1. Sample lambda offspring
            offspring = []
            # Sample standard normal
            z = rng.standard_normal((lam, dim))
            # Transform to x
            # x = mu + sigma * (z @ L.T)
            x_offspring = mu + sigma * (z @ L.T)
            
            # Evaluate offspring
            f_offspring = []
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                f_val = safe_call(x_offspring[i])
                if f_val is None:
                    # If evaluation fails, treat as infinity
                    f_offspring.append(float('inf'))
                else:
                    f_offspring.append(f_val)
            
            # If we ran out of budget mid-generation, pad with inf
            while len(f_offspring) < lam:
                f_offspring.append(float('inf'))
                x_offspring.append(np.full(dim, np.nan)) # Dummy

            # 2. Sort by fitness
            # We only care about the top mu_eff
            # np.argsort returns indices that would sort the array
            sorted_indices = np.argsort(f_offspring)
            # Get the top mu_eff indices
            top_indices = sorted_indices[:mu_eff]
            
            # 3. Update mean
            # y = weighted average of the top mu_eff individuals
            x_top = x_offspring[top_indices]
            # Weights are already normalized
            y = np.sum(weights[:, np.newaxis] * x_top, axis=0)
            mu_new = mu + y

            # 4. Update step size sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (1/sigma) * (sqrt(mu_eff) * y)
            # Note: y is the vector from old mu to new mu, so it's in x-space.
            # The standard update uses the standardized step.
            # y_std = y / sigma
            y_std = y / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff_inv) * y_std
            
            # Check if p_sigma is too large
            # If ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(gen+1)}) > 1.4 + c_sigma, reduce sigma
            # For simplicity, we can use a simpler check: if ||p_sigma|| is very large, reduce sigma.
            # But the standard CMA-ES uses a dynamic threshold.
            # Let's implement the standard check.
            # We need to track the number of generations to compute the denominator.
            # This is getting complex. A simpler adaptation:
            # If the step size leads to many points outside the box, it might be too large.
            # Or just use the standard formula if we track the iteration count properly.
            
            # Let's stick to a simpler, robust variant often used in practice:
            # Update sigma based on the norm of p_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Expected norm of p_sigma under stationary conditions is approx sqrt(mu_eff_inv)
            # If norm_p_sigma is significantly larger, we might have found a direction of improvement 
            # or the step size is too big.
            # Standard CMA-ES:
            # h_sigma = 1 if norm(p_sigma) / sqrt(1 - (1-c_sigma)^{2(gen+1)}) < (1.4 + c_sigma) * E[N(0,I)]
            # E[N(0,I)] = sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim^2))
            # This is complex. Let's use a simpler heuristic:
            # If the best fitness improved significantly, keep sigma.
            # Actually, let's just implement the standard update with a fixed denominator approximation 
            # or skip the h_sigma check and just update sigma multiplicatively based on p_sigma.
            
            # A common simplification:
            # sigma = sigma * exp(c_sigma/d_sigma * (||p_sigma||/E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim^2))
            c_sigma_d_sigma = c_sigma / d_sigma
            expected_norm = np.sqrt(dim) * (1 - 1.0/(4.0*dim) - 1.0/(25.0*dim*dim))
            if expected_norm > 0:
                sigma *= np.exp(c_sigma_d_sigma * (norm_p_sigma / expected_norm - 1.0))
            
            # 5. Update covariance matrix C
            # p_c = (1 - c_c) * p_c + c_c * (y / sigma)
            p_c = (1 - c_c) * p_c + c_c * (y / sigma)
            
            # Outer product of the top mu_eff standardized steps
            # z_top = z[top_indices]
            # The update is C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + delta_sig * C) + c_mu * sum(w_i * z_i z_i^T)
            # Where z_i are the standardized steps (y_i / sigma)
            # y_i = x_i - mu (old mu)
            # z_i = y_i / sigma
            
            # We need the standardized vectors for the top mu_eff individuals
            # x_top are the x-values.
            # y_top = x_top - mu (old mu)
            y_top = x_top - mu
            z_top = y_top / sigma
            
            # Sum of weighted outer products
            # c_mu * sum(w_i * z_i z_i^T)
            # This can be computed as c_mu * (Z^T * W * Z) where Z is mu_eff x dim, W is diag(weights)
            # Efficiently: c_mu * (Z.T @ (weights[:, np.newaxis] * Z))
            C_update = np.zeros((dim, dim))
            for i in range(mu_eff):
                C_update += weights[i] * np.outer(z_top[i], z_top[i])
            
            # Delta_sig term: delta_sig * C
            # delta_sig = c_sigma * (1 - c_sigma) * ||p_sigma||^2 / dim
            # (This is an approximation, standard is more complex)
            delta_sig = c_sigma * (1 - c_sigma) * (norm_p_sigma ** 2) / dim
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_sig * C) + c_mu * C_update
            
            # Re-orthogonalize C to ensure it is positive semi-definite
            # C = 0.5 * (C + C.T)
            C = 0.5 * (C + C.T)
            
            # Update Cholesky factor
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity (or diagonal)
                C = np.eye(dim)
                L = np.eye(dim)
                p_c = np.zeros(dim)
                p_sigma = np.zeros(dim)
                sigma = 1.0 # Reset step size
            
            # Update mu
            mu = mu_new

            # Check convergence: if sigma is too small or C is diagonal
            if sigma < 1e-10:
                break
            # Also check if the spread of the population is very small
            # This is handled by the budget limit mostly.

        # Final check: if we have budget left, do a few local polish steps?
        # CMA-ES usually converges well. If we have time, we can do a simple coordinate descent
        # or just return the best. Given the budget is 500 and CMA-ES uses ~10 per gen,
        # we might have ~40 generations. This is usually sufficient for dim=5.
        
        return best_x, best_f
