"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        # Standard: 4 + floor(3 * ln(n))
        # For n=5, 4 + 3*1.6 = 8.8 -> 8
        # Let's use a slightly larger population for better exploration initially?
        # 4 + 3*ln(5) is standard.
        pop_size = 4 + int(3 * np.log(dim)) 
        # For dim=5, pop_size = 8.
        
        # Initial step size
        # Domain is [-5, 5]^5, minimizer in [-4, 4]^5.
        # Distance from center to corner is ~ 5*sqrt(5) ~ 11.
        # But we want to cover the likely region [-4,4].
        # sigma = 2.0 is reasonable.
        sigma = 2.0
        
        # Adaptation rates
        mu = pop_size // 2  # Number of selected individuals
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance reduction factor
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        
        # Damping for sigma adaptation
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Initial mean
        # Start at 0,0,0,0,0. The hidden shift is in [-4,4].
        # CMA-ES will adapt.
        mean = np.zeros(dim)
        
        # Initial covariance matrix
        C = np.eye(dim)
        
        # Eigenvalue decomposition for sampling
        D = np.sqrt(np.diag(C))
        B = np.eye(dim)
        
        # Evolution path for sigma adaptation
        p_sigma = np.zeros(dim)
        
        # Evolution path for mean adaptation
        p_c = np.zeros(dim)
        
        # Best solution tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_time():
            return time.time() > deadline

        # Initial evaluation of mean
        f_mean = func(mean)
        calls_made += 1
        update_best(mean, f_mean)
        
        # Main loop
        while calls_made < budget and not check_time():
            # Generate population
            # We need to generate pop_size individuals.
            # But we must respect the call budget.
            # If remaining budget < pop_size, we might not be able to complete a generation.
            # CMA-ES usually requires a full generation.
            # If we can't afford a full generation, we can break or do a partial one?
            # Standard CMA-ES breaks.
            if budget - calls_made < pop_size:
                # If we have some calls left, maybe do a (1+1) step or just break?
                # Breaking is safer to maintain CMA-ES integrity, but wastes budget.
                # Let's try to use remaining budget for a simple hill climb or just break.
                # Given the scoring, early progress is good. If we are close to end,
                # a few extra random steps might help.
                # But CMA-ES state is complex. Let's just break to avoid partial generations
                # which might mess up adaptation if not handled carefully.
                # Actually, we can just run (1+1) ES with the current sigma for remaining calls.
                # Or just break. The previous program used NM.
                # Let's stick to CMA-ES logic: if we can't do a full gen, we stop.
                # To use the last few calls efficiently, we can do a local search.
                # But for simplicity and robustness, let's just break.
                # Wait, if we break, we might waste 7 calls.
                # Let's do a simple (1+1) step for remaining calls using current mean and sigma.
                remaining = budget - calls_made
                for _ in range(remaining):
                    if check_time():
                        break
                    z = np.random.standard_normal(dim)
                    x_trial = mean + sigma * z
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_made += 1
                    if f_trial < f_mean:
                        mean = x_trial
                        f_mean = f_trial
                        update_best(mean, f_mean)
                        # Simple step size adaptation for this tail phase
                        sigma *= 1.1
                        sigma = np.clip(sigma, 0.01, 10.0)
                    else:
                        sigma *= 0.9
                        sigma = np.clip(sigma, 0.01, 10.0)
                break
            
            # Sample pop_size points
            samples = []
            f_values = []
            
            for i in range(pop_size):
                # Sample from N(mean, sigma^2 * C)
                # z ~ N(0, I)
                z = np.random.standard_normal(dim)
                x_new = mean + sigma * (B @ (D * z))
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls_made += 1
                update_best(x_new, f_new)
                
                samples.append(x_new)
                f_values.append(f_new)
                
                if check_time():
                    # If time runs out mid-generation, we should stop.
                    # We have partial population.
                    # We can still update the best, but adaptation might be skewed.
                    # For safety, just break out of the main loop entirely.
                    break
            
            if check_time():
                break
            
            # Sort samples by fitness
            indices = np.argsort(f_values)
            samples = [samples[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            
            # Select top mu
            selected_samples = np.array(samples[:mu])
            selected_weights = weights
            
            # New mean
            mean_new = np.sum(selected_weights[:, None] * selected_samples, axis=0)
            
            # Update evolution path for mean (p_c)
            # p_c = (1 - cc) * p_c + sqrt(cc * (2 - cc) * mueff) * C^{-1/2} * (mean_new - mean) / sigma
            # We need C^{-1/2}. We have B and D. C = B D^2 B^T. C^{-1/2} = B D^{-1} B^T.
            # So C^{-1/2} * v = B @ (D^{-1} * (B.T @ v))
            diff = mean_new - mean
            C_inv_sqrt_diff = B @ (D ** (-1) * (B.T @ diff))
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * C_inv_sqrt_diff / sigma
            
            # Check for termination of sigma adaptation if p_c is too large
            # hsig = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - cc)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mueff + 1) / (dim + 1))
            # We don't track gen explicitly, but we can estimate.
            # For now, let's skip the hsig check for simplicity, or use a fixed threshold.
            # If p_c norm is very large, sigma adaptation might be unstable.
            # Let's just update sigma.
            
            # Update sigma
            # sigma = sigma * exp(cs / damps * (||p_sigma|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(dim) * (1 - 1/(4dim) + 1/(21dim^2))
            # p_sigma = (1 - cs) * p_sigma + sqrt(cs * (2 - cs) * mueff) * C^{-1/2} * (mean_new - mean) / sigma
            # Wait, the formula for p_sigma uses the difference between the new mean and the old mean?
            # Yes, in the standard CMA-ES, p_sigma is updated with the normalized difference.
            # Actually, the standard formula is:
            # p_sigma = (1 - cs) * p_sigma + sqrt(cs * (2 - cs) * mueff) * C^{-1/2} * (mean_new - mean) / sigma
            # This is the same term as in p_c but without the C_inv_sqrt part?
            # No, p_sigma uses the raw difference scaled by sigma and C^{-1/2}?
            # Let's check the standard pseudocode.
            # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mueff) * C^{-1/2} * (x_mean_new - x_mean_old) / sigma
            # Yes.
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt_diff / sigma
            
            # Calculate norm of p_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Expected norm of standard normal
            expected_norm = np.sqrt(dim) * (1 - 1.0/(4*dim) + 1.0/(21*dim**2))
            
            # Update sigma
            sigma *= np.exp(cs / damps * (norm_p_sigma / expected_norm - 1))
            
            # Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (1 - hsig) * cc * (2 - cc) * C) + cmu * sum(weights_i * (y_i y_i^T))
            # where y_i = (x_i - mean_old) / sigma
            # Note: We use mean_old for the difference to update C.
            
            # Calculate y_i
            y_vectors = (selected_samples - mean) / sigma
            y_outer = np.sum(selected_weights[:, None, None] * y_vectors[:, :, None] * y_vectors[:, None, :], axis=0)
            
            # hsig is typically 1 if p_c is not too large.
            # Let's assume hsig = 1 for now, or calculate it properly.
            # hsig = 1 if ||p_c|| / sqrt(1 - (1-cc)^(2*gen)) < (1.4 + 2/(dim+1)) * sqrt((mueff+1)/(dim+1))
            # Without gen count, let's just use 1.
            hsig = 1.0
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C) + cmu * y_outer
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update mean
            mean = mean_new
            
            # Update B and D (Eigen decomposition)
            # This is the most expensive part.
            # For dim=5, it's fast.
            try:
                D, B = np.linalg.eigh(C)
                # eigh returns eigenvalues in ascending order.
                # We need D = sqrt(eigenvalues).
                # Ensure positive eigenvalues (numerical stability)
                D = np.sqrt(np.maximum(D, 1e-12))
            except np.linalg.LinAlgError:
                # Fallback if decomposition fails
                D = np.sqrt(np.diag(C))
                B = np.eye(dim)
            
            # Check convergence
            # If sigma is very small or C is near singular, we might be converged.
            if sigma < 1e-8:
                break
            if np.max(D) / np.min(D) > 1e14:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
