"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f
            calls_made += 1
            try:
                val = float(func(np.asarray(x, dtype=np.float64)))
                if np.isfinite(val):
                    if val < best_f:
                        best_f = val
                        best_x = np.array(x, dtype=np.float64)
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        # CMA-ES Parameters
        # sigma: initial standard deviation. Box is [-5,5]^5, so range is 10.
        # A good initial sigma is roughly 1/5 to 1/10 of the range, or based on expected distance to optimum.
        # Since optimum is in [-4,4], starting from 0 is reasonable.
        sigma = 2.0
        lambda_ = 4 * int(np.ceil(3 * np.log(dim)))  # Population size
        if lambda_ < 4:
            lambda_ = 4
        
        # Weights
        mu = lambda_ // 2
        weights = np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (dim + 1) ** 2 + c_sigma) / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Precompute cholesky factorization updates
        def update_chol(C, C_sqrt):
            # Perform Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # If not positive definite, regularize
                C_reg = C + 1e-10 * np.eye(dim)
                L = np.linalg.cholesky(C_reg)
                return L

        # Initial call to satisfy "first call" requirement and establish baseline
        # Also, it's good to evaluate the center of the box
        x0 = np.zeros(dim)
        f0 = safe_func(x0)
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        # Main Loop
        # We need to ensure we don't exceed budget.
        # CMA-ES consumes lambda_ calls per generation.
        
        while calls_made + lambda_ <= budget and time.time() < deadline:
            # Generate population
            # x_i = x_mean + sigma * B * z_i
            # Where B is the matrix square root of C
            # Using Cholesky: C = B B^T, so B = L
            # But we maintain C_sqrt such that C = C_sqrt C_sqrt^T
            # So we sample z ~ N(0, I) and x = x_mean + sigma * C_sqrt @ z
            
            population = []
            fitness = []
            
            # Sample population
            for i in range(lambda_):
                z = np.random.randn(dim)
                x_i = x_mean + sigma * (C_sqrt @ z)
                # Clip to box? The problem says func clips, but CMA-ES works better if we stay within bounds 
                # or at least handle it. The task says "func(x) clips a finite length-5 vector to the box".
                # So we can pass any vector, it gets clipped. 
                # However, sampling outside might be inefficient. 
                # Let's allow it, the function handles it.
                f_i = safe_func(x_i)
                population.append(x_i)
                fitness.append(f_i)
                
                if time.time() > deadline - 5:
                    break
            
            if len(population) < lambda_:
                break

            # Sort by fitness
            sorted_indices = np.argsort(fitness)
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitness = [fitness[i] for i in sorted_indices]
            
            # Update mean
            x_new_mean = np.sum([weights[i] * sorted_population[i] for i in range(mu)], axis=0)
            
            # Update step size
            # ps = (1 - c_sigma) ps + h_sigma * sqrt(mu_eff) * c_sigma / sigma * B^-1 (x_new_mean - x_mean)
            # B = C_sqrt
            # B^-1 = C_inv_sqrt
            # We need C_inv_sqrt. We can compute it from C_sqrt.
            # C = C_sqrt C_sqrt^T. C^-1 = C_sqrt^-T C_sqrt^-.
            # So C_inv_sqrt = C_sqrt^-.
            
            # Compute C_inv_sqrt from current C_sqrt
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            diff = (x_new_mean - x_mean)
            norm_diff = np.linalg.norm(diff)
            
            # h_sigma condition
            # E ||B^-1 z|| = 1 - 1/(2 dim) + ... approx 1
            # Condition: ||ps|| / sqrt(1 - (1-c_sigma)^{2 gen}) < (1.4 + 2/(dim+1)) * E||N(0,I)||
            # Simplified: if norm_diff/sigma is small enough
            h_sigma = 1.0
            if norm_diff > 0:
                # Approximate expected norm of standard normal vector in dim dimensions
                # E||z|| = sqrt(2) * Gamma((dim+1)/2) / Gamma(dim/2)
                # For dim=5, approx 2.1
                e_norm = np.sqrt(2) * (np.exp(np.math.lgamma((dim+1)/2)) / np.exp(np.math.lgamma(dim/2)))
                # Threshold
                threshold = (1.4 + 2/(dim+1)) * e_norm
                # We don't track generation count for the denominator, so we use a simplified heuristic
                # or just check if the step is "normal"
                # A common heuristic: h_sigma = 1 if norm_diff < threshold * sigma
                # Actually, the standard formula uses the history of ps.
                # Let's use a simpler check: if the update is not too large
                if norm_diff < threshold * sigma:
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0

            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(mu_eff) * c_sigma / sigma * (C_inv_sqrt @ diff)
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / e_norm - 1))
            
            # Clamp sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Update evolution paths
            # pc = (1 - c_c) pc + h_c * sqrt(mu_eff) * c_c * B^-1 (x_new_mean - x_mean)
            # h_c condition: ||ps|| / sqrt(1 - (1-c_sigma)^{2 gen}) < (1.4 + 2/(dim+1)) * E||N(0,I)||
            # Reusing the same heuristic
            h_c = 1.0
            if norm_diff > 0:
                if norm_diff < threshold * sigma:
                    h_c = 1.0
                else:
                    h_c = 0.0

            pc = (1 - c_c) * pc + h_c * np.sqrt(mu_eff) * c_c * (C_inv_sqrt @ diff)

            # Update covariance matrix
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - h_c^2) c_c (1 - c_c) C) + c_mu sum w_i (y_i y_i^T)
            # Where y_i = B^-1 (x_i - x_mean)
            
            # Compute y_i for the mu best individuals
            y_list = []
            for i in range(mu):
                y_i = C_inv_sqrt @ (sorted_population[i] - x_mean)
                y_list.append(y_i)
            
            # Outer product sum
            sum_ww_yy = np.zeros((dim, dim))
            for i in range(mu):
                sum_ww_yy += weights[i] * np.outer(y_list[i], y_list[i])
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_c**2) * c_c * (1 - c_c) * C) + c_mu * sum_ww_yy
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            # Update C_sqrt (Cholesky)
            C_sqrt = update_chol(C, C_sqrt)
            
            # Update mean
            x_mean = x_new_mean

            # Safety check: if x_mean goes far outside, pull it back? 
            # The function clips, so it's safe, but might be inefficient.
            # Let's just let it run.

        # Final polish if we have budget left
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline - 5:
            # Try a few random perturbations around best_x
            for _ in range(5):
                if calls_made >= budget - 1 or time.time() > deadline - 1:
                    break
                perturbation = np.random.randn(dim) * 0.1 * sigma
                x_try = best_x + perturbation
                f_try = safe_func(x_try)
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f

# EVOLVE-BLOCK-END
