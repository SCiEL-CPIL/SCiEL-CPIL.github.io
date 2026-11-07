import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We stop at 310s to be safe.
        deadline = start_time + 310.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            # Clip to box [-5, 5] as per problem description
            x_clipped = np.clip(x, -5.0, 5.0)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # --- CMA-ES Implementation ---
        
        # Parameters
        lambda_pop = 10  # Population size
        mu = lambda_pop // 2  # Number of parents
        
        # Weights
        weights = np.log(lambda_pop / 2.0 + 1.0) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2.0) / (lambda_pop + mu_eff + 5.0)
        c_c = (4.0 + mu_eff / lambda_pop) / (lambda_pop + 4.0 + 2.0 * mu_eff / lambda_pop)
        c_1 = 2.0 / (lambda_pop**2 + 6.0)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (lambda_pop**2 + 2.0))
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) * (lambda_pop - mu_eff - 2.0) / (mu_eff + 1.0) - 1.0)) + c_sigma

        # Initial state
        # Start at the center of the box, which is often a good heuristic for BBOB
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size. Box is [-5,5], so 2.0 covers a good chunk.
        C = np.eye(dim)
        C_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute Cholesky decomposition of C (initially I, so L=I)
        # We need to update L every generation or use SVD/Chol for sampling
        # For simplicity and speed in low dim, we can use Cholesky update or just recompute
        # Recomputing Cholesky for 5x5 is very fast.
        
        def sample_from_mvnormal(mean, sigma, C):
            # Cholesky decomposition of C
            L = np.linalg.cholesky(C)
            z = np.random.standard_normal((lambda_pop, dim))
            # Transform to get correlated samples
            # x = mean + sigma * z @ L.T
            return mean + sigma * (z @ L.T)

        # Main Loop
        # We have 500 budget. lambda_pop=10. Max generations = 50.
        
        while calls < budget and time_ok():
            # 1. Sample population
            pop = sample_from_mvnormal(mean, sigma, C)
            
            # 2. Evaluate
            fitness = np.array([eval_func(x) for x in pop])
            
            # 3. Sort by fitness
            order = np.argsort(fitness)
            pop = pop[order]
            fitness = fitness[order]
            
            # 4. Update mean
            mean_new = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # 5. Update covariance matrix
            # Weighted outer product of the differences
            y = pop[:mu] - mean
            C_new = C
            for i in range(mu):
                C_new += weights[i] * np.outer(y[i], y[i])
            # Update rule: C = (1 - c_1 - c_mu) * C + c_1 * (1 + (1-c_1)*d_sigma)*pc*pc.T + c_mu * sum(w_i * y_i * y_i.T)
            # Wait, the standard update is:
            # C <- (1 - c1 - cmu) * C + c1 * (1 + (1-c1)*dsigma) * pc * pc.T + cmu * sum(wi * yi * yi.T)
            
            # Let's re-calculate C_new properly
            sum_yi_yi = np.sum(weights[:, None, None] * y[:, :, None] * y[:, None, :], axis=0)
            
            # Update path cm
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * sigma * (C_inv @ (mean_new - mean))
            
            # Update sigma
            # norm(pc) is needed
            # Note: C_inv @ (mean_new - mean) is the step in the transformed space
            # Actually, the standard formula uses:
            # sigma <- sigma * exp(c_sigma/d_sigma * (||pc|| / (sqrt(mu_eff) * E||N(0,I)||) - 1))
            # E||N(0,I)|| for dim d is approx sqrt(d) * (1 - 1/(4d) + ...)
            # For dim=5, E||N|| ~ 2.13
            
            e_norm = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2)) # Approximation
            # More accurate: gamma((d+1)/2) / gamma(d/2) * sqrt(2)
            # For d=5: gamma(3)/gamma(2.5) * sqrt(2) = 2 / (1.3293) * 1.414 ~ 2.13
            
            norm_pc = np.linalg.norm(pc)
            sigma *= np.exp(c_sigma / d_sigma * (norm_pc / (np.sqrt(mu_eff) * e_norm) - 1.0))
            
            # Update C
            # We need C_inv for the pc update above, but we update C after?
            # Standard CMA-ES updates C using the old C_inv? No, usually C_inv is updated after C.
            # However, the pc update uses C_inv. If we update C, C_inv changes.
            # In many implementations, C_inv is updated periodically or via Sherman-Morrison.
            # For dim=5, we can just recompute C_inv = inv(C_new) after updating C.
            # But the pc update in step 5 used the OLD C_inv. This is correct for the standard algorithm.
            
            C = (1 - c_1 - c_mu) * C + c_1 * (1 + (1 - c_1) * d_sigma) * np.outer(pc, pc) + c_mu * sum_yi_yi
            
            # Ensure C is positive definite
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update C_inv
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                # If singular, add small identity
                C_inv = np.linalg.inv(C + 1e-10 * np.eye(dim))

            mean = mean_new
            
            # Check for stagnation or convergence
            # If sigma becomes too small or too large, we might restart or stop
            if sigma < 1e-8:
                break
                
            # Optional: If no improvement for many generations, restart?
            # With CMA-ES, this is less necessary if initialized well.
            
        return best_x, best_f
