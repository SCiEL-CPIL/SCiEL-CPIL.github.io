"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Simplified CMA-ES with time-based budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # Population size
        # Weights
        mu = lambda_ // 2
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Chi_lambda is approx for standard normal
        chi_lambda = lambda_ ** 0.5 * (1 - 1 / (4 * lambda_) - 1 / (25 * lambda_ ** 2))
        
        # Initial state
        x_m = np.random.uniform(-5, 5, dim)
        # Start with a reasonable sigma, maybe 1.0 or 2.0
        sigma = 1.0
        C = np.eye(dim)
        C_pow_half = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x_m.copy()
        best_f = func(x_m)
        
        # Precompute eigendecomposition update interval
        # Usually every 1/dim generations, but for small dim we can do more often
        # Let's do it every few iterations to save time
        
        counter = 0
        # We need to track calls. Initial call used 1.
        calls_made = 1
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            # We need to generate lambda samples
            # To save time, we can batch generate if possible, but func is called one by one
            
            ys = np.zeros((lambda_, dim))
            fs = np.zeros(lambda_)
            
            # Generate samples
            # y = x_m + sigma * C_pow_half @ z
            # We can generate z for all lambda at once
            Z = np.random.standard_normal((lambda_, dim))
            # Apply C_pow_half
            # This is matrix multiplication: (lambda_, dim) @ (dim, dim)
            # Z @ C_pow_half.T
            ys = x_m + sigma * (Z @ C_pow_half.T)
            
            # Evaluate
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                # Clip to box
                y_i = np.clip(ys[i], -5, 5)
                f_i = func(y_i)
                fs[i] = f_i
                calls_made += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = y_i.copy()
            
            if calls_made >= budget:
                break
            
            # Sort by fitness
            # Only consider valid fs (though all should be valid here)
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Update mean
            w = weights
            x_m_new = np.sum(w[:, np.newaxis] * ys[:mu], axis=0)
            
            # Update step size control
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * C_pow_half @ (x_m_new - x_m)
            # Note: In standard CMA-ES, the update is in the transformed space
            # But since we apply C_pow_half to generate, the direction in transformed space is (x_m_new - x_m)/sigma
            # Actually, the standard update for ps is:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * C^{-1/2} * (x_m_new - x_m) / sigma
            # Wait, C^{-1/2} is the inverse of C^{1/2}.
            # Let's compute C_inv_half. Since C is symmetric positive definite, we can use eigendecomposition.
            
            # To avoid full eigendecomp every step, we can approximate or do it periodically.
            # For dim=5, eigendecomp is cheap. Let's do it every step for simplicity and correctness.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive
            eigvals = np.maximum(eigvals, 1e-20)
            C_inv_half = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            C_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Update ps
            # The vector (x_m_new - x_m) is in the original space.
            # We need to transform it to the "natural" coordinates of the distribution.
            # The distribution is N(x_m, sigma^2 * C).
            # So z = C^{-1/2} @ (x - x_m) / sigma
            # The update rule for ps uses the step in the transformed space.
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * C^{-1/2} @ (x_m_new - x_m) / sigma
            delta_m = (x_m_new - x_m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_half @ delta_m)
            
            # Update sigma
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (counter + 1)))) < (1.4 + 2 / (dim + 1)) * chi_lambda
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / chi_lambda - 1)) # c_sigma/c_sigma is 1
            
            # Update covariance matrix
            # pc = (1 - c_c) * pc + h_sigma * c_c * (x_m_new - x_m) / sigma
            # Note: h_sigma is a boolean, used to decide if we update pc with the mean shift
            # In some implementations, if h_sigma is false, pc is just decayed.
            if h_sigma:
                pc = (1 - c_c) * pc + c_c * delta_m
            else:
                pc = (1 - c_c) * pc
            
            # Rank-1 update
            # C = (1 - c_1 - c_mu * mueff + mu * h1 * c_1) * C + c_1 * pc @ pc.T
            # where h1 is the Heaviside function for the pc update
            # Actually, the standard formula is:
            # C = (1 - c_1 - c_mu) * C + c_1 * pc @ pc.T + c_mu * sum(w_i * y_i @ y_i.T)
            # where y_i are the selected steps in the transformed space?
            # No, y_i in the covariance update are the steps from the old mean.
            # Let's stick to the standard:
            # C = (1 - c_1 - c_mu * mueff) * C + c_1 * pc @ pc.T + c_mu * sum_{i=1}^mu w_i * (x_i - x_m_old) @ (x_i - x_m_old).T / sigma^2
            # Wait, the steps are (x_i - x_m_old).
            # The covariance is updated in the original space.
            
            # Let's use the eigendecomposition to ensure C stays positive definite and update C_half
            # But first, update C
            # h1 = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (counter + 1)))) < (1.4 + 2 / (dim + 1)) * chi_lambda
            # This is similar to h_sigma but for pc. Let's just use h_sigma for now or compute h1.
            # For simplicity and stability, let's use the rank-mu update carefully.
            
            # Compute sum of outer products of selected steps (normalized by sigma)
            # steps = (ys[:mu] - x_m) / sigma
            steps = (ys[:mu] - x_m) / sigma
            # Weighted sum of outer products
            # We can do: sum(w_i * step_i @ step_i.T)
            # This is (steps.T @ w) @ steps ? No.
            # It is sum_i w_i * (step_i @ step_i.T)
            # Which is (steps * w[:, np.newaxis]).T @ steps ?
            # Let A = steps (mu x dim). We want sum_i w_i A_i A_i^T.
            # This is A.T @ diag(w) @ A.
            # So: (steps.T * w) @ steps  -> (dim x mu) @ (mu x dim) -> (dim x dim)
            # Wait, steps.T is (dim x mu). w is (mu,).
            # steps.T * w broadcasts w to columns? No, w is 1D.
            # (steps.T * w) works if w is (mu,).
            rank_mu_term = (steps.T * w) @ steps
            
            # Combine updates
            # C = (1 - c_1 - c_mu * mueff) * C + c_1 * pc @ pc.T + c_mu * rank_mu_term
            # Note: The factor for C decay is (1 - c_1 - c_mu * mueff)
            # But we also have the h1 term. If h1 is false, we might drop the pc term or handle it differently.
            # Standard CMA-ES:
            # C = (1 - c_1 - c_mu) * C + c_1 * pc @ pc.T + c_mu * sum(w_i * y_i y_i^T)
            # where y_i are the selected steps.
            # The decay factor is actually (1 - c_1 - c_mu * mueff) in some variants?
            # Let's check the formula in Hansen's paper.
            # C_{t+1} = (1 - c_1 - c_mu) C_t + c_1 p_c p_c^T + c_mu sum_{i=1}^mu w_i y_{i:t} y_{i:t}^T
            # So decay is (1 - c_1 - c_mu).
            
            decay = 1 - c_1 - c_mu
            C = decay * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu_term
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update C_half using eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_half = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update mean
            x_m = x_m_new
            
            # Prevent sigma from becoming too large or small
            sigma = np.clip(sigma, 1e-12, 1e12)
            
            counter += 1
            
            # Check for convergence
            if np.linalg.norm(x_m_new - x_m) < 1e-12 * (1 + np.linalg.norm(x_m)):
                # Stagnant
                # We can break or restart. For now, just break to save time.
                # Or we can perturb.
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
