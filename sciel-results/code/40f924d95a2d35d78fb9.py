"""Ridge-aware optimizer: CMA-ES with large initial sigma for robust shift handling and rapid ridge adaptation."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES optimizer tailored for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        # CMA-ES Parameters
        # lambda: population size
        # For dim=5, default is 8. We use 10 for slightly faster convergence per generation.
        lam = 10
        # mu: number of weighted points
        mu = lam // 2
        
        # Weights: proportional to log rank
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial mean and sigma
        # Start at 0, but with a large sigma to cover the box [-5, 5]^5
        # If the shift is large, this allows exploration.
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Cholesky factor of C
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(dim)
            
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation parameters (standard CMA-ES values)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        max_iter = 1000 # Safety, will break on budget

        for it in range(max_iter):
            if not time_ok() or calls_used >= budget:
                break

            # 1. Sample population
            ys = rng.multivariate_normal(np.zeros(dim), C, size=lam)
            # Alternatively, use Cholesky for speed and stability:
            # ys = (rng.normal(size=(lam, dim)) @ C_sqrt.T)
            xs = np.array([x_mean + sigma * y for y in ys])
            
            # Evaluate
            fs = []
            for x in xs:
                if calls_used >= budget:
                    break
                x_clipped = clip(x)
                f_val = func(x_clipped)
                calls_used += 1
                fs.append(f_val)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
            
            if len(fs) < lam:
                break
                
            fs = np.array(fs)
            
            # 2. Sort by fitness
            order = np.argsort(fs)
            xs = xs[order]
            fs = fs[order]
            
            # 3. Update mean
            y_w = np.sum(weights[:, np.newaxis] * (xs[:mu] - x_mean), axis=0)
            x_mean = x_mean + sigma * y_w
            
            # 4. Update sigma (CSA)
            y_sigma = np.sum(weights[:, np.newaxis] * (xs[:mu] - x_mean + sigma * y_w) / sigma, axis=0)
            # Actually, standard update:
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * (y_w / sigma)
            # Wait, y_w is already scaled by sigma in the mean update? No.
            # Let's stick to the standard formulas carefully.
            
            # Recalculate y_w without sigma scaling for the path
            # y_w = sum w_i (x_i - old_mean) / sigma
            y_w_scaled = np.sum(weights[:, np.newaxis] * (xs[:mu] - (x_mean - sigma * y_w)), axis=0) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w_scaled
            
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(it+1)}) < (1.4 + 2/(dim+1)) * E[||N(0,I)||]
            # Approximate E[||N||]
            e_norm = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            # For simplicity and robustness, we can skip h_sigma check or use a fixed threshold
            # Given the short run, skipping h_sigma is often acceptable, but let's include a simple version.
            # Actually, for dim=5 and short runs, we can just update sigma.
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / e_norm - 1))
            
            # 5. Update p_c
            h_c = 1 # Simplification, usually checks alignment
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w_scaled
            
            # 6. Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_c) c_c (2 - c_c) C) + c_mu sum w_i y_i y_i^T
            # Note: y_i here are the normalized steps (x_i - old_mean) / sigma
            ys_sorted = (xs[:mu] - (x_mean - sigma * y_w)) / sigma
            
            delta_h = 1 - h_c
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_h * c_c * (2 - c_c) * C) + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(ys_sorted, ys_sorted), axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small identity if not positive definite
                C = C + 1e-8 * np.eye(dim)
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)
                    C = np.eye(dim)
                    break # Reset if totally broken, but unlikely to happen in 500 calls

        return best_x, best_f
# EVOLVE-BLOCK-END
