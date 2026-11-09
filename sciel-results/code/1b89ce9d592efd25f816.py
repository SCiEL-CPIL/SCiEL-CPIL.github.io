"""CMA-ES optimizer for BBOB-style f17 (Schaffers F7) with warm start."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 10, warm-started with random initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Adaptation rates
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initialization Phase: Warm Start
        # Use 5 random points to initialize the search
        init_count = 5
        if budget <= init_count:
            # If budget is very small, just do random search
            best_x = None
            best_f = np.inf
            for _ in range(budget):
                x = rng.uniform(-5, 5, d)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x
            return best_x, best_f

        init_samples = rng.uniform(-5, 5, (init_count, d))
        init_fvals = [func(x) for x in init_samples]
        init_best_idx = np.argmin(init_fvals)
        x_mean = init_samples[init_best_idx].copy()
        f_best_val = init_fvals[init_best_idx]
        best_x = x_mean.copy()
        
        calls_made = init_count

        # Initialize CMA-ES state
        sigma = 1.0  # Start with a moderate step size
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        def _c_inv_sqrt(C_mat):
            """Compute inverse square root of a symmetric positive definite matrix."""
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.clip(eigvals, 1e-12, None)
            inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
            return eigvecs @ (inv_sqrt_eigvals[:, None] * eigvecs.T)

        while calls_made < budget:
            # Compute C^{-1/2}
            C_inv_sqrt = _c_inv_sqrt(C)
            
            # Generate population
            samples = []
            fvals = []
            
            for i in range(lambda_):
                z = rng.standard_normal(d)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                samples.append(y)
                fval = func(y)
                fvals.append(fval)
                calls_made += 1
                if calls_made >= budget:
                    break
            
            if len(samples) < mu:
                break
            
            # Sort by fitness
            idx = np.argsort(fvals[:len(samples)])
            sorted_samples = [samples[i] for i in idx[:mu]]
            sorted_fvals = [fvals[i] for i in idx[:mu]]
            
            # Update best
            if sorted_fvals[0] < f_best_val:
                f_best_val = sorted_fvals[0]
                best_x = sorted_samples[0].copy()
            
            # Weighted mean of best mu
            x_new = np.zeros(d)
            for w, s in zip(weights[:len(sorted_samples)], sorted_samples):
                x_new += w * s
            
            # Update path evolution strategies
            delta = (x_new - x_mean) / (sigma * np.sqrt(2.0)) if sigma > 1e-12 else np.zeros(d)
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * delta
            norm_pc = np.linalg.norm(p_c)
            
            if norm_pc / chi_d < (1.4 + 1 / c_c) * (1 - c_sigma):
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * delta
            else:
                p_sigma = (1 - c_sigma) * p_sigma
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(p_sigma) / chi_d - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            # Update covariance matrix
            mu_1 = mueff
            rank_mu_update = np.zeros((d, d))
            for w, s in zip(weights[:len(sorted_samples)], sorted_samples):
                v_i = (s - x_mean) / sigma
                rank_mu_update += w * np.outer(v_i, v_i)
            
            C = ((1 - c_1 - c_mu) * C + 
                 c_1 * (2 - mu_1) * np.outer(p_c, p_c) + 
                 c_mu * rank_mu_update)
            
            # Symmetrize and ensure PSD
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ (eigvals[:, None] * eigvecs.T)
            
            # Update mean
            x_mean = x_new
            
            # Optional: Restart if converged
            if sigma < 1e-5 and calls_made < budget * 0.8:
                # Restart with a new random center
                x_mean = rng.uniform(-5, 5, d)
                sigma = 1.0
                C = np.eye(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                # Do not reset best_x/f_best_val, we keep the global best

        return best_x, f_best_val
# EVOLVE-BLOCK-END
