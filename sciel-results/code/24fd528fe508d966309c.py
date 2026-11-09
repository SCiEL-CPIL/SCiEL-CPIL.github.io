"""Combined optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard for BBOB-style f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before the 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim

        def elapsed_ok():
            return time.time() < deadline

        # --- Initialization ---
        # Start near the center, as f13 is typically centered or shifted but not extreme
        x = rng.uniform(-1, 1, dim)
        f_best = float(func(x))
        x_best = x.copy()
        
        # CMA-ES parameters
        # Lambda: population size
        lam = 4 + int(3 * np.log(dim))
        if lam < 4:
            lam = 4
        if lam > 10:
            lam = 10
            
        sigma = 1.0
        # Covariance matrix (initially identity)
        C = np.eye(dim)
        # Inverse square root of C, initially identity
        C_inv_sqrt = np.eye(dim)
        
        # Evolution path for sigma adaptation
        p_sigma = np.zeros(dim)
        # Evolution path for covariance matrix adaptation
        p_c = np.zeros(dim)
        
        # Weights for ranking
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Learning rates
        c_sigma = (mu + 0.5) / (dim + 2)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu + 2) / (dim + 2) ** 2 - 1))
        
        # Damping factor
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        
        calls_made = 1
        no_improve = 0
        max_no_improve = 50 # Stall detection for restarts if CMA-ES gets stuck

        while calls_made < self.budget:
            if not elapsed_ok():
                break

            # Generate offspring
            # y_i = x + sigma * B * z_i
            z = rng.standard_normal((lam, dim))
            # Transform z using C_inv_sqrt
            # y = x + sigma * (C_inv_sqrt @ z.T).T
            y = x + sigma * (z @ C_inv_sqrt.T)
            y = np.clip(y, -5, 5)
            
            # Evaluate offspring
            fy = np.array([float(func(yi)) for yi in y])
            calls_made += lam
            
            # If we exceeded budget or time during evaluation, we can't update properly, 
            # but we can still record the best found.
            if not elapsed_ok() or calls_made >= self.budget:
                # Check if any new point is better
                min_idx = np.argmin(fy)
                if fy[min_idx] < f_best:
                    f_best = fy[min_idx]
                    x_best = y[min_idx].copy()
                break

            # Sort offspring
            order = np.argsort(fy)
            y_sorted = y[order]
            fy_sorted = fy[order]
            
            # Update best
            if fy_sorted[0] < f_best:
                f_best = fy_sorted[0]
                x_best = y_sorted[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Update mean
            x_old = x.copy()
            x = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Update evolution paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x - x_old) / sigma
            mu_eff = 1 / np.sum(weights**2)
            c_sigma_eff = c_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x - x_old) / sigma)
            
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x - x_old) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (x - x_old) / sigma)
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - (1 - c_sigma)^2) * C) + c_mu * sum w_i y_i y_i^T
            # Note: The term (1 - (1 - c_sigma)^2) * C accounts for the change in p_sigma
            # Actually, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - (1 - c_sigma)^2) * C) + c_mu * sum w_i (y_i - x_old)(y_i - x_old)^T
            # Wait, the standard CMA-ES update uses the ranked differences from the OLD mean.
            # Let's use the simplified rank-mu update which is numerically stable.
            
            # Recalculate C_inv_sqrt periodically or use the inverse square root of the updated C.
            # For simplicity and speed in Python, we can recompute C_inv_sqrt every few steps or use a cheaper update.
            # However, for dim=5, computing eigenvalues is fast.
            
            # Let's do the full update:
            # Delta C = c_1 * (p_c p_c^T + (1 - (1 - c_sigma)^2) * C) + c_mu * sum w_i (y_i - x_old)(y_i - x_old)^T
            # C = (1 - c_1 - c_mu) * C + Delta C
            
            # Compute rank-1 update
            rank1 = np.outer(p_c, p_c)
            # The term (1 - (1 - c_sigma)^2) * C is equivalent to (2*c_sigma - c_sigma^2) * C
            # This term ensures positive definiteness and accounts for the sigma adaptation effect on C.
            rank1 += (1 - (1 - c_sigma)**2) * C
            
            # Compute rank-mu update
            # We need the differences y_i - x_old
            diffs = y_sorted[:mu] - x_old
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(diffs[i], diffs[i])
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            # sigma = sigma * exp(c_sigma/d_sigma * (||p_sigma||/chi_N - 1))
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1))
            
            # Keep sigma in a reasonable range
            if sigma < 1e-7:
                sigma = 1e-7
            if sigma > 10.0:
                sigma = 10.0

            # Update C_inv_sqrt
            # We need the inverse square root of C.
            # Eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues to 0
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 1.0

            # Stall handling: If no improvement for a long time, restart with a new center
            if no_improve >= max_no_improve:
                # Try a random restart
                x_new = rng.uniform(-5, 5, dim)
                fx_new = float(func(x_new))
                calls_made += 1
                if fx_new < f_best:
                    x = x_new
                    f_best = fx_new
                    x_best = x.copy()
                    sigma = 1.0
                    C = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    no_improve = 0
                else:
                    # If restart didn't help, shrink sigma and continue
                    sigma *= 0.5
                    no_improve = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
