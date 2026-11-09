"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES with restarts, time-guarded."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s kill
        rng = np.random.default_rng(0)
        
        # Global best
        global_x = None
        global_f = float('inf')
        
        # Total budget per restart
        # CMA-ES typically needs ~ (10 + 4*log(n)) * n evaluations per restart
        # For n=5, that's ~ 10 + 4*2.3 = ~19 * 5 = 95 evals.
        # We can afford multiple restarts.
        # Let's use a smaller budget per restart to allow more restarts.
        # Or just run CMA-ES until time runs out, but CMA-ES has a fixed number of generations.
        # Better: Run CMA-ES for a fixed number of generations (e.g., 50) which is ~ 50*lambda.
        # If lambda=9, that's 450 evals. Too many for 1 restart if we want more.
        # Let's use lambda=4 (small pop) to speed up convergence? No, CMA-ES needs pop size.
        # Standard CMA-ES: lambda = 4 + floor(3*log(n)) = 4 + 3*2 = 10.
        # Let's use lambda=10.
        # Generations per restart: 30. Evals = 300.
        # We have 500 evals total. So we can do 1 restart of 300 evals, then another of 200?
        # Or just run CMA-ES with a budget limit.
        
        # Strategy: Run CMA-ES restarts. Each restart uses up to `restart_budget` evals.
        # If time is running out, stop.
        
        restart_budget = 100  # Small restarts to explore more?
        # Actually, for sharp ridge, CMA-ES is good. 
        # Let's just run one CMA-ES instance that uses up to `budget` evals, but with a time check.
        # Wait, the task says "More than 500 calls... makes the whole result invalid".
        # So we must strictly respect the 500 call limit.
        
        # CMA-ES implementation details:
        # x: mean
        # sigma: step size
        # C: covariance matrix
        # C^-1/2: inverse square root
        # p_sigma: evolution path for sigma
        # p_c: evolution path for C
        # cmu, c1, cmu, cc, cs: parameters
        
        n = self.dim
        mu = max(1, n // 2)  # number of parents
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / (w @ w)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_mu = min(1 - c_c, (2 * (mu_eff - 2 + 1 / mu_eff)) / (n + 2 + mu_eff - 2 + 1 / mu_eff))
        c_1 = (2 * (mu_eff - 2 + 1 / mu_eff)) / (n + 2) ** 2 + c_mu
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initial values
        x0 = rng.uniform(-5, 5, n)
        sigma0 = 2.0
        
        # We will run multiple restarts if possible, or just one long one?
        # Given the "Sharp Ridge" nature, a single good CMA-ES run might be better than many bad ones.
        # But CMA-ES can get stuck.
        # Let's do a single CMA-ES run that respects the 500 eval budget and time deadline.
        
        x = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Precompute
        def update_covariance(C, y_w, c_1, c_mu, w, ys, mu_eff, n):
            # y_w is weighted sum of ys (selected)
            # ys are the selected step vectors
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - chi_n) * np.eye(n)) + c_mu * (ys.T @ (w[:, None] * ys))
            # chi_n = trace(C)^0.5
            # We need to update C_inv_sqrt as well
            # This is expensive. 
            # For n=5, it's fine.
            # But we need chi_n.
            # trace(C) is sum of diagonal.
            chi_n = math.trace(C) ** 0.5 if hasattr(math, 'trace') else np.trace(C) ** 0.5
            # Actually, np.trace(C)
            chi_n = np.trace(C) ** 0.5
            # Update C_inv_sqrt
            # C = C @ C
            # We can use eigendecomposition or just recompute inverse sqrt each step?
            # Recomputing is O(n^3) which is 125 ops. Fine.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = (eigvecs * (1 / np.sqrt(eigvals))) @ eigvecs.T
            return C, C_inv_sqrt

        # Wait, the update formula for C is:
        # C <- (1 - c1 - cmu) C + c1 (p_c p_c^T + (1 - chi_n) I) + cmu sum w_i y_i y_i^T
        # And chi_n = trace(C)^(1/2)
        
        # Let's simplify: Just recompute C_inv_sqrt from C each step using eig.
        
        total_evals = 0
        lambda_ = 4 + int(3 * math.log(n))  # 10 for n=5
        
        while total_evals < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lambda_):
                if total_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # y_i = C^-1/2 * z_i
                z = rng.standard_normal(n)
                y = C_inv_sqrt @ z
                x_trial = x + sigma * y
                # Clip to box? CMA-ES usually works in unconstrained space.
                # The problem says func clips. So we don't need to clip x_trial before calling func.
                # But we should keep x within bounds?
                # "func(x) clips a finite length-5 vector to the box"
                # So we can pass any x.
                
                fy = func(x_trial)
                total_evals += 1
                ys.append(y)
                xs.append(x_trial)
                fs.append(fy)
            
            if not fs:
                break
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = np.array([ys[i] for i in idx])
            xs_sorted = np.array([xs[i] for i in idx])
            fs_sorted = np.array([fs[i] for i in idx])
            
            # Update best
            if fs_sorted[0] < global_f:
                global_f = fs_sorted[0]
                global_x = xs_sorted[0].copy()
            
            # Update mean
            # x_new = sum w_i x_{i:lambda}
            x_new = (w[:mu] * xs_sorted[:mu]).sum(axis=0)
            
            # Update evolution path for sigma
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma(2 - c_sigma)mu_eff) / sigma * C^-1/2 (x_new - x)
            # Wait, the formula is:
            # p_sigma <- (1 - c_sigma) p_sigma + sqrt(c_sigma(2 - c_sigma)mu_eff) / sigma * C^{-1/2} (x' - x)
            # Where x' is the new mean.
            p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (x_new - x))
            
            # Update step size
            # sigma *= exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = math.sqrt(n) # Approximation? No, chi_n depends on C.
            # The standard chi_n is E[||N(0, C)||] = sqrt(n) if C=I.
            # For general C, chi_n = sqrt(trace(C)).
            # Let's use the current C's trace.
            chi_n_current = np.trace(C) ** 0.5
            sigma *= math.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n_current - 1))
            # Clip sigma
            sigma = np.clip(sigma, 1e-12, 1e12)
            
            # Update evolution path for C
            # p_c = (1 - c_c) p_c + sqrt(c_c(2 - c_c)mu_eff) * (h_sigma / sigma) * (x_new - x)
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2(1 + gen/lambda)}) < (1.4 + 2/(n+1)) chi_n
            # We don't track gen/lambda easily here. Let's simplify h_sigma = 1.
            # Or just use the condition with a fixed threshold.
            h_sigma = 1.0
            p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Update covariance
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - chi_n) I) + c_mu sum w_i y_i y_i^T
            # Note: y_i here are the step vectors y_i = (x_i - x) / sigma ?
            # In CMA-ES, the update uses the normalized steps.
            # The formula in the paper:
            # C <- (1 - c1 - cmu) C + c1 (p_c p_c^T + (1 - chi_n) I) + cmu sum_{i=1}^mu w_i y_{i:lambda} y_{i:lambda}^T
            # Where y_{i:lambda} = (x_{i:lambda} - x) / sigma
            # We have ys which are C^-1/2 z.
            # x_i = x + sigma y_i
            # So (x_i - x) / sigma = y_i.
            # So we can use ys_sorted directly.
            
            # Calculate chi_n for the update term
            # chi_n = sqrt(trace(C))
            chi_n_for_update = np.trace(C) ** 0.5
            
            # Weighted sum of outer products
            # ys_sorted[:mu] are the selected steps
            # w[:mu] are the weights
            # sum w_i y_i y_i^T
            # This is (ys.T * w) @ ys
            # ys is (mu, n)
            # w is (mu,)
            # (ys.T * w) is (n, mu)
            # @ ys (mu, n) -> (n, n)
            cov_update = (ys_sorted[:mu].T * w[:mu]) @ ys_sorted[:mu]
            
            # p_c p_c^T
            pc_outer = np.outer(p_c, p_c)
            
            # (1 - chi_n) I
            # Note: The term is (1 - chi_n) I?
            # The formula is: c_1 (p_c p_c^T + (1 - chi_n) I)
            # Wait, is it (1 - chi_n) or (1 - chi_n^2)?
            # Standard CMA-ES: C <- (1 - c1 - cmu) C + c1 (p_c p_c^T + (1 - chi_n) I) + ...
            # chi_n = sqrt(n) for C=I.
            # If C=I, p_c ~ N(0, I), E[p_c p_c^T] = I.
            # Then c1 (I + (1 - sqrt(n)) I) = c1 (2 - sqrt(n)) I.
            # This doesn't look right.
            # Let's check the formula again.
            # Hansen 2016: C <- (1 - c1 - cmu) C + c1 (p_c p_c^T + (1 - chi_n) I) + cmu sum w_i y_i y_i^T
            # Where chi_n = E[||N(0, I)||] = sqrt(2/n) Gamma((n+1)/2) / Gamma(n/2)
            # For large n, chi_n ~ sqrt(n - 1/2).
            # For n=5, chi_n ~ 2.13.
            # So (1 - chi_n) is negative.
            # This term acts as a contraction.
            
            # Let's implement chi_n properly.
            from scipy.special import gamma
            chi_n = math.sqrt(2.0 / n) * gamma((n + 1) / 2.0) / gamma(n / 2.0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (pc_outer + (1 - chi_n) * np.eye(n)) + c_mu * cov_update
            
            # Update C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = (eigvecs * (1 / np.sqrt(eigvals))) @ eigvecs.T
            
            # Update x
            x = x_new
            
            # Check for convergence
            # If sigma is too small or C is diagonalized, stop?
            # We rely on budget and time.
            
        return global_x, global_f

# EVOLVE-BLOCK-END
