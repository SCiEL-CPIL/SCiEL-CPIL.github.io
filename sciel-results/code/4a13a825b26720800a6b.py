"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES with adaptive sigma and time guard.
    Uses a small population and restarts to handle the ill-conditioned landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def eval_point(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial broad search
        # Try a few random points to get a decent starting point
        for _ in range(10):
            if calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            eval_point(x)
        
        # CMA-ES loop with restarts
        # We will run CMA-ES from the best point found so far.
        # If stagnation is detected (no improvement for a while) or time runs out, we stop.
        
        # Parameters
        pop_size = 4 + int(np.floor(3 * np.log(dim)))
        pop_size = max(pop_size, 4)
        
        # We'll do a sequence of CMA-ES runs, restarting if stuck.
        # Since the function is ill-conditioned, a single run might get stuck.
        # However, with a budget of 500, we can't do many full CMA-ES runs.
        # Let's try one robust CMA-ES run first.
        
        # Initialize CMA-ES
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size
        
        # Covariance matrix
        C = np.eye(dim)
        # Precompute eigen decomposition for updates (not strictly needed for simple update)
        
        # Weights
        mu = pop_size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # History for stagnation detection
        best_f_history = []
        stagnation_limit = 50 # If no improvement for 50 evals, consider restarting or stopping
        
        current_best_f = best_f
        evals_since_improvement = 0
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break
            
            # Generate population
            xs = []
            for i in range(pop_size):
                # Sample from N(x_mean, sigma^2 C)
                # We use a simple Cholesky or just standard normal if C is identity initially.
                # For efficiency and simplicity in this budget, we might approximate or use full CMA.
                # Let's use full CMA update.
                
                # Eigen decomposition of C
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to avoid negative issues
                eigvals = np.maximum(eigvals, 1e-10)
                
                # Sample
                z = rng.standard_normal(dim)
                # Transform z
                y = x_mean + sigma * (eigvecs * np.sqrt(eigvals)) @ z
                y = np.clip(y, -5, 5)
                xs.append(y)
            
            # Evaluate
            fs = np.array([eval_point(x) for x in xs])
            
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = np.array(xs)[order]
            fs_sorted = fs[order]
            
            # Update best
            if fs_sorted[0] < current_best_f:
                current_best_f = fs_sorted[0]
                evals_since_improvement = 0
            else:
                evals_since_improvement += pop_size
            
            # Check stagnation
            if evals_since_improvement > stagnation_limit:
                # Try a local restart or just break if we've used enough budget
                if calls_made > budget * 0.8:
                    break
                else:
                    # Restart from best point with larger sigma?
                    # Or just continue with current CMA-ES but maybe reset C to I?
                    # Let's try a simple local perturbation restart
                    x_mean = best_x.copy()
                    sigma = max(sigma * 2.0, 0.5)
                    C = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    evals_since_improvement = 0
                    continue

            # CMA-ES Update
            # Weighted mean
            x_new = np.sum(w[:, np.newaxis] * xs_sorted[:mu], axis=0)
            
            # Update sigma
            # p_sigma update
            # Need inverse sqrt of C
            C_sqrt_inv = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_new - x_mean) / sigma
            
            # Hessian correction (approximate)
            # chi_n = sqrt(dim) * (1 + 1/(4*dim) - 1/(21*dim^2))
            chi_n = np.sqrt(dim) * (1 + 1/(4*dim) - 1/(21*dim**2))
            
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - 10))) < (1.4 + 2/(dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Rank-mu update
            hsigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - 10)))) / chi_n < (1.4 + 2/(dim + 1))
            hsigmac = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (calls_made - 10)))) / np.sqrt(c_c * (2 - c_c) * mu_eff) < (1.4 + 2/(dim + 1))
            
            if hsigmac:
                rank_mu = (c_1 + c_mu) * (np.outer(p_c, p_c) / (c_c * (2 - c_c) * mu_eff) - np.eye(dim))
            else:
                rank_mu = np.zeros((dim, dim))
            
            # Update C
            C = (1 + c_1 * (hsigma - 1) + c_mu) * C - (c_1 + c_mu) * np.outer(x_new - x_mean, x_new - x_mean) / sigma**2
            # Wait, standard update is:
            # C = (1 + c1*(hsigma-1) + cmu - c1 - cmu) * C + ...
            # Let's use the standard formula:
            C = (1 + c_1 * (hsigma - 1) + c_mu - c_1 - c_mu) * C
            # Add rank-1 update
            C += c_1 * (np.outer(p_c, p_c) / (c_c * (2 - c_c) * mu_eff) - (1 - hsigmac) * np.eye(dim))
            # Add rank-mu update
            C += c_mu * sum(w[i] * np.outer(xs_sorted[i] - x_mean, xs_sorted[i] - x_mean) / sigma**2 for i in range(mu))
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = (eigvecs * eigvals) @ eigvecs.T
            
            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
