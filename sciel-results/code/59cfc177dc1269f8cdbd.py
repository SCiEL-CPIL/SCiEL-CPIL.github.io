import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.calls = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

        # Helper to evaluate and track
        def evaluate(x):
            if self.calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = func(x)
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # CMA-ES Parameters
        # For dim=5, lambda=9 is standard, mu=4
        lam = 9
        mu = 4
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Strategy:
        # 1. Initial broad search with CMA-ES starting at origin, sigma=1.5
        # 2. If budget remains, do a fine-tuning phase with smaller sigma around best point.
        # 3. If CMA-ES fails or stagnates, fallback to Nelder-Mead or simple random restarts.
        
        # Phase 1: CMA-ES
        x_mean = np.zeros(dim)
        sigma = 1.5
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        f0 = evaluate(x_mean)
        
        # Precompute eigendecomposition helper
        def get_sample_matrix():
            try:
                # Use Cholesky for speed if possible, but Eigen is more robust for PD checks in loops if C is updated carefully.
                # Actually, for dim=5, eig is fine.
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive definiteness
                eigvals = np.maximum(eigvals, 1e-12)
                std_eig = np.sqrt(eigvals)
                return eigvecs, std_eig
            except:
                return np.eye(dim), np.ones(dim)

        # Main CMA-ES Loop
        while self.calls < budget:
            if time.time() > deadline:
                break
                
            n_off = min(lam, budget - self.calls)
            if n_off <= 0:
                break
            
            # Sample offspring
            eigvecs, std_eig = get_sample_matrix()
            
            offspring = []
            for i in range(n_off):
                z = np.random.randn(dim)
                y = x_mean + sigma * (eigvecs * std_eig) @ z
                fy = evaluate(y)
                offspring.append((y, fy))
                
            if time.time() > deadline:
                break
                
            # Sort
            offspring.sort(key=lambda t: t[1])
            
            # Update Mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * offspring[i][0]
            x_new_mean = y_w
            
            # Check for stagnation or improvement
            diff = x_new_mean - x_mean
            norm_diff = np.linalg.norm(diff)
            
            if norm_diff < 1e-12:
                # Stagnation, try to increase sigma or break
                sigma *= 1.5
                if sigma > 10:
                    break
                continue
                
            # Update Paths
            # C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            # h_sigma
            # Approximation for expected norm of p_sigma
            # E||p_sigma|| = sigma * sqrt(1 - (1-c_sigma)^2) ... actually it's chi_n * sqrt(1 - (1-c_sigma)^2) * ...
            # Standard check:
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^2) < chi_n * (1 + 2/(dim+1))
            # Let's use a simpler robust check:
            # If the step is "normal" size, h_sigma = 1
            # Theoretical expected norm of C^{-1/2}(x_new - x_old)/sigma is chi_n
            # So ||p_sigma|| should be around chi_n * sqrt(1 - (1-c_sigma)^2) ? No.
            # ||p_sigma|| is a random walk.
            # Let's stick to the standard formula approximation.
            # h_sigma = 1.0 if np.linalg.norm(p_sigma) < chi_n * (1 + 1e-3) else 0.0
            # This is a common simplification.
            
            # More accurate h_sigma:
            # chi_n is the expected norm of N(0, I).
            # The term (C^{-1/2} (x_new - x_old) / sigma) has expected norm chi_n.
            # p_sigma is a moving average.
            # Let's just use h_sigma = 1.0 for simplicity in this short budget, or a simple threshold.
            # If we detect a huge jump, we might want to reset paths.
            h_sigma = 1.0
            if np.linalg.norm(p_sigma) > chi_n * 2.0:
                 h_sigma = 0.0
                 # Reset paths if outlier
                 p_sigma = np.zeros(dim)
                 p_c = np.zeros(dim)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            # Update Sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update Covariance
            # Rank-1
            c1_term = c_1 * np.outer(p_c, p_c)
            
            # Rank-Mu
            cmu_term = np.zeros((dim, dim))
            for i in range(mu):
                y_i = offspring[i][0]
                diff_i = y_i - x_new_mean
                cmu_term += w[i] * np.outer(diff_i, diff_i)
            
            C = (1 - c_1 - c_mu + mu_eff * c_mu) * C + c1_term + c_mu * cmu_term
            C = (C + C.T) / 2 # Symmetrize
            
            x_mean = x_new_mean
            
            # Optional: Restart logic
            # If we've made very little progress, maybe restart?
            # For now, rely on sigma adaptation.

        # Phase 2: Fine-tuning with Nelder-Mead or Local Search
        # If we still have budget, do a local refinement around the best point found.
        if self.calls < budget and time.time() < deadline:
            # Use a simple local search or Nelder-Mead
            # Let's do a few random perturbations around best_x with decreasing step size
            current_best = self.best_x.copy()
            current_best_f = self.best_f
            
            # Try Nelder-Mead if available and budget allows
            # But Nelder-Mead can be expensive in calls.
            # Let's do a simple coordinate descent or random restarts.
            
            # Random restarts around best point
            # Generate a few points in a small ball around best_x
            for _ in range(10):
                if self.calls >= budget or time.time() > deadline:
                    break
                # Random point in small neighborhood
                # Step size proportional to current sigma? Or fixed small?
                # Let's use a fixed small step, e.g., 0.1
                x_try = current_best + 0.1 * np.random.randn(dim)
                f_try = evaluate(x_try)
                if f_try < self.best_f:
                    current_best = self.best_x.copy()
                    current_best_f = self.best_f
                    # If improved, maybe do more local search
                    # For now, just continue loop
            
        return self.best_x, self.best_f
