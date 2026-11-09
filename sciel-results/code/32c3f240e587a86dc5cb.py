import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Global state
        best_x = np.zeros(n)
        best_f = np.inf
        eval_count = 0
        
        class _BudgetExceeded(Exception):
            pass
        
        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise _BudgetExceeded
            if eval_count >= budget:
                raise _BudgetExceeded
            f_val = func(np.array(x, dtype=np.float64))
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=np.float64)
            return f_val

        # Phase 1: Initial sampling to find a good starting point
        # For Lunacek bi-Rastrigin, explore both global and local minima regions
        n_samples = min(30, budget)
        rng = np.random.default_rng(42)
        
        # First, try origin and a few structured points
        safe_func(np.zeros(n))
        
        # Try points near the known Lunacek structure
        # The global minimum is typically near a shifted location
        for offset in [0.5, -0.5, 1.0, -1.0]:
            x = np.full(n, offset)
            if eval_count < budget:
                safe_func(x)
        
        # Then random samples with better coverage
        for i in range(1, n_samples):
            if eval_count >= budget:
                break
            x = rng.uniform(-5, 5, n)
            safe_func(x)

        if best_f == np.inf:
            return np.zeros(n), 0.0

        # Phase 2: CMA-ES with careful tuning
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size - smaller for Lunacek's narrow basin
        
        # CMA-ES parameters
        lam = 4 * n  # 20 for dim=5
        mu = lam // 2  # 10
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) - 1/(20*n**2))
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0, (mu_eff - 2) / (n + mu_eff)) + cs
        cmu = 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff)
        dsigma = 1.0
        
        # Covariance matrix
        C = np.eye(n)
        Cs = np.zeros(n)
        ps = np.zeros(n)
        
        def _cov_to_b_d(C_mat, n_dim):
            # C = B D^2 B^T
            # Use Cholesky: C = L L^T, so B = L, D = 1
            try:
                L = np.linalg.cholesky(C_mat)
                return L, np.ones(n_dim)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-12, None)
                return eigvecs, np.sqrt(eigvals)
        
        B, D = _cov_to_b_d(C, n)
        
        # Adaptive restart counter
        no_improve_count = 0
        max_no_improve = 15
        restart_count = 0
        max_restarts = 3
        
        while eval_count < budget and time.time() < deadline:
            # Generate lambda samples
            ys = []
            xs = []
            fs = []
            
            for k in range(lam):
                if time.time() > deadline:
                    break
                if eval_count >= budget:
                    break
                    
                z = rng.standard_normal(n)
                y = B @ (D * z)
                x = np.clip(x_mean + sigma * y, -5, 5)
                ys.append(y)
                xs.append(x)
                fs.append(safe_func(x))
                
            if eval_count >= budget:
                break
            
            # Check for timeout
            if time.time() > deadline:
                break
            
            # Sort by fitness
            fs_array = np.array(fs)
            idx = np.argsort(fs_array)
            
            # Track improvement
            if fs_array[idx[0]] < best_f - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # If no improvement for too long, restart with adjusted sigma
            if no_improve_count > max_no_improve:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Try different scales on restart
                if restart_count % 2 == 1:
                    sigma *= 2.0  # Expand
                else:
                    sigma *= 0.5  # Contract
                no_improve_count = 0
                # Reset some internal state
                ps = np.zeros(n)
                Cs = np.zeros(n)
                C = np.eye(n)
                B, D = _cov_to_b_d(C, n)
                continue
            
            # Calculate new mean
            w_sorted = w[:mu]
            selected_x = [xs[i] for i in idx[:mu]]
            new_mean = np.average(selected_x, axis=0, weights=w_sorted)
            
            # Calculate C^(-1/2) for ps update
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = (eigvecs * (1/np.sqrt(eigvals))) @ eigvecs.T
            
            # Update ps
            ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) * (C_inv_sqrt @ (new_mean - x_mean) / sigma)
            
            # Check hsig
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / chi_n - (1 - 0.5 * cs)) / (1 + cs) > 1.4
            hsig = 1.0 if hsig else 0.0
            
            # Update sigma
            sigma = sigma * np.exp((dsigma / ds) * (norm_ps / chi_n - 1))
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)
            
            # Update Cs
            Cs = (1 - cs) * Cs + hsig * cs * ps
            
            # Update C
            y_sel = np.array([ys[idx[i]] for i in range(mu)])
            y_bar = np.average(y_sel, axis=0, weights=w_sorted)
            
            # Term 1: Decay
            decay = 1 - cs * cs * (1 - ds) - cmu
            # Term 2: Cs Cs^T
            term_cs = cs * cs * (1 - ds) * np.outer(Cs, Cs)
            # Term 3: Weights * Outer products
            term_cmu = cmu * np.average([np.outer(y, y) for y in (y_sel - y_bar)], axis=0, weights=w_sorted)
            
            C = decay * C + term_cs + term_cmu
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = new_mean
            
            # Recompute B, D for next iteration
            B, D = _cov_to_b_d(C, n)
            
            # If sigma is too small, we're likely converged
            if sigma < 0.01:
                break
        
        # Phase 3: Final polish - local refinement around best point
        remaining = budget - eval_count
        if remaining > 10 and time.time() < deadline - 5:
            # Do a fine-grained local search
            polish_sigma = 0.1
            for _ in range(min(remaining // 2, 50)):
                if time.time() > deadline - 5 or eval_count >= budget:
                    break
                x = np.clip(best_x + rng.normal(0, polish_sigma, n), -5, 5)
                f_val = safe_func(x)
                # Gradually reduce step size
                polish_sigma *= 0.95
        
        return best_x, best_f
