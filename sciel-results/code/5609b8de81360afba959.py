import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s margin before 300s limit
        
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
            try:
                f_val = func(np.array(x, dtype=np.float64))
                if not np.isfinite(f_val):
                    f_val = 1e10
            except Exception:
                f_val = 1e10
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=np.float64)
            return f_val

        # Phase 1: Initial sampling to find good starting points for both minima
        n_samples = min(40, budget)
        rng = np.random.default_rng(42)
        
        # Try origin
        safe_func(np.zeros(n))
        
        # Try structured points near known Lunacek minima regions
        for offset in [0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
            x = np.full(n, offset)
            if eval_count < budget:
                safe_func(x)
        
        # Random samples with better coverage
        for i in range(1, n_samples):
            if eval_count >= budget:
                break
            x = rng.uniform(-5, 5, n)
            safe_func(x)

        if best_f == np.inf:
            return np.zeros(n), 0.0

        # Phase 2: Dual CMA-ES - one for global (smooth) min, one for local (Rastrigin) min
        # Split budget: 60% for global-target, 40% for local-target
        budget_global = int(budget * 0.6) - eval_count
        budget_local = budget - eval_count - budget_global
        if budget_global < 20:
            budget_global = 20
            budget_local = budget - eval_count - budget_global
        if budget_local < 10:
            budget_local = 10

        def run_cmaes(x_start, sigma_init, max_evals, rng_seed):
            """Run CMA-ES from x_start with given sigma for max_evals evaluations."""
            local_eval_count = 0
            local_best_f = best_f  # Start from current best
            local_best_x = best_x.copy()
            
            # CMA-ES parameters
            lam = 4 * n
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1 / np.sum(w**2)
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) - 1/(20*n**2))
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            ds = 1 + 2 * max(0, (mu_eff - 2) / (n + mu_eff)) + cs
            cmu = 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff)
            
            x_mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(n)
            Cs = np.zeros(n)
            ps = np.zeros(n)
            
            def _cov_to_b_d(C_mat, n_dim):
                try:
                    L = np.linalg.cholesky(C_mat)
                    return L, np.ones(n_dim)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C_mat)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    return eigvecs, np.sqrt(eigvals)
            
            B, D = _cov_to_b_d(C, n)
            
            no_improve_count = 0
            max_no_improve = 12
            
            while local_eval_count < max_evals and time.time() < deadline:
                # Generate lambda samples
                ys = []
                xs = []
                fs = []
                
                for k in range(lam):
                    if time.time() > deadline or local_eval_count >= max_evals:
                        break
                    
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    x = np.clip(x_mean + sigma * y, -5, 5)
                    ys.append(y)
                    xs.append(x)
                    fs.append(safe_func(x))
                    local_eval_count += 1
                    
                if local_eval_count >= max_evals or time.time() > deadline:
                    break
                
                if len(fs) == 0:
                    break
                
                # Sort by fitness
                fs_array = np.array(fs)
                idx = np.argsort(fs_array)
                
                # Track improvement
                if fs_array[idx[0]] < local_best_f - 1e-10:
                    local_best_f = fs_array[idx[0]]
                    local_best_x = xs[idx[0]].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # If no improvement, restart with adjusted sigma
                if no_improve_count > max_no_improve:
                    sigma *= 1.5
                    no_improve_count = 0
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
                hsig = 1.0 if (norm_ps / chi_n - (1 - 0.5 * cs)) / (1 + cs) > 1.4 else 0.0
                
                # Update sigma
                sigma = sigma * np.exp((1.0 / ds) * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 0.005, 10.0)
                
                # Update Cs
                Cs = (1 - cs) * Cs + hsig * cs * ps
                
                # Update C
                y_sel = np.array([ys[idx[i]] for i in range(mu)])
                y_bar = np.average(y_sel, axis=0, weights=w_sorted)
                
                decay = 1 - cs * cs * (1 - ds) - cmu
                term_cs = cs * cs * (1 - ds) * np.outer(Cs, Cs)
                term_cmu = cmu * np.average([np.outer(y, y) for y in (y_sel - y_bar)], axis=0, weights=w_sorted)
                
                C = decay * C + term_cs + term_cmu
                C = (C + C.T) / 2
                
                # Update mean
                x_mean = new_mean
                
                # Recompute B, D for next iteration
                B, D = _cov_to_b_d(C, n)
                
                if sigma < 0.005:
                    break
            
            return local_best_x, local_best_f, local_eval_count

        # Run global-target CMA-ES (smaller sigma, targeting smooth global min)
        rng_global = np.random.default_rng(123)
        gx, gf, ge = run_cmaes(best_x.copy(), 0.5, budget_global, rng_global)
        
        # Run local-target CMA-ES (larger sigma, targeting Rastrigin local min)
        rng_local = np.random.default_rng(456)
        # Start local search from a different point if possible
        local_start = best_x.copy()
        # Try to find a point in the Rastrigin region
        if eval_count < budget - 5:
            for trial in range(3):
                x_try = best_x + rng.uniform(-2, 2, n)
                x_try = np.clip(x_try, -5, 5)
                f_try = safe_func(x_try)
                if f_try < 50:  # Reasonable value, might be in local min region
                    local_start = x_try.copy()
                    break
        
        lx, lf, le = run_cmaes(local_start, 1.5, budget_local, rng_local)

        # Phase 3: Nelder-Mead refinement on the best point found
        remaining = budget - eval_count
        if remaining > 5 and time.time() < deadline - 5:
            # Use Nelder-Mead for final polish
            from scipy.optimize import minimize
            
            # Try NM from the best point found
            try:
                res = minimize(
                    lambda x: safe_func(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining, 100), 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass

        return best_x, best_f
