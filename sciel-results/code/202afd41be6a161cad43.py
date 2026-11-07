"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            return fx

        def cmaes_search(x0, sigma0, max_calls):
            """
            Runs a CMA-ES instance starting from x0 with initial sigma.
            Returns (best_x, best_f, calls_made)
            """
            nonlocal best_x, best_f, call_count
            
            n = self.dim
            # CMA-ES parameters
            lam = max(10, 4 + int(3 * np.log(n)))  # Population size
            mu = lam // 2                           # Number of parents
            # Weights (standard log-linear weights)
            w = np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            # Strategy parameters
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # State
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            invsqrtC = np.eye(n)
            
            # Initial evaluation
            fx0 = make_call(mean)
            if fx0 is None:
                return mean, np.inf, 1
            
            if np.isfinite(fx0) and fx0 < best_f:
                best_f = fx0
                best_x = mean.copy()
            
            calls_in_this_run = 1
            stagnation_counter = 0
            last_improvement_f = fx0
            last_improvement_calls = 1
            
            while calls_in_this_run < max_calls:
                if call_count >= self.budget:
                    break
                if time.time() > deadline - 2.0:
                    break
                
                # Sample offspring
                ys = np.zeros((lam, n))
                fs = np.zeros(lam)
                
                for i in range(lam):
                    z = rng.standard_normal(n)
                    # Cholesky decomposition for sampling (approximated by invsqrtC for speed in small n)
                    # For n=5, full Cholesky is cheap
                    try:
                        L = np.linalg.cholesky(C)
                        y = mean + sigma * (L @ z)
                    except np.linalg.LinAlgError:
                        y = mean + sigma * z
                    
                    y = np.clip(y, -5, 5)
                    fi = make_call(y)
                    if fi is None:
                        break
                    ys[i] = y
                    fs[i] = fi
                    
                    if np.isfinite(fi) and fi < best_f:
                        best_f = fi
                        best_x = y.copy()
                        last_improvement_f = fi
                        last_improvement_calls = calls_in_this_run
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                    
                    calls_in_this_run += 1
                
                if call_count >= self.budget:
                    break
                    
                # Sort by fitness
                idx = np.argsort(fs)
                ys = ys[idx]
                fs = fs[idx]
                
                # Update mean
                new_mean = np.sum(w[:, np.newaxis] * ys[:mu], axis=0)
                
                # Update step size control (CS)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ (new_mean - mean) / sigma)
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)) else 0
                
                # Update covariance matrix
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (new_mean - mean) / sigma
                
                rank_one = np.outer(pc, pc)
                rank_mu = np.sum(w[:, np.newaxis, np.newaxis] * np.outer(ys[:mu] - mean, ys[:mu] - mean), axis=0)
                
                C = (1 + c1 * hsig * (cc - 1) - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                
                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update inverse square root
                try:
                    invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
                except np.linalg.LinAlgError:
                    invsqrtC = np.eye(n)
                
                # Update sigma
                sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
                
                # Update mean
                mean = new_mean
                
                # Convergence check
                if np.linalg.norm(mean - x0) < 1e-6 * sigma or np.all(np.abs(mean) < 1e-6):
                    break
                
                # Restart condition: if no improvement for a while
                if stagnation_counter > lam * 3:
                    break

            return best_x, best_f, calls_in_this_run

        # Strategy:
        # 1. Initial probe to get a baseline and maybe find a good starting point.
        # 2. Run CMA-ES from that point.
        # 3. If budget remains and stagnation occurred, restart from best found so far with smaller sigma.

        # Initial probe: Center
        x_start = np.zeros(self.dim)
        fx_start = make_call(x_start)
        if fx_start is None:
            return None, np.inf
        
        if np.isfinite(fx_start) and fx_start < best_f:
            best_f = fx_start
            best_x = x_start.copy()

        # First CMA-ES run
        # Use a generous initial sigma
        sigma_init = 2.0 
        max_calls_first = 200
        x_best, f_best, calls_made = cmaes_search(x_start, sigma_init, max_calls_first)
        
        # If we have budget left, do a second run with refined sigma around the best found
        if best_x is not None and call_count < self.budget - 10:
            # Reduce sigma for local exploitation
            sigma_refined = max(0.5, sigma_init * 0.2)
            x_start2 = best_x.copy()
            # We need to re-evaluate the start point? No, we have the value.
            # But CMA-ES implementation above evaluates the start point. 
            # To avoid double counting or wasting a call, we can just start the loop 
            # but the internal make_call will handle the count.
            # However, the internal cmaes_search evaluates mean first.
            # So we will spend 1 call.
            max_calls_second = self.budget - call_count - 10
            if max_calls_second > 20:
                x_best2, f_best2, calls_made2 = cmaes_search(x_start2, sigma_refined, max_calls_second)
        
        # Final fallback if something went wrong
        if best_x is None:
            x = np.zeros(self.dim)
            fx = make_call(x)
            if fx is not None and np.isfinite(fx):
                best_x = x.copy()
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
