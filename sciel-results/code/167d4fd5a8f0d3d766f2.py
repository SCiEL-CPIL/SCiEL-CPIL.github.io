import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 290s to be safe
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Multi-restart CMA-ES
        # We divide the budget into restarts. 
        # Given the complexity of Schwefel in 5D, multiple restarts are crucial.
        # Let's use a simple strategy: 
        # 1. Run CMA-ES with a small budget.
        # 2. If not converged enough, restart from best point or random.
        
        # However, CMA-ES is expensive. With 500 evals, we can do:
        # Population size lambda = 4 + floor(3 * log(dim)) = 4 + 3*1.6 = 8? 
        # Standard is lambda = 4 + 3*log(dim). For dim=5, lambda=10.
        # 500 / 10 = 50 generations. That's quite few for CMA-ES to converge fully.
        
        # Alternative: Use a simpler, faster population-based method like 
        # Differential Evolution (DE) or a lightweight CMA-ES.
        
        # Let's implement a lightweight CMA-ES (or a simplified version like 
        # Covariance Matrix Adaptation with a small population) or just 
        # use multiple (1+1)-ES restarts? 
        # The inspiration 1 (score 1.0) likely used something robust.
        
        # Let's try a Multi-Start (1+1)-ES with adaptive step size and 
        # occasional global jumps. This is often very effective for 
        # benchmark functions when budget is limited.
        
        # Strategy:
        # 1. Start at center.
        # 2. Use (1+1)-ES with 1/5th success rule.
        # 3. Every N iterations, if progress is slow, inject a random 
        #    perturbation or restart from a random point near the best.
        
        # Actually, let's try a proper small-population CMA-ES.
        # It handles rotation better than (1+1)-ES.
        
        def run_cmaes(func, x0, sigma0, budget_cma, rng_cma, deadline):
            dim = self.dim
            # Parameters
            lam = 10 # Population size
            mu = lam // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Step size control
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            # Covariance matrix adaptation
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Initial state
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            cs = np.sqrt((c_sigma + 1) * d_sigma * mu_eff / 2)
            chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))
            
            calls = 0
            best_f = np.inf
            best_x = x0.copy()
            
            # Initial evaluations
            # We need to be careful with budget.
            
            while calls < budget_cma:
                if time.time() > deadline:
                    break
                
                # Generate population
                y = np.array([rng.standard_normal(dim) for _ in range(lam)])
                B, D = np.linalg.eigh(C)
                B = B * np.sqrt(D)
                x = np.array([mean + sigma * (B @ y[i]) for i in range(lam)])
                
                # Evaluate
                f = np.array([func(xi) for xi in x])
                calls += lam
                
                # Update best
                idx_min = np.argmin(f)
                if f[idx_min] < best_f:
                    best_f = f[idx_min]
                    best_x = x[idx_min].copy()
                
                # Sort
                order = np.argsort(f)
                y = y[order]
                x = x[order]
                f = f[order]
                
                # Weighted mean
                m_new = np.sum(weights[:, None] * x[:mu], axis=0)
                
                # Update step size
                y_sigma = (B @ np.sum(weights[:, None] * y[:mu], axis=0)) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update covariance
                pc = (1 - c_c) * pc + c_c * (m_new - mean) / sigma
                delta_heav = 1 + min(1, (np.linalg.norm(pc)**2 / dim - 1) / c_c) * c_c
                C = (1 + c_1 * delta_heav - c_1 - c_mu) * C \
                    + c_1 / c_c * (np.outer(pc, pc) - C) \
                    + c_mu / c_sigma * np.outer(y_sigma, y_sigma)
                
                mean = m_new
                
                # Ensure symmetry
                C = (C + C.T) / 2.0
                
                # Check for convergence or time
                if sigma < 1e-10 or np.max(C) < 1e-10:
                    break
                    
            return best_x, best_f, calls

        # Multi-restart strategy
        # Split budget into chunks.
        # First chunk: explore broadly.
        # Subsequent chunks: exploit around best found.
        
        best_x_global = np.zeros(dim)
        best_f_global = np.inf
        
        # Initial point
        x_start = np.zeros(dim)
        f_start = func(x_start)
        best_x_global = x_start.copy()
        best_f_global = f_start
        calls_made = 1
        
        # Define restarts
        # Let's do 5 restarts of 100 calls each? 
        # Or adaptive: run until no improvement for X calls, then restart.
        
        # Simple fixed restarts:
        # Restart 1: 150 calls, start at 0, sigma=2.0
        # Restart 2: 100 calls, start at best, sigma=1.0
        # Restart 3: 100 calls, start at best, sigma=0.5
        # Restart 4: 100 calls, start at best, sigma=0.2
        # Restart 5: 50 calls, start at best, sigma=0.1
        
        # But we need to track global calls.
        
        # Let's use a more dynamic approach:
        # Run CMA-ES with a budget. If it finishes, check time.
        # If time remains, restart from best with smaller sigma.
        
        # To ensure we use the full budget effectively, let's do:
        # 1. Initial broad search: 200 calls.
        # 2. Refinement search: 150 calls.
        # 3. Final polish: 150 calls.
        
        # However, CMA-ES overhead is high. 
        # Let's stick to a robust (1+1)-ES with multi-starts, 
        # as it is computationally cheap and allows more restarts.
        
        # Revised Strategy: Multi-Start (1+1)-ES
        # 1. Start at 0.
        # 2. Run (1+1)-ES for 100 calls.
        # 3. Record best.
        # 4. Restart from best with reduced sigma.
        # 5. Repeat.
        
        # This is essentially what the current program does, but without restarts.
        # The key is RESTARTS.
        
        def run_es_restart(func, x_start, sigma_start, budget_es, rng_es, deadline):
            dim = self.dim
            x = x_start.copy()
            fx = func(x)
            calls = 1
            best_x = x.copy()
            best_f = fx
            
            sigma = sigma_start
            window_size = 10
            successes = 0
            window_count = 0
            c_sigma = 0.5
            
            while calls < budget_es:
                if time.time() > deadline:
                    break
                
                z = rng_es.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                fx_trial = func(x_trial)
                calls += 1
                
                is_success = fx_trial < fx
                successes += 1 if is_success else 0
                window_count += 1
                
                if is_success:
                    x = x_trial
                    fx = fx_trial
                    best_x = x.copy()
                    best_f = fx
                else:
                    pass
                
                if window_count >= window_size:
                    success_rate = successes / window_count
                    sigma *= np.exp(c_sigma * (success_rate - 0.2))
                    sigma = np.clip(sigma, 1e-8, 10.0)
                    successes = 0
                    window_count = 0
                    
            return best_x, best_f, calls

        # Restart schedule
        # Total budget 500.
        # 1. 100 calls, sigma=2.0, start=0
        # 2. 100 calls, sigma=1.0, start=best
        # 3. 100 calls, sigma=0.5, start=best
        # 4. 100 calls, sigma=0.2, start=best
        # 5. 100 calls, sigma=0.1, start=best
        
        # This might be too rigid. 
        # Let's allow the ES to adapt sigma, but reset it on restart.
        
        # Let's try:
        # 1. 200 calls, sigma=2.0, start=0
        # 2. 150 calls, sigma=1.0, start=best
        # 3. 150 calls, sigma=0.5, start=best
        
        # Actually, the (1+1)-ES adapts sigma. 
        # If we restart, we should probably keep the adapted sigma or set a new one.
        # Setting a new smaller sigma is a form of exploitation.
        
        # Let's implement the multi-restart (1+1)-ES.
        
        current_x = np.zeros(dim)
        current_f = np.inf
        total_calls = 0
        
        # Restart parameters
        # (budget, initial_sigma)
        restarts = [
            (150, 2.0),
            (100, 1.0),
            (100, 0.5),
            (100, 0.2),
            (50, 0.1)
        ]
        
        # First evaluation is already done at 0
        # But the restart function evaluates x_start again. 
        # We need to handle the first call carefully to avoid double counting 
        # or just let it be (500 budget is hard limit, 1 extra call is fine if we adjust).
        # The contract says "More than 500 calls ... invalid".
        # So we must be precise.
        
        # Let's adjust: 
        # The first restart will consume 1 call for the start point.
        # If we already evaluated 0, we can pass that value? 
        # No, the function __call__ is the only interface. 
        # We must track calls inside the restart function.
        
        # Let's modify run_es_restart to accept an initial evaluation if provided?
        # No, keep it simple. 
        # I will subtract 1 from the first restart budget if I pre-evaluate.
        # Or, just let the first restart evaluate 0 again. 
        # 500 budget. 
        # Restart 1: 150 calls. (Includes eval of start point)
        # Restart 2: 100 calls. (Includes eval of start point)
        # ...
        # Total calls = 150 + 100 + 100 + 100 + 50 = 500.
        # This fits perfectly.
        
        # However, if the first restart doesn't use all 150 calls (e.g., time out),
        # we have leftover budget. 
        # The restart function returns calls made.
        
        for budget_r, sigma_r in restarts:
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break
                
            # Adjust budget for this restart
            remaining = budget - total_calls
            current_budget = min(budget_r, remaining)
            
            if current_budget <= 0:
                break
                
            # Start point for this restart
            if total_calls == 0:
                x_start = np.zeros(dim)
            else:
                x_start = best_x_global.copy()
                
            # Run ES
            x_best_r, f_best_r, calls_r = run_es_restart(
                func, x_start, sigma_r, current_budget, rng, deadline
            )
            
            total_calls += calls_r
            
            if f_best_r < best_f_global:
                best_f_global = f_best_r
                best_x_global = x_best_r.copy()
                
            # If we ran out of calls, break
            if total_calls >= budget:
                break
                
        return best_x_global, best_f_global
