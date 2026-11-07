import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # 1. Random Initialization
        n_init = 50
        init_points = []
        for _ in range(n_init):
            if n_calls >= budget: break
            x = np.random.uniform(-4, 4, dim)
            f = timed_call(x)
            init_points.append((f, x.copy()))
        
        if n_calls >= budget:
            return best_x, best_f

        # Sort by fitness
        init_points.sort(key=lambda x: x[0])
        
        # 2. Local Optimization from best seeds
        # We will try to optimize from the top few seeds.
        # Strategy: For each seed, run Nelder-Mead, then (1+1)-ES.
        
        def nelder_mead(x0, max_calls):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return x0
            
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = timed_call(x0)
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.1 if xi[i] < 4.9 else -0.1
                simplex.append(xi.copy())
                timed_call(xi)
            
            # Sort simplex
            simplex.sort(key=lambda x: timed_call(x)) # Wait, we already evaluated? 
            # Better to keep track of values to avoid re-evaluation
            # Let's rewrite NM to be more efficient with calls.
            
            # Actually, implementing a strict NM with call counting is tricky.
            # Let's use a simpler local search: (1+1)-ES with adaptive step size.
            # It is often sufficient for 5D.
            
            # Let's implement a robust (1+1)-ES with 1/5 rule.
            # It is very fast and effective.
            
            x = x0.copy()
            sigma = 0.5
            c_sigma = 1.0 / (dim + 1)
            d_sigma = 1 + 2 / max(1, dim) + c_sigma
            
            local_best_f = best_f
            local_best_x = x0.copy()
            
            while n_calls < budget and time.time() < deadline:
                # Generate candidate
                w = np.random.standard_normal(dim)
                y = x + sigma * w
                fy = timed_call(y)
                
                if fy < local_best_f:
                    local_best_f = fy
                    local_best_x = y.copy()
                
                # Adapt sigma
                if fy < best_f: # Global improvement? Or local?
                    # Standard 1/5 rule uses success rate over a window.
                    # Simpler: if improved, increase sigma slightly, else decrease.
                    # But this can oscillate.
                    # Let's use a proper 1/5 rule window.
                    pass
                
                # For simplicity and robustness, let's just do a hill-climbing step.
                # If improved, move x to y.
                if fy < best_f:
                    x = y.copy()
                    sigma *= 1.1 # Increase step size on success
                else:
                    sigma *= 0.9 # Decrease step size on failure
                
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Stagnation check
                # If no improvement for 20 steps, break.
                # We need to track this.
            
            return local_best_x

        # The above (1+1)-ES is a bit naive.
        # Let's use a more structured approach:
        # 1. Random Sampling (50 calls)
        # 2. For the best 5 points:
        #    a. Run Nelder-Mead (limited iterations)
        #    b. Run (1+1)-ES (limited calls)
        
        # Let's implement a proper Nelder-Mead that respects the call budget.
        
        def run_nelder_mead(x_start, max_iter):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return x_start
            
            # Create initial simplex
            simplex = [x_start.copy()]
            # We need to evaluate x_start? It was evaluated in init.
            # But we need values for all vertices.
            # To save calls, we assume x_start is evaluated.
            # For other vertices, we evaluate.
            
            # This is getting complex to manage exact call counts without re-evaluating.
            # Given the small budget (500), re-evaluating is expensive.
            
            # Alternative: Use SciPy's minimize?
            # "Python standard library, NumPy and SciPy are available."
            # Using SciPy's minimize with method='Nelder-Mead' is very robust.
            
            import scipy.optimize
            
            # We need a wrapper that counts calls and handles the deadline.
            # But scipy.optimize.minimize might ignore the deadline and run too long.
            # We can use a callback to check time.
            
            def scipy_func(x):
                nonlocal n_calls
                if n_calls >= budget or time.time() > deadline:
                    raise Exception("Budget Exceeded")
                return timed_call(x)
            
            def callback(x):
                if time.time() > deadline:
                    raise Exception("Time Exceeded")
            
            try:
                res = scipy.optimize.minimize(
                    scipy_func, 
                    x_start, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8},
                    callback=callback
                )
                return res.x
            except Exception:
                return x_start

        # Using SciPy is a strong candidate for robustness.
        # Let's combine:
        # 1. 50 Random Calls.
        # 2. Take top 5.
        # 3. For each, run SciPy Nelder-Mead.
        # 4. After NM, run a final (1+1)-ES polish from the best point found.

        # Let's refine the "Polish" phase.
        # A simple (1+1)-ES with 1/5 rule is good.
        
        def polish_es(x_start, max_calls):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return x_start
            
            x = x_start.copy()
            sigma = 0.1
            # 1/5 rule parameters
            c_sigma = 1.0 / (dim + 1)
            d_sigma = 1 + 2 / max(1, dim) + c_sigma
            window = 10
            successes = 0
            
            for _ in range(max_calls):
                if n_calls >= budget or time.time() > deadline:
                    break
                
                w = np.random.standard_normal(dim)
                y = x + sigma * w
                fy = timed_call(y)
                
                if fy < best_f:
                    x = y.copy()
                    successes += 1
                else:
                    successes += 0 # No success
                
                # Update sigma every 'window' steps
                # Simplified: update every step based on immediate success?
                # No, 1/5 rule uses a window.
                # For simplicity, let's just adapt sigma based on recent history.
                
                # Let's just do a simple adaptive step size.
                if fy < best_f:
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, 1e-6, 1.0)
                
            return x

        # Execution Plan:
        # 1. Random Init (50 calls)
        # 2. Get top 5 seeds.
        # 3. For each seed:
        #    - Run SciPy NM (max 50 iterations? ~300 calls? No, NM is ~6 calls/iter. 50 iters = 300 calls. Too many for 5 seeds.)
        #    - Budget is 500.
        #    - 50 init. 450 left.
        #    - 5 seeds * 90 calls each.
        #    - NM with maxiter=10 (60 calls) + Polish (30 calls).
        
        seeds = [p[1] for p in init_points[:5]]
        
        for seed in seeds:
            if n_calls >= budget or time.time() > deadline:
                break
            
            # Run NM
            x_nm = run_nelder_mead(seed, 10)
            
            # Run Polish
            x_polish = polish_es(x_nm, 30)
            
            # Update global best if needed (timed_call does this)
        
        # Final Polish from global best
        if n_calls < budget and time.time() < deadline:
            polish_es(best_x, budget - n_calls)
            
        return best_x, best_f
