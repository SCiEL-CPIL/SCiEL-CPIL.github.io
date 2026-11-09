import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(42)

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial random sampling
        n_init = 30
        for _ in range(n_init):
            if time.time() > deadline: break
            x = rng.uniform(-5.0, 5.0, n)
            try_func(x)
        
        if best_f == np.inf:
            return best_x, best_f

        # Phase 2: Estimate Ridge Direction
        # For f13 (Sharp Ridge), the function is flat along the ridge and steep orthogonal to it.
        # We search for the direction that causes the smallest increase in f for a fixed step.
        # This identifies the "valley" axis.
        
        n_dirs = 60
        step_size = 1.0
        best_dir_score = np.inf
        best_ridge_dir = None
        
        # We evaluate f(x + step*d) and f(x - step*d)
        # If the direction is orthogonal to the ridge, f increases significantly.
        # If the direction is along the ridge, f increases minimally (or stays flat).
        
        for _ in range(n_dirs):
            if time.time() > deadline: break
            d = rng.standard_normal(n)
            norm_d = np.linalg.norm(d)
            if norm_d == 0: continue
            d = d / norm_d
            
            x_plus = best_x + step_size * d
            x_minus = best_x - step_size * d
            
            f_plus = try_func(x_plus)
            f_minus = try_func(x_minus)
            
            # Score: how much did the function value increase?
            # We use the minimum of the two increases to be robust to asymmetric noise/shifts
            inc_plus = f_plus - best_f
            inc_minus = f_minus - best_f
            
            # If both are negative, we moved to a better point, which is great, but doesn't necessarily mean it's the ridge direction.
            # However, for a valley, moving along the ridge usually keeps f low.
            # Moving orthogonal moves f high.
            # So we want to minimize the *worst* case increase? Or the average?
            # Let's use the minimum increase. If we can move in a direction without increasing f much, it's likely the ridge.
            score = min(inc_plus, inc_minus)
            
            if score < best_dir_score:
                best_dir_score = score
                best_ridge_dir = d.copy()
                
        if best_ridge_dir is None:
            # Fallback if no direction found (shouldn't happen)
            best_ridge_dir = rng.standard_normal(n)
            best_ridge_dir /= np.linalg.norm(best_ridge_dir)

        # Phase 3: 1D Optimization along the Ridge
        # Use Golden Section Search to find the best point along best_ridge_dir
        
        def f_line(t):
            x = best_x + t * best_ridge_dir
            x = np.clip(x, -5.0, 5.0)
            return try_func(x)
        
        # Define a search interval for t
        # Start with a wide interval and narrow it down
        # We know best_x is good. Let's search in [-10, 10]
        a = -10.0
        b = 10.0
        
        # Golden ratio
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = f_line(c)
        fd = f_line(d)
        
        # Iterate
        max_iter_1d = 30
        for _ in range(max_iter_1d):
            if time.time() > deadline: break
            if calls >= budget - 50: break # Reserve budget for CMA-ES
            
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f_line(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f_line(d)
                
        # Final best in 1D
        t_best = (a + b) / 2
        x_1d_best = best_x + t_best * best_ridge_dir
        x_1d_best = np.clip(x_1d_best, -5.0, 5.0)
        f_1d_best = try_func(x_1d_best)
        
        if f_1d_best < best_f:
            best_x = x_1d_best
            best_f = f_1d_best

        # Phase 4: CMA-ES Polish
        # Use CMA-ES to refine the position, correcting for any errors in the ridge direction
        # and finding the exact bottom of the valley.
        
        if time.time() < deadline and calls < budget - 10:
            # Simple CMA-ES implementation
            lambda_ = 8
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            mean = best_x.copy()
            sigma = 0.5 # Start with a small sigma since we are already close
            
            B = np.eye(n)
            D = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            
            # Run CMA-ES until budget or time is exhausted
            while calls < budget - 5 and time.time() < deadline:
                # Generate offspring
                offspring = []
                for i in range(lambda_):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    x = mean + sigma * (B @ (D @ z))
                    x = np.clip(x, -5.0, 5.0)
                    f = try_func(x)
                    offspring.append((f, x))
                
                if len(offspring) < mu:
                    break
                
                # Sort offspring
                offspring.sort(key=lambda x: x[0])
                
                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * offspring[i][1]
                    
                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (new_mean - mean) / sigma
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (new_mean - mean) / sigma
                
                # Update covariance matrix
                # H = ps * ps.T + (||ps||^2 / chi_n^2 - 1) * pc * pc.T
                # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (||pc||^2 / n - 1) * I) + cmu * sum(weights * (xi - mean)(xi - mean).T)
                
                # For simplicity and speed in 5D, we can update B and D using eigen-decomposition of C
                # But full CMA-ES covariance update is heavy. 
                # Given n=5, it's manageable.
                
                C = np.eye(n) # We didn't track C explicitly in the simplified loop above? 
                # Wait, the previous code snippet in the prompt didn't track C correctly for a full CMA-ES.
                # Let's implement a proper CMA-ES step for C update.
                
                # Actually, let's just use scipy's minimize with method='Nelder-Mead' or 'Powell' 
                # if CMA-ES implementation is too risky/long. 
                # However, CMA-ES is better for this.
                # Let's stick to a simplified CMA-ES that tracks C.
                
                # Re-initialize C tracking for this phase? No, we need to carry it over.
                # Since we didn't initialize C in the previous block, let's do a local CMA-ES here.
                
                # Let's just use Nelder-Mead with a large maxiter and tight tolerances as a fallback if CMA-ES is complex.
                # But Nelder-Mead can fail on ridges.
                # Powell's method is better for ridges.
                
                # Let's switch to Powell's Method for the final polish.
                # Powell's method is a derivative-free method that finds conjugate directions, which aligns well with the ridge.
                
                break # Break out of manual CMA-ES loop to use Powell

            # Use Powell's Method for final polish
            if time.time() < deadline and calls < budget - 5:
                try:
                    res = minimize(func, best_x, method='Powell', 
                                   bounds=[(-5, 5)] * n,
                                   options={'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-12})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0)
                except Exception:
                    pass

        return best_x, best_f
