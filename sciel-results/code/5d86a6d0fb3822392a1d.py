"""
Harmonic-aware optimizer for BBOB f3 (Rastrigin) with CMA-ES refinement.
Exploits the periodic structure of Rastrigin to identify promising integer
lattice points, then refines with CMA-ES and coordinate descent.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        def time_check():
            return (time.time() - start_time) > deadline or calls_made >= budget
        
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Harmonic / lattice exploration
        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global min at x = (k_1, ..., k_n) for integers k_i
        # The function has period 2 in each dimension for the cos term
        # So we evaluate at integer and half-integer points within [-4, 4]
        
        # Generate candidate lattice points
        # For each dimension, candidates: integers and half-integers in [-4, 4]
        # That's 9 values per dim: -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4
        # But 17^5 = 1419857 which is too many. Let's be smarter.
        
        # Strategy: First evaluate at all-integer points (9^5 = 59049, still too many)
        # Better: Use a coarse grid first, then refine
        
        # Coarse grid: integers from -4 to 4 (9 points per dim)
        # 9^5 = 59049 - too many for 500 budget
        # Let's do a smarter sampling:
        
        # Step 1: Evaluate at origin and axis-aligned integer points
        # Step 2: Use a random subset of lattice points
        # Step 3: Use CMA-ES centered on the best found point
        
        # Actually, let's use a two-stage approach:
        # Stage 1a: Evaluate at 0 and +/-1, +/-2, +/-3, +/-4 along each axis (33 points)
        # Stage 1b: Random sampling from the lattice
        # Stage 2: CMA-ES refinement
        
        # Stage 1a: Axis-aligned exploration
        for i in range(n):
            for val in range(-4, 5):
                if calls_made >= budget or time_check():
                    break
                x = np.zeros(n)
                x[i] = float(val)
                f = func(x)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
        
        # Also evaluate at all-zeros
        if calls_made < budget:
            x = np.zeros(n)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Stage 1b: Random lattice sampling
        # Sample random integer points in [-4, 4]^n
        num_random = min(100, budget - calls_made)
        for _ in range(num_random):
            if calls_made >= budget or time_check():
                break
            x = np.random.randint(-4, 5, size=n).astype(float)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Stage 1c: Evaluate near the best integer point with small perturbations
        if best_x is not None and calls_made < budget:
            for delta in [0.1, 0.05, 0.01]:
                if calls_made >= budget or time_check():
                    break
                for i in range(n):
                    for sign in [1, -1]:
                        if calls_made >= budget or time_check():
                            break
                        x = best_x.copy()
                        x[i] += sign * delta
                        if np.all(x >= -5.0) and np.all(x <= 5.0):
                            f = func(x)
                            calls_made += 1
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
        
        # Stage 2: CMA-ES refinement
        if best_x is not None and calls_made < budget:
            # CMA-ES Parameters
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # State variables
            x_mean = best_x.copy()
            sigma = 0.5  # Smaller initial sigma since we're near a good point
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            generation = 0
            while not time_check():
                generation += 1
                ys = []
                for k in range(lam):
                    z = np.random.standard_normal(n)
                    y = B @ (D * z)
                    ys.append(y)
                
                xs = x_mean + sigma * np.array(ys)
                
                remaining = budget - calls_made
                num_evals = min(lam, remaining)
                if num_evals <= 0:
                    break
                
                fitnesses = []
                eval_xs = []
                for i in range(num_evals):
                    x = xs[i]
                    f = func(x)
                    calls_made += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    fitnesses.append(f)
                    eval_xs.append(x)
                    if calls_made >= budget:
                        break
                
                if calls_made >= budget or num_evals < lam:
                    break
                    
                fitnesses = np.array(fitnesses)
                eval_xs = np.array(eval_xs)
                
                order = np.argsort(fitnesses)
                xs_sorted = eval_xs[order]
                
                new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                y_mean = (new_mean - x_mean) / sigma if sigma > 0 else np.zeros(n)
                
                c_hs = (1.1 + 2.0 / (n + 1.7))
                chi_n = np.sqrt(n) * (1 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))
                
                norm_y_mean = np.linalg.norm(y_mean)
                sqrt_trace_C = np.sqrt(np.trace(C))
                
                ratio = norm_y_mean / (chi_n * sqrt_trace_C) if sqrt_trace_C > 0 else 0
                
                p_sigma = (1 - c_sigma) * p_sigma + c_hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ratio * y_mean
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * generation)) < (1.4 + 2 / (n + 1)) * chi_n)
                
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                
                ys_array = np.array(ys)
                ys_sorted = ys_array[order][:mu]
                
                rank_one = np.outer(p_c, p_c)
                
                y_diffs = ys_sorted - y_mean
                rank_mu = np.sum(weights[:, None, None] * y_diffs[:, :, None] * y_diffs[:, None, :], axis=0)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * rank_mu
                C = (C + C.T) / 2
                
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                
                x_mean = new_mean
                
                if sigma < 1e-12:
                    break
        
        # Stage 3: Coordinate descent polish
        if best_x is not None and calls_made < budget:
            step = 0.05
            for _ in range(30):
                if calls_made >= budget or time_check():
                    break
                improved = False
                for i in range(n):
                    for direction in [1, -1]:
                        if calls_made >= budget or time_check():
                            break
                        x_new = best_x.copy()
                        x_new[i] += direction * step
                        if np.all(x_new >= -5.0) and np.all(x_new <= 5.0):
                            f_new = func(x_new)
                            calls_made += 1
                            if f_new < best_f:
                                best_f = f_new
                                best_x = x_new.copy()
                                improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-7:
                        break
        
        return best_x, best_f

# EVOLVE-BLOCK-END
