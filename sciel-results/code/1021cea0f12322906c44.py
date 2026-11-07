"""Hybrid Sobol + CMA-ES + Nelder-Mead optimizer for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid optimizer: Sobol initialization, CMA-ES global search, Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = None
        best_f = float('inf')
        all_x = []
        all_f = []

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f, all_x, all_f
            if eval_count >= budget:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            eval_count += 1
            f_val = float(func(x))
            all_x.append(x.copy())
            all_f.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Sobol Initialization ---
        # Use a simple Sobol-like sequence or Halton for space-filling
        # Since scipy might not have a direct Sobol generator in all versions easily accessible 
        # without setup, we use a stratified random approach or a simple low-discrepancy sequence.
        # Let's use a simple Halton sequence for the first few points.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(count, dim):
            points = []
            for i in range(1, count + 1):
                point = []
                for d in range(dim):
                    # Find the (d+1)-th prime
                    base = 2
                    while True:
                        is_prime = True
                        for p in range(2, int(np.sqrt(base)) + 1):
                            if base % p == 0:
                                is_prime = False
                                break
                        if is_prime:
                            break
                        base += 1
                        if d == 0: # Hack: just use small primes 2,3,5,7,11
                            if base == 2 and d==0: break
                            if base == 3 and d==1: break
                            if base == 5 and d==2: break
                            if base == 7 and d==3: break
                            if base == 11 and d==4: break
                            if base > 11: break # Fallback
                    # Simplified: use fixed primes for dim=5
                    primes = [2, 3, 5, 7, 11]
                    b = primes[d]
                    f = 1.0
                    r = 0.0
                    idx = i
                    while idx > 0:
                        f /= b
                        r += f * (idx % b)
                        idx //= b
                    point.append(r)
                # Map [0,1] to [-5,5]
                point = np.array(point) * 10.0 - 5.0
                points.append(point)
            return points

        n_init = 30
        init_points = halton_sequence(n_init, dim)
        # Add center
        init_points.append(np.zeros(dim))
        
        for pt in init_points:
            if eval_count >= budget - 50 or time.time() > deadline:
                break
            wrapped_func(pt)

        # --- Phase 2: CMA-ES ---
        # Initialize CMA-ES with best point from initialization
        if best_x is None:
            best_x = np.zeros(dim)
        
        # CMA-ES parameters
        pop_size = 10
        sigma = 2.0
        x_mean = best_x.copy()
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim) # Approximation for initialization
        step_size = sigma
        
        # Pre-compute constants
        mu = pop_size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial Cholesky factor
        L = np.linalg.cholesky(C)

        cma_evals = 0
        max_cma_evals = int(budget * 0.6)
        stagnation_counter = 0
        prev_best_f = best_f

        while eval_count < budget - 50 and cma_evals < max_cma_evals and time.time() < deadline:
            # Generate population
            xs = []
            for i in range(pop_size):
                z = np.random.normal(0, 1, dim)
                x = x_mean + step_size * (L @ z)
                x = np.clip(x, -5.0, 5.0)
                xs.append(x)
            
            # Evaluate
            fs = []
            for x in xs:
                f = wrapped_func(x)
                fs.append(f)
                cma_evals += 1
                if eval_count >= budget - 50:
                    break
            
            if len(fs) < pop_size:
                break

            # Sort
            order = np.argsort(fs)
            xs_sorted = [xs[i] for i in order]
            fs_sorted = [fs[i] for i in order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += w[i] * xs_sorted[i]
            
            # Update step size (sigma)
            # Compute C^{-1/2} (approx using current C)
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (C_inv_sqrt @ (x_new_mean - x_mean)) / step_size
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (cma_evals // pop_size + 1))) < (1 + 2 / (dim + 1)) * chi_n)
            
            # Update C
            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_new_mean - x_mean) / step_size
            
            delta_h = 1 if h_sigma else 0
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(w[i] * np.outer(xs_sorted[i] - x_mean, xs_sorted[i] - x_mean) for i in range(mu))
            
            # Regularization
            C += 1e-10 * np.eye(dim)
            
            # Update step size
            step_size *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Ensure step size doesn't get too small or too large
            step_size = np.clip(step_size, 1e-8, 10.0)
            
            x_mean = x_new_mean
            
            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = 0.5 * C + 0.5 * np.eye(dim)
                L = np.linalg.cholesky(C)

            # If stagnated for too long, restart from a random point or best point with larger sigma
            if stagnation_counter > 20:
                # Restart
                x_mean = best_x.copy()
                step_size = 1.0
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation_counter = 0
                # Optional: evaluate a few random points to escape
                for _ in range(5):
                    if eval_count >= budget - 50: break
                    pt = best_x + np.random.normal(0, 2.0, dim)
                    pt = np.clip(pt, -5.0, 5.0)
                    wrapped_func(pt)

        # --- Phase 3: Nelder-Mead Refinement ---
        # Use remaining budget to polish the best point
        if best_x is not None and eval_count < budget and time.time() < deadline:
            # Simple Nelder-Mead implementation
            simplex = [best_x.copy()]
            f_vals = [best_f]
            
            # Build initial simplex
            for i in range(dim):
                x_new = best_x.copy()
                x_new[i] += 0.1
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = wrapped_func(x_new)
                simplex.append(x_new)
                f_vals.append(f_new)
                
                if eval_count >= budget:
                    break

            # Nelder-Mead steps
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            max_nm_iter = budget - eval_count
            for _ in range(max_nm_iter):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Update global best
                if f_vals[0] < best_f:
                    best_f = f_vals[0]
                    best_x = simplex[0].copy()
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = wrapped_func(x_ref)
                
                if f_ref < f_vals[-2]:
                    if f_ref < f_vals[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5.0, 5.0)
                        f_exp = wrapped_func(x_exp)
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            f_vals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            f_vals[-1] = f_ref
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_vals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = wrapped_func(x_con)
                        if f_con <= f_ref:
                            simplex[-1] = x_con
                            f_vals[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_vals[i] = wrapped_func(simplex[i])
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = wrapped_func(x_con)
                        if f_con < f_vals[-1]:
                            simplex[-1] = x_con
                            f_vals[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_vals[i] = wrapped_func(simplex[i])
                            
        return best_x, best_f
# EVOLVE-BLOCK-END
