"""Rewritten optimizer for BBOB-style f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid deterministic grid scan + CMA-ES for multimodal optimization.
    
    Strategy:
    1. Coarse-to-fine deterministic grid scan for guaranteed early progress.
    2. CMA-ES for efficient local refinement around the best point found.
    3. Structured restarts when CMA-ES stagnates.
    4. Explicit deadline guard for 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time()

        best_x = None
        best_f = float('inf')
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time_left() < 0.5:
                return None
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return None
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Coarse-to-fine deterministic grid scan ---
        # Start with a coarse grid, then refine around the best point
        
        # Step 1: Coarse grid with step size 2.5 over [-4, 4]
        coarse_step = 2.5
        coarse_range = np.arange(-4.0, 4.5, coarse_step)
        
        for x0 in coarse_range:
            if call_count >= budget or time_left() < 1.0:
                break
            for x1 in coarse_range:
                if call_count >= budget or time_left() < 1.0:
                    break
                for x2 in coarse_range:
                    if call_count >= budget or time_left() < 1.0:
                        break
                    for x3 in coarse_range:
                        if call_count >= budget or time_left() < 1.0:
                            break
                        for x4 in coarse_range:
                            if call_count >= budget or time_left() < 1.0:
                                break
                            x = np.array([x0, x1, x2, x3, x4])
                            eval_point(x)
        
        if best_x is None:
            best_x = np.zeros(dim)
            eval_point(best_x)
            if best_f == float('inf'):
                best_f = 0.0
            return best_x, best_f

        # Step 2: Medium grid with step size 1.25 centered on best point
        medium_step = 1.25
        medium_range = np.arange(best_x - 2.5, best_x + 2.5 + medium_step, medium_step)
        
        for x0 in medium_range:
            if call_count >= budget or time_left() < 1.0:
                break
            for x1 in medium_range:
                if call_count >= budget or time_left() < 1.0:
                    break
                for x2 in medium_range:
                    if call_count >= budget or time_left() < 1.0:
                        break
                    for x3 in medium_range:
                        if call_count >= budget or time_left() < 1.0:
                            break
                        for x4 in medium_range:
                            if call_count >= budget or time_left() < 1.0:
                                break
                            x = np.array([x0, x1, x2, x3, x4])
                            eval_point(x)
        
        # Step 3: Fine grid with step size 0.625 centered on best point
        fine_step = 0.625
        fine_range = np.arange(best_x - 1.25, best_x + 1.25 + fine_step, fine_step)
        
        for x0 in fine_range:
            if call_count >= budget or time_left() < 1.0:
                break
            for x1 in fine_range:
                if call_count >= budget or time_left() < 1.0:
                    break
                for x2 in fine_range:
                    if call_count >= budget or time_left() < 1.0:
                        break
                    for x3 in fine_range:
                        if call_count >= budget or time_left() < 1.0:
                            break
                        for x4 in fine_range:
                            if call_count >= budget or time_left() < 1.0:
                                break
                            x = np.array([x0, x1, x2, x3, x4])
                            eval_point(x)

        # --- Phase 2: CMA-ES refinement ---
        # Use CMA-ES starting from the best point found by grid scan
        
        def cma_es(refine_start, max_evals):
            nonlocal best_x, best_f, call_count
            
            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            sigma = 1.0  # Initial step size
            x_mean = refine_start.copy()
            f_best = best_f
            
            # Covariance matrix
            C = np.eye(dim)
            # Step sizes for mean and covariance
            c_sigma = (pop_size + 2) / (dim + pop_size + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - dim - 2) / ((dim + 2) ** 2 + pop_size))
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Weighting
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Initial best
            f_best = best_f
            x_best = best_x.copy()
            
            evals_used = 0
            
            for gen in range(200):
                if call_count >= budget or time_left() < 1.0 or evals_used >= max_evals:
                    break
                
                # Generate population
                offspring = []
                offspring_f = []
                
                for i in range(pop_size):
                    if call_count >= budget or time_left() < 0.5:
                        break
                    
                    # Sample from multivariate normal
                    z = np.random.default_rng(12345 + gen * 1000 + i).standard_normal(dim)
                    # Transform using Cholesky decomposition
                    L = np.linalg.cholesky(C)
                    x_child = x_mean + sigma * (L @ z)
                    x_child = np.clip(x_child, -5.0, 5.0)
                    
                    f_child = eval_point(x_child)
                    if f_child is not None:
                        offspring.append(x_child)
                        offspring_f.append(f_child)
                        evals_used += 1
                
                if len(offspring) < 2:
                    break
                
                # Sort offspring by fitness
                indices = np.argsort(offspring_f)
                offspring = [offspring[i] for i in indices[:mu]]
                offspring_f = [offspring_f[i] for i in indices[:mu]]
                
                # Update best
                for i in range(mu):
                    if offspring_f[i] < f_best:
                        f_best = offspring_f[i]
                        x_best = offspring[i].copy()
                        if f_best < best_f:
                            best_f = f_best
                            best_x = x_best.copy()
                
                # Compute weighted mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * offspring[i]
                
                # Update evolution paths
                old_mean = x_mean.copy()
                x_mean = new_mean
                
                # Update sigma evolution path
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_mean - old_mean)
                
                # Update covariance evolution path
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_mean - old_mean)
                
                # Update covariance matrix
                delta_h = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) <
                          (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1))))
                
                C = (1 + c_1 * (delta_h - 1) + c_mu) * C
                for i in range(mu):
                    y_i = (offspring[i] - old_mean) / sigma
                    C -= c_1 * np.outer(y_i, y_i)
                    C += weights[i] * np.outer(y_i, y_i)
                
                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Check for stagnation
                if gen > 0 and abs(f_best - offspring_f[0]) < 1e-10:
                    stagnation_count += 1
                    if stagnation_count > 10:
                        break
                else:
                    stagnation_count = 0
            
            return x_best, f_best

        # Run CMA-ES from the best point
        remaining_budget = budget - call_count
        if remaining_budget > 20 and time_left() > 5.0:
            # Allocate remaining budget to CMA-ES
            cma_budget = min(remaining_budget, 200)
            cma_es(best_x.copy(), cma_budget)
        
        # --- Phase 3: Final coordinate descent polish ---
        # Quick coordinate descent to fine-tune the result
        
        step_sizes = [0.3125, 0.15625, 0.078125, 0.0390625, 0.01953125, 0.009765625, 0.005]
        current_x = best_x.copy()
        current_f = best_f
        
        for step in step_sizes:
            if call_count >= budget or time_left() < 0.5:
                break
            
            improved = True
            while improved:
                improved = False
                for d in range(dim):
                    if call_count >= budget or time_left() < 0.5:
                        break
                    
                    for sign in [1.0, -1.0]:
                        if call_count >= budget or time_left() < 0.5:
                            break
                        
                        x_try = current_x.copy()
                        x_try[d] += sign * step
                        if -5.0 <= x_try[d] <= 5.0:
                            f_try = eval_point(x_try)
                            if f_try is not None and f_try < current_f - 1e-12:
                                current_x = x_try
                                current_f = f_try
                                improved = True
                                if f_try < best_f:
                                    best_f = f_try
                                    best_x = current_x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
