"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        # CMA-ES parameters
        # Standard CMA-ES uses population size lambda = 4 + floor(3 log n)
        # For n=5, lambda = 4 + 3*log(5) ~ 4 + 4.8 -> 8.
        # However, for a multimodal function like Lunacek with limited budget,
        # a larger population helps exploration.
        # Let's use lambda = 12 or 16.
        # With 500 calls, and lambda=12, we get ~40 generations.
        # This is reasonable.
        
        # To improve early AOCC, we want to find a decent minimum fast.
        # Starting from the center [0,0,0,0,0] is often a good start for Lunacek 
        # because the global minimum is usually at a shifted corner, but the 
        # "shallow" minimum is often near the origin or the shift is not too extreme 
        # in the first few dimensions. However, the problem statement says 
        # "hidden seeded shift... minimizer lies in [-4, 4]^5".
        # Actually, Lunacek's global min is typically at a point like [4,4,4,4,4] 
        # relative to the center of the box, or shifted.
        # The origin [0,0,0,0,0] is a local minimum (the shallow one) in the standard 
        # Lunacek function. The global minimum is the "deep" one.
        # If we start at 0, we might get stuck in the shallow well.
        # If we start at a random point or a corner, we might find the deep well.
        
        # Strategy: 
        # 1. Start CMA-ES from the origin? No, that's the local min.
        # 2. Start from a point that is likely to be in the basin of the global min.
        #    In standard Lunacek, the global min is at x* = x0 + shift.
        #    The shift is hidden. 
        #    However, the function is symmetric in a way? No, it's bi-modal.
        #    One mode is at 0 (shallow), one is at x* (deep).
        #    The distance between them is significant.
        
        # Let's use a multi-start CMA-ES or a single CMA-ES with a large initial sigma.
        # If we start at 0 with large sigma, we might jump out.
        # Or we can start at a deterministic point like [4,4,4,4,4] which is 
        # often close to the global min in these benchmarks.
        
        # Let's try starting at [4,4,4,4,4] with a medium sigma.
        # If that fails, we can restart. But CMA-ES is expensive to restart.
        
        # Alternative: Use the "success rule" CMA-ES which adapts well.
        
        # Let's pick initial mean m = np.zeros(dim) or np.ones(dim)*4.0?
        # Given the score of the previous program (2.08) used [4,4,4,4,4] first,
        # it suggests that area is good.
        
        m = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
        sigma = 1.0
        
        # Population size
        # Let's use lambda = 10
        lam = 10
        mu = lam // 2  # number of selected individuals
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Step size control
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        rho_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        
        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for square root
        # We need C^{-1/2} and C^{1/2}
        
        # Helper to update C and get square roots
        def update_covariance(C, weights, y, z, c_1, c_mu, dim, mu):
            # y: (mu, dim) matrix of selected steps (transformed to search space)
            # z: (mu, dim) matrix of selected steps (in original space? No, CMA-ES uses y in search space)
            # Actually, standard CMA-ES updates C using the weighted outer products of the selected steps y_i
            # y_i = (x_i - m_old) / sigma
            
            C_new = C + c_1 * (z @ z.T - C) # Wait, z is usually the path history for CS? No.
            # The update is:
            # C <- (1 - c_1 - c_mu) C + c_1 (z z^T + (1 - h_sigma) C) + c_mu sum w_i y_i y_i^T
            # where z is the evolution path for covariance.
            
            # For simplicity and robustness, let's use a simplified rank-mu update if possible,
            # or just stick to the standard formulas.
            
            # Let's implement the standard CMA-ES update properly.
            
            # We need the evolution path for covariance p_c
            # p_c = (1 - c_c) p_c + h_c * sqrt(c_c (2 - c_c) mu_eff) * C^{-1/2} * (sum w_i y_i)
            
            # This requires C^{-1/2}.
            # We will maintain eigen decomposition of C.
            pass
            
        # To keep the code simple and robust within the time limit, 
        # I will implement a simplified CMA-ES that uses PCA of the population 
        # to adapt the covariance, which is less stable but easier to code correctly 
        # from scratch without a full library.
        
        # Actually, let's use a simpler approach: 
        # **Adaptive Coordinate Descent with Random Projections**
        # Or **Nelder-Mead** with restarts? Nelder-Mead is often bad for high dim.
        
        # Let's go with a robust **Randomized Coordinate Descent** with adaptive step sizes,
        # but using **random directions** instead of just axes, to handle the rotation.
        
        # Strategy:
        # 1. Start at [4,4,4,4,4].
        # 2. Maintain a set of random directions (Gaussian).
        # 3. Perform line search along these directions.
        # 4. If stuck, restart from a different point.
        
        # This is essentially a form of stochastic local search.
        
        # Let's try **Powell's Method** (conjugate direction set) adapted for black-box.
        # Powell's method is very effective for smooth functions and handles non-orthogonal coordinates.
        # It is deterministic and often outperforms coordinate descent on rotated landscapes.
        
        def powell_method(func, x0, budget, calls_ref, deadline):
            nonlocal calls_made, best_f, best_x
            
            x = x0.copy()
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            if calls_made >= budget or time.time() > deadline:
                return x, fx
            
            # Initial directions: unit vectors
            directions = [np.eye(dim, dtype=float)[i] for i in range(dim)]
            
            # Powell's method loop
            # Each cycle:
            # 1. For each direction, do a line search (1D optimization) along that direction.
            # 2. Compute the new direction as the difference between the start and end of the cycle.
            # 3. Replace the direction that gave the worst improvement with the new direction.
            
            # 1D optimization: Golden section or simple ternary search?
            # Since we have a limited budget, we can do a simple parabolic interpolation or 
            # just a few steps of golden section.
            
            def line_search(x_start, direction, fx_start, max_iter=5):
                # Simple golden section search within a reasonable bound
                # We need to find the step size alpha that minimizes f(x_start + alpha * direction)
                
                # Determine bounds for alpha
                # Start with alpha = 1.0
                # Expand until we find a valley, or shrink if we are already in one?
                # For robustness, let's just try a few discrete alpha values?
                # No, that's inefficient.
                
                # Let's use a simple "try +/-" approach with adaptive step size,
                # similar to coordinate descent but along arbitrary directions.
                
                alpha = 1.0
                x_curr = x_start
                f_curr = fx_start
                x_best = x_start
                f_best = fx_start
                
                for _ in range(max_iter):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Try positive step
                    x_trial = x_curr + alpha * direction
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_made += 1
                    if f_trial < f_best:
                        f_best = f_trial
                        x_best = x_trial
                        # Move in this direction
                        x_curr = x_trial
                    else:
                        # Try negative step
                        x_trial = x_curr - alpha * direction
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls_made += 1
                        if f_trial < f_best:
                            f_best = f_trial
                            x_best = x_trial
                            # Move in negative direction
                            x_curr = x_trial
                            direction = -direction # Flip direction for next step? 
                            # Actually, if we moved negative, the "best" direction is negative.
                            # But we are returning the point. The direction update in Powell 
                            # relies on the net displacement.
                        
                    # Shrink step size if we are converging?
                    # For now, keep alpha constant or shrink slightly.
                    alpha *= 0.7
                
                return x_best, f_best

            # Main Powell Loop
            # We have ~500 calls.
            # Each line search takes ~5-10 calls.
            # 5 directions * 10 calls = 50 calls per cycle.
            # ~10 cycles possible.
            
            for cycle in range(20): # Max cycles
                if calls_made >= budget or time.time() > deadline:
                    break
                
                x_cycle_start = x.copy()
                fx_cycle_start = fx
                max_improvement = 0
                worst_dir_idx = 0
                worst_improvement = np.inf
                
                for i, dir_vec in enumerate(directions):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    x_new, fx_new = line_search(x, dir_vec, fx, max_iter=4)
                    improvement = fx - fx_new
                    if improvement > max_improvement:
                        max_improvement = improvement
                    if improvement < worst_improvement:
                        worst_improvement = improvement
                        worst_dir_idx = i
                    
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Compute new direction
                x_new_dir = x - x_cycle_start
                norm_new_dir = np.linalg.norm(x_new_dir)
                if norm_new_dir > 1e-8:
                    new_dir = x_new_dir / norm_new_dir
                else:
                    new_dir = np.zeros(dim)
                
                # Replace the worst direction with the new one
                if max_improvement > 0:
                    directions[worst_dir_idx] = new_dir
                else:
                    # If no improvement, we might be stuck. 
                    # Break or restart?
                    # Let's break and rely on external restarts.
                    break
            
            return x, fx

        # Multi-start Powell
        # Start points:
        # 1. [4,4,4,4,4]
        # 2. [-4,-4,-4,-4,-4]
        # 3. [0,0,0,0,0]
        # 4. Random point
        
        starts = [
            np.array([4.0, 4.0, 4.0, 4.0, 4.0]),
            np.array([-4.0, -4.0, -4.0, -4.0, -4.0]),
            np.zeros(dim),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0]),
        ]
        
        for start in starts:
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Allocate remaining budget to this start
            # If we have < 50 calls, skip
            if budget - calls_made < 50:
                break
                
            powell_method(func, start, budget, calls_made, deadline)
            
            # Check if we found a very good solution
            # If best_f is close to expected global min (e.g. < -50), we might stop?
            # But we don't know f_opt.
            # Let's just continue if we have budget.

        # Final polish: Coordinate descent with very small steps
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            scales = [0.1, 0.05, 0.01, 0.005, 0.001]
            for scale in scales:
                if calls_made >= budget or time.time() > deadline:
                    break
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    # Try +
                    x_trial = x.copy()
                    x_trial[d] += scale
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_made += 1
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    # Try -
                    x_trial = x.copy()
                    x_trial[d] -= scale
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_made += 1
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
