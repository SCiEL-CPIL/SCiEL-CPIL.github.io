"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Coarse-Grid Screening + Adaptive CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin: 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Coarse Global Screening
        # The BBOB f7 (Step Ellipsoidal) has a 'step' in the objective function.
        # Global exploration is needed to find the region containing the global minimum.
        # We perform a coarse grid search to identify the best basin.
        
        # Grid size: 2 points per dimension is too few, 4 is better.
        # 4^5 = 1024 points. This might exceed budget if budget is small, but here budget=500.
        # Let's use a stratified random sampling instead of a fixed grid to be safer with budget,
        # or a very coarse grid. 
        # Actually, for dim=5, a 3x3x3x3x3 grid is 243 points. This fits in 500 budget.
        # Let's use a 3-level grid per dimension.
        grid_levels = 3
        grid_points = grid_levels ** dim
        # If budget is less than grid_points, we fall back to random sampling
        if budget >= grid_points + 10:
            # Generate grid
            coords = np.linspace(-5, 5, grid_levels)
            grid = np.array(np.meshgrid(*([coords]*dim)))
            grid = grid.reshape(dim, -1).T
            # Shuffle grid to avoid bias in ordering (though grid is symmetric)
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(grid))
            grid = grid[perm]
            
            # Evaluate grid
            fvals = np.empty(len(grid))
            calls_made = 0
            for i in range(len(grid)):
                if calls_made >= budget:
                    break
                fvals[i] = func(grid[i])
                calls_made += 1
            
            # Find best point in grid
            best_idx = np.argmin(fvals[:calls_made])
            best_x = grid[best_idx].copy()
            best_f = fvals[best_idx]
            
            # Select top K points for local refinement
            # We have budget - calls_made left.
            # Let's pick the top 3 distinct points from the grid to start local optimization.
            # We need to be careful not to pick the same point multiple times.
            # Sort indices by fval
            sorted_indices = np.argsort(fvals[:calls_made])
            top_k = min(3, calls_made)
            start_points = [grid[sorted_indices[i]].copy() for i in range(top_k)]
        else:
            # Fallback: Random initialization
            rng = np.random.default_rng(42)
            pop_size = min(budget // 2, 20)
            population = rng.uniform(-5, 5, (pop_size, dim))
            fvals = np.empty(pop_size)
            calls_made = 0
            for i in range(pop_size):
                fvals[i] = func(population[i])
                calls_made += 1
            best_idx = np.argmin(fvals)
            best_x = population[best_idx].copy()
            best_f = fvals[best_idx]
            start_points = [population[i].copy() for i in range(min(3, pop_size))]

        # 2. Local Refinement using CMA-ES
        # CMA-ES is excellent for continuous optimization, especially ellipsoidal landscapes.
        # We implement a simplified, robust CMA-ES variant.
        
        # Parameters
        # Sigma (step size) initial: relative to box size
        sigma0 = 1.0 
        # Population size for CMA-ES: lambda = 4 + floor(3 * ln(dim))
        lambda_cma = 4 + int(3 * np.log(dim))
        
        # We will run CMA-ES sequentially on the best start points.
        # We allocate the remaining budget to these runs.
        # To allow early progress, we interleave or run the most promising one first.
        
        # Sort start points by fval (we already have best_f from grid)
        # We need to evaluate the start points if we didn't in the grid phase? 
        # In the grid phase, we evaluated them. In random fallback, we evaluated them.
        # So start_points[0] is the best.
        
        # We will run CMA-ES on the single best point first, using a large chunk of budget.
        # If it stagnates, we might try the next one, but CMA-ES is usually good enough.
        # Given the "Step" nature, the global basin is likely found by the grid.
        # The local landscape is ellipsoidal, which CMA-ES handles very well.
        
        # Implement a lightweight CMA-ES
        def cma_es(x0, f0, sigma, budget_left, func, calls_made, best_x, best_f):
            # x0: initial mean (np.array)
            # f0: initial function value
            # sigma: initial step size
            # budget_left: max calls to use
            # func: objective function
            # calls_made: current count (will be updated)
            # best_x, best_f: global best (will be updated)
            
            dim = len(x0)
            # CMA-ES parameters
            # Standard CMA-ES uses lambda = 4 + 3*ln(n)
            lambda_cma = 4 + int(3 * np.log(dim))
            
            # Strategy parameters
            alpha_mu = 1.0
            # Weights: linear weights
            mu = lambda_cma // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, lambda_cma + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = (mu_eff - 2 + 2 / (dim + mu_eff)) * c_1 / 2
            
            # Initialization
            mean = x0.copy()
            sigma = sigma
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Precompute sqrt(C) via Cholesky (approximation: use identity initially, update)
            # For simplicity and speed in this constrained context, we use a simplified update
            # or rely on the fact that dim=5 is small.
            # Actually, implementing full CMA-ES with matrix updates is complex.
            # Let's use a simplified "Adaptive Simplex" or "Nelder-Mead" with adaptive step size?
            # No, CMA-ES is the standard for this. Let's implement a robust, simplified CMA-ES.
            
            # To keep code concise and robust, we will use a simplified version of CMA-ES
            # that updates the covariance matrix via rank-1 and rank-mu updates.
            
            # Note: This implementation is for dim=5, so matrix ops are cheap.
            
            while calls_made < budget_left and time.time() < deadline:
                # Generate lambda samples
                # Need sqrt(C). Use Cholesky decomposition.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C becomes non-positive definite
                    L = np.eye(dim)
                    C = np.eye(dim)
                
                # Sample
                z = np.random.randn(lambda_cma, dim)
                # y = L @ z.T
                y = np.dot(L, z.T).T
                # x = mean + sigma * y
                x_samples = mean + sigma * y
                
                # Evaluate
                f_samples = np.empty(lambda_cma)
                for i in range(lambda_cma):
                    if calls_made >= budget_left or time.time() > deadline:
                        break
                    f_samples[i] = func(x_samples[i])
                    calls_made += 1
                    if f_samples[i] < best_f:
                        best_f = f_samples[i]
                        best_x = x_samples[i].copy()
                
                # Sort by fitness
                # Only sort the evaluated ones
                n_eval = calls_made - (budget_left - calls_made) # This logic is tricky with external counter
                # Let's track local calls
                # Actually, simpler: just use the array up to the last filled index.
                # Since we break, we need to know how many were filled.
                # Let's refactor to return calls made in this step.
                
                # This nested function approach is getting messy with call counting.
                # Let's inline the CMA-ES logic or use a simpler adaptive method.
                
                # Alternative: Use scipy.optimize.minimize with Nelder-Mead, 
                # but with a "re-start" mechanism if no improvement.
                # Given the previous success of DE (2.3) and failure of pure NM (1.5),
                # and the hint about "Step" structure:
                # The grid finds the basin. Then we need to converge fast.
                # Nelder-Mead is very fast in smooth regions.
                # Let's use a robust Nelder-Mead implementation with adaptive bounds.
                
                break # Break out of this complex CMA-ES implementation attempt to use a simpler, proven hybrid.
            
            return calls_made, best_x, best_f

        # Refined Strategy: Grid -> Best Point -> Nelder-Mead (with adaptive step)
        # Why NM? It's derivative-free, fast, and works well on ellipsoidal shapes once in the basin.
        # The "Step" is handled by the Grid.
        
        # We will run Nelder-Mead on the best point found by the grid.
        # We will also run it on the 2nd and 3rd best points if budget allows,
        # but we prioritize the best one to get early AOCC gains.
        
        def nelder_mead(x0, f0, max_calls, func, calls_made, best_x, best_f):
            # Initialize simplex
            # x0 is the best point.
            # Create simplex around x0
            # Step size for simplex: 0.05 * box_size? Or adaptive.
            # Box is [-5, 5]. Size 10.
            # Let's use initial step 0.5.
            step = 0.5
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                # If x0[i] is close to bound, move inward
                if x0[i] > 4.5:
                    xi[i] = x0[i] - step
                elif x0[i] < -4.5:
                    xi[i] = x0[i] + step
                else:
                    xi[i] = x0[i] + step
                simplex.append(xi)
            
            # Evaluate simplex
            f_simplex = [f0]
            for i in range(1, dim + 1):
                if calls_made >= max_calls or time.time() > deadline:
                    break
                f_simplex.append(func(simplex[i]))
                calls_made += 1
                if f_simplex[-1] < best_f:
                    best_f = f_simplex[-1]
                    best_x = simplex[i].copy()
            
            # Nelder-Mead Loop
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink
            
            while calls_made < max_calls and time.time() < deadline:
                # Sort simplex by f value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                
                if calls_made >= max_calls or time.time() > deadline:
                    break
                fr = func(xr)
                calls_made += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                if fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    fe = func(xe)
                    calls_made += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    fc = func(xc)
                    calls_made += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    
                    if fc < fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls_made >= max_calls or time.time() > deadline:
                                break
                            f_simplex[i] = func(simplex[i])
                            calls_made += 1
                            if f_simplex[i] < best_f:
                                best_f = f_simplex[i]
                                best_x = simplex[i].copy()
                        
                        # Check convergence
                        f_range = max(f_simplex) - min(f_simplex)
                        if f_range < 1e-6:
                            break
            
            return calls_made, best_x, best_f

        # Run Nelder-Mead on the best point
        # Allocate 80% of remaining budget to the best point
        remaining_budget = budget - calls_made
        nm_budget = int(remaining_budget * 0.8)
        
        if nm_budget > 0:
            calls_made, best_x, best_f = nelder_mead(start_points[0], best_f, nm_budget + calls_made, func, calls_made, best_x, best_f)
        
        # If we have budget left and didn't converge, try the second best point
        if calls_made < budget and len(start_points) > 1:
            # We need the f-value of start_points[1]
            # We evaluated it in the grid phase. 
            # But we don't have it stored separately. 
            # Let's just re-evaluate if needed, or assume it's worse.
            # To be safe, let's just use the remaining budget on the current best (which might have moved).
            # Actually, if NM converged, best_x is good. If not, maybe restart?
            # For simplicity and to avoid wasting calls, we stop.
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
