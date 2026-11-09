import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            
            calls += 1
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
                
            if not np.isfinite(fx):
                fx = float('inf')
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Search using Halton-like low-discrepancy sequence
        # This provides better coverage of the search space than uniform random
        # or simple grids, which is crucial for finding the basin of attraction
        # in shifted/rotated Rosenbrock landscapes.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(index, dims):
            point = np.zeros(dims)
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for d in range(dims):
                point[d] = halton(index, bases[d % len(bases)])
            return point

        # Generate a set of global candidates
        # Include origin and structured points
        candidates = [np.zeros(dim)]
        
        # Add Halton points in [-4, 4]^5
        # 30 points should be sufficient for initial basin identification
        n_global = 30
        for i in range(1, n_global + 1):
            p = halton_sequence(i, dim)
            p = p * 8.0 - 4.0  # Map [0,1] to [-4,4]
            candidates.append(p)
            
        # Evaluate global candidates
        global_results = []
        for p in candidates:
            if calls >= budget or time.time() > deadline:
                break
            f_val = safe_call(p)
            global_results.append((f_val, p))
        
        if not global_results:
            return best_x, best_f

        # Sort by function value to prioritize best basins
        global_results.sort(key=lambda x: x[0])

        # Phase 2: Local Refinement using L-BFGS-B
        # L-BFGS-B is efficient for Rosenbrock due to its quasi-Newton properties
        # and ability to handle bounds. We use the top few global candidates
        # as starting points for multi-start local optimization.
        
        top_k = min(5, len(global_results))
        
        for i in range(top_k):
            if calls >= budget - 10 or time.time() > deadline:
                break
            
            f_init, x_init = global_results[i]
            
            # If we already found a very good value, we might skip some
            # but let's try to refine the best ones first.
            
            # Use L-BFGS-B with tight tolerances for convergence
            try:
                res = minimize(
                    safe_call,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12,
                        'gtol': 1e-6,
                        'maxls': 20
                    }
                )
            except Exception:
                pass

        # Phase 3: Final Polish with Nelder-Mead if budget remains
        # Nelder-Mead is derivative-free and can help if L-BFGS-B stopped early
        # or got stuck in a flat region. We initialize the simplex around the
        # current best point.
        
        remaining = budget - calls
        if remaining > 20 and best_f < float('inf'):
            x0 = best_x
            sigma = 0.1  # Small step size for polishing
            
            # Create initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                if xi[i] + sigma > 5:
                    xi[i] -= 2 * sigma
                elif xi[i] - sigma < -5:
                    xi[i] += 2 * sigma
                else:
                    xi[i] += sigma
                simplex.append(xi)
            
            simplex_f = []
            for i in range(dim + 1):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                simplex_f.append(safe_call(simplex[i]))
            
            if len(simplex_f) == dim + 1:
                simplex_data = list(zip(simplex_f, simplex))
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_shrink = 0.5
                
                # Run NM iterations until budget or convergence
                while calls < budget - 5 and time.time() < deadline:
                    simplex_data.sort(key=lambda item: item[0])
                    f_worst = simplex_data[-1][0]
                    x_worst = simplex_data[-1][1]
                    f_best = simplex_data[0][0]
                    x_best = simplex_data[0][1]
                    
                    if f_best < best_f:
                        best_f = f_best
                        best_x = x_best.copy()
                    
                    f_spread = f_worst - f_best
                    if f_spread < 1e-8:
                        break
                        
                    centroid = np.mean([item[1] for item in simplex_data[:-1]], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    f_r = safe_call(x_r)
                    
                    if f_r < f_best:
                        # Expansion
                        x_e = centroid + gamma * (centroid - x_worst)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = safe_call(x_e)
                        if f_e < f_r:
                            simplex_data[-1] = (f_e, x_e)
                        else:
                            simplex_data[-1] = (f_r, x_r)
                    elif f_r < simplex_data[-2][0]:
                        # Accept reflection
                        simplex_data[-1] = (f_r, x_r)
                    else:
                        # Contraction
                        if f_r < f_worst:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_call(x_c)
                        
                        if f_c <= f_r:
                            simplex_data[-1] = (f_c, x_c)
                        else:
                            # Shrink
                            for i in range(1, len(simplex_data)):
                                simplex_data[i] = (
                                    simplex_data[i][0],
                                    x_best + sigma_shrink * (simplex_data[i][1] - x_best)
                                )
                            # Re-evaluate shrunken points
                            for i in range(1, len(simplex_data)):
                                if calls >= budget - 5 or time.time() > deadline:
                                    break
                                simplex_data[i] = (
                                    safe_call(simplex_data[i][1]),
                                    simplex_data[i][1]
                                )
        
        return best_x, best_f
