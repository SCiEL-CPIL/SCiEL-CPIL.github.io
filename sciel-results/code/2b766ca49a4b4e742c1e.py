"""Rewritten optimizer using Multi-Start Adaptive Nelder-Mead for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a Multi-Start Nelder-Mead strategy.
    
    The Lunacek bi-Rastrigin function (f24) is multimodal and non-smooth.
    Standard gradient-based methods fail. PSO can be slow to converge.
    Nelder-Mead (a simplex method) is effective for low-dimensional problems
    and can handle non-smoothness if the simplex size is appropriate.
    We use a multi-start strategy to avoid local minima, with adaptive
    restarts if the simplex stagnates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin
        self._call_count = 0
        best_x_global = None
        best_f_global = np.inf

        def counted_func(x):
            nonlocal best_x_global, best_f_global
            if self._call_count >= self.budget:
                return np.inf
            self._call_count += 1
            # Clip input to be safe, although func does it too
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            if val < best_f_global:
                best_f_global = val
                best_x_global = x_clipped.copy()
            return val

        def nelder_mead(start_x, max_evals):
            """
            Run Nelder-Mead optimization from start_x for up to max_evals calls.
            Returns the best point and value found within this run.
            """
            nonlocal best_x_global, best_f_global
            if self._call_count >= self.budget or time.time() > deadline:
                return start_x, np.inf

            # Initialize Simplex
            x0 = np.clip(start_x, -5, 5)
            f0 = counted_func(x0)
            
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            # Initial step size
            # For dim 5, a step of 0.5 is a reasonable heuristic for [-5,5]
            step = 0.5 
            for i in range(self.dim):
                xi = x0.copy()
                # Ensure we stay within bounds for the initial simplex
                if x0[i] + step > 5.0:
                    xi[i] = x0[i] - step
                else:
                    xi[i] = x0[i] + step
                simplex.append(xi)
                simplex_f.append(counted_func(xi))
                
                if self._call_count >= self.budget or time.time() > deadline:
                    break

            if len(simplex) < self.dim + 1:
                # Not enough budget to even initialize simplex properly
                idx_min = np.argmin(simplex_f)
                return simplex[idx_min], simplex_f[idx_min]

            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Stagnation tracking
            best_f_in_nm = min(simplex_f)
            best_x_in_nm = simplex[np.argmin(simplex_f)].copy()
            stagnant_count = 0
            stagnation_threshold = 1e-8
            max_stagnant = 10 # Restart if no improvement for 10 iterations
            
            iters = 0
            # We use max_evals as a soft limit, but also check global budget
            while self._call_count < self.budget and time.time() < deadline:
                iters += 1
                if iters > max_evals:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Check for stagnation
                if simplex_f[0] < best_f_in_nm - stagnation_threshold:
                    best_f_in_nm = simplex_f[0]
                    best_x_in_nm = simplex[0].copy()
                    stagnant_count = 0
                else:
                    stagnant_count += 1
                    
                if stagnant_count > max_stagnant:
                    # Stagnation detected. Break out to allow a restart.
                    break

                # Calculate Centroid (excluding worst point)
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = counted_func(xr)
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = counted_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept Reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = counted_func(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrinkage
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = counted_func(simplex[i])
                            if self._call_count >= self.budget:
                                break
                        if self._call_count >= self.budget:
                            break

            return best_x_in_nm, best_f_in_nm

        # --- Main Multi-Start Logic ---
        
        # Strategy: 
        # 1. Try the origin (0,0,0,0,0) first, as f24 often has a global min near there 
        #    (depending on the hidden shift, but it's a good heuristic).
        # 2. Try a few random points.
        # 3. Allocate budget to each start. If a start stagnates, move to the next.
        # 4. If budget remains, polish the global best.
        
        rng = np.random.default_rng(123)
        
        # Define start points
        # Start 1: Origin
        start_points = [np.zeros(self.dim)]
        
        # Start 2-5: Random points in [-4, 4]^5
        for _ in range(4):
            start_points.append(rng.uniform(-4, 4, self.dim))
            
        # Allocate budget per start
        # Total budget 500. 
        # We want to give each start a chance, but not waste too much on a bad one.
        # Let's say 80 calls per start for the first 4 starts (320 total), 
        # leaving 180 for the 5th start or polishing.
        # Actually, let's just loop and break if stagnant.
        
        starts_used = 0
        for start_pt in start_points:
            if self._call_count >= self.budget or time.time() > deadline:
                break
            
            # Allocate a sub-budget for this NM run
            # If we have 4 starts left, give each ~100.
            # Simpler: Just run NM with a max of 100 evals per start.
            # If it finishes early due to stagnation, we move on.
            max_evals_per_start = 100
            
            x_best_local, f_best_local = nelder_mead(start_pt, max_evals_per_start)
            
            # Update global best (already done in counted_func, but ensure we have a valid x)
            if best_x_global is None:
                best_x_global = x_best_local
                best_f_global = f_best_local

        # If we still have budget, do a final polish from the global best
        if self._call_count < self.budget and best_x_global is not None:
            remaining_budget = self.budget - self._call_count
            # Use all remaining budget for a final NM polish from the best point
            # This helps refine the solution if the previous runs converged to a local min
            # but the simplex was still large enough to move.
            x_polished, f_polished = nelder_mead(best_x_global, remaining_budget)
            if f_polished < best_f_global:
                best_f_global = f_polished
                best_x_global = x_polished

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
