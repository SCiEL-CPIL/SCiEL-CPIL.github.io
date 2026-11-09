import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard: The evaluator kills at 330s, but we want to return well before.
        # Let's aim to finish within 280s to be safe.
        start_time = time.time()
        deadline = start_time + 280
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            # Ensure x is a numpy array of correct shape and within bounds
            # The problem statement says func(x) clips, but we should ensure our inputs are clean
            # to avoid any potential issues with the evaluator's clipping logic if it's strict.
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            
            calls += 1
            try:
                val = float(func(x))
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration / Initialization
        # Bent Cigar is convex. We don't need extensive global search.
        # However, a hidden shift might place the minimum away from 0.
        # Let's try a few random points to pick a good center for the simplex.
        
        # We have 500 calls. NM in 5D typically needs ~50-100 calls to converge well from a decent start.
        # Spending 5-10 calls on initialization is acceptable.
        
        num_initial = 5
        candidates = []
        for _ in range(num_initial):
            if time.time() > deadline:
                break
            x_init = rng.uniform(-4.0, 4.0, dim)
            f_init = safe_func(x_init)
            candidates.append((x_init, f_init))
            
        if not candidates:
            # Fallback if time ran out immediately (unlikely)
            return np.zeros(dim), float('inf')
            
        # Sort and pick the best
        candidates.sort(key=lambda x: x[1])
        x0, f0 = candidates[0]
        
        # Initialize Simplex
        # Standard NM simplex: x0 and x0 + delta * e_i
        # Choose delta based on the domain size. 1.0 is a reasonable start for [-5,5].
        # If the function is very sharp, a smaller delta might be better, but 1.0 is robust.
        delta = 1.0
        
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += delta
            x_i = np.clip(x_i, -5.0, 5.0)
            f_i = safe_func(x_i)
            simplex.append(x_i)
            simplex_f.append(f_i)
            
            if time.time() > deadline:
                break

        # Main Nelder-Mead Loop
        # Parameters
        rho = 1.0  # Reflection
        chi = 2.0  # Expansion
        gamma = 0.5  # Contraction
        sigma = 0.5  # Shrink
        
        max_iter = 200 # Safety cap, but budget/time will likely stop it earlier
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Sort simplex by function value
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Check convergence
            # If the spread of function values is very small, we are converged
            if simplex_f[-1] - simplex_f[0] < 1e-12:
                break
            
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            xr = centroid + rho * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_func(xr)
            
            if fr < simplex_f[0]:
                # Reflection is better than the best point
                # Try Expansion
                xe = centroid + chi * (xr - centroid)
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_func(xe)
                
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Reflection is better than the second worst, but not the best
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Reflection is worse than the second worst
                # Try Contraction
                xc = centroid + gamma * (simplex[-1] - centroid)
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_func(xc)
                
                if fc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        simplex_f[i] = safe_func(simplex[i])
                    
                    # Re-sort after shrink
                    indices = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in indices]
                    simplex_f = [simplex_f[i] for i in indices]

        return best_x, best_f
