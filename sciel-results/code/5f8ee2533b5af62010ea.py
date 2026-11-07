import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            # Time guard
            if time.time() > deadline:
                return 1e10
            # Budget guard
            if call_count >= self.budget:
                return 1e10
            
            call_count += 1
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Phase 1: Initialization and Exploration
        # Generate a set of candidate starting points
        # We use a mix of deterministic and random points
        rng = np.random.RandomState(42)
        
        candidates = []
        # Origin
        candidates.append(np.zeros(self.dim))
        # Random points in the central region
        for _ in range(4):
            candidates.append(rng.uniform(-4, 4, self.dim))
        
        # Evaluate candidates
        # We must be careful not to exceed budget here, but 5 calls is safe
        for x_cand in candidates:
            f_cand = wrapped_func(x_cand)
            if f_cand < best_f:
                best_f = f_cand
                best_x = x_cand.copy()

        # Phase 2: Main Nelder-Mead Optimization
        # Initialize simplex for NM
        # Use a random rotation to align with potential ridges
        # The simplex should be large enough to explore the neighborhood
        delta = 2.0
        x0 = best_x.copy()
        
        # Create a random orthogonal matrix to rotate the simplex
        # This helps in case the ridge is rotated
        Q, _ = np.linalg.qr(rng.randn(self.dim, self.dim))
        
        # Construct simplex: x0 + delta * Q * e_i
        simplex = [x0]
        for i in range(self.dim):
            e_i = np.zeros(self.dim)
            e_i[i] = 1.0
            v = Q @ e_i
            xi = x0 + delta * v
            simplex.append(xi)
        
        # Run NM
        # maxiter is set high, but we rely on call_count and time
        # xatol and fatol control convergence
        try:
            res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                           options={'initial_simplex': simplex, 
                                    'maxiter': 200, 
                                    'xatol': 1e-4, 
                                    'fatol': 1e-4,
                                    'adaptive': True})
        except Exception:
            pass

        # Phase 3: Polishing
        # If we have budget left, do a fine-grained search
        if call_count < self.budget - 10 and time.time() < deadline - 5:
            # Create a smaller simplex around the current best
            delta_small = 0.1
            x0_polish = best_x.copy()
            simplex_polish = [x0_polish]
            for i in range(self.dim):
                e_i = np.zeros(self.dim)
                e_i[i] = 1.0
                # Use the same rotation for consistency, or axis-aligned for fine tune?
                # Axis aligned might be safer for final polish if the ridge is already aligned
                # But if the ridge is still rotated, the previous NM should have aligned it.
                # Let's use a small random rotation again to be safe
                Q2, _ = np.linalg.qr(rng.randn(self.dim, self.dim))
                v = Q2 @ e_i
                xi = x0_polish + delta_small * v
                simplex_polish.append(xi)
                
            try:
                res2 = minimize(wrapped_func, x0_polish, method='Nelder-Mead', 
                               options={'initial_simplex': simplex_polish, 
                                        'maxiter': 50, 
                                        'xatol': 1e-6, 
                                        'fatol': 1e-6})
            except Exception:
                pass

        return best_x, best_f
