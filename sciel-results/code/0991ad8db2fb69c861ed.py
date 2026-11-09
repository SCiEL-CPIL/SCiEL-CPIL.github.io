"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Rastrigin-aware optimizer with scipy local refinement and integer-grid exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Exploit Rastrigin structure - sample integer-aligned points
        # The global minimum of Rastrigin is at integer-aligned points (after shift)
        
        # Phase 1a: Evaluate origin and its immediate integer neighbors
        candidates = [np.zeros(dim)]
        
        # Add points with one coordinate at ±1
        for d in range(dim):
            for sign in [1, -1]:
                pt = np.zeros(dim)
                pt[d] = sign
                candidates.append(pt)
        
        # Add points with two coordinates at ±1
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        pt = np.zeros(dim)
                        pt[d1] = s1
                        pt[d2] = s2
                        candidates.append(pt)
        
        # Evaluate all candidates
        for pt in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            fx = call_func(pt)
            if fx is None:
                break
        
        # Phase 1b: Broader integer search if we have budget
        # Sample more integer points
        int_range = range(-4, 5)
        n_integer_samples = 200
        for _ in range(n_integer_samples):
            if call_count >= budget or time.time() > deadline:
                break
            pt = np.array([rng.choice(int_range) for _ in range(dim)], dtype=float)
            pt = np.clip(pt, -5, 5)
            fx = call_func(pt)
            if fx is None:
                break
        
        # Phase 2: Use scipy.optimize.minimize with Nelder-Mead for local refinement
        # This is more robust than manual coordinate descent
        
        if call_count < budget and time.time() < deadline:
            # Try scipy Nelder-Mead from the best point found
            x0 = best_x.copy()
            
            def wrapped_func(x):
                return call_func(x)
            
            try:
                result = minimize(wrapped_func, x0, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                                           'adaptive': True})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass
        
        # Phase 2b: Fine coordinate descent polish after scipy
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Try to snap to nearest integer in each dimension
            # First, try each integer coordinate in each dimension
            for d in range(dim):
                if call_count >= budget or time.time() > deadline:
                    break
                
                best_coord_val = fx
                best_coord = x[d]
                
                # Search integer coordinates near current position
                for ic in range(int(x[d]) - 2, int(x[d]) + 3):
                    if call_count >= budget or time.time() > deadline:
                        break
                    pt = x.copy()
                    pt[d] = float(ic)
                    pt = np.clip(pt, -5, 5)
                    fx_test = call_func(pt)
                    if fx_test is None:
                        break
                    if fx_test < best_coord_val - 1e-10:
                        best_coord_val = fx_test
                        best_coord = float(ic)
                
                if best_coord_val < fx:
                    x[d] = best_coord
                    fx = best_coord_val
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
            
            # Now do fine-tuning around the integer-aligned point
            for step in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                if call_count >= budget or time.time() > deadline:
                    break
                
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            fx_new = call_func(x_new)
                            if fx_new is None:
                                break
                            if fx_new < fx - 1e-12:
                                x = x_new
                                fx = fx_new
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break
        
        # Phase 3: If we haven't found a very good solution, do broader search
        # Check if best_f is reasonably low (close to 0 + offset)
        # Rastrigin at minimum has value 0 before offset
        # After offset, f_opt is in [-100, 100]
        
        # If best_f is still high, we might be in a local minimum
        # Do some random restarts with scipy local refinement
        if call_count < budget and time.time() < deadline and best_f > 5.0:
            # best_f > 5 suggests we're not at the global min (Rastrigin min is 0+offset)
            n_restarts = min(15, (budget - call_count) // 15)
            for _ in range(n_restarts):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Random restart near integer grid
                pt = np.array([rng.choice(range(-4, 5)) for _ in range(dim)], dtype=float)
                # Add small noise
                pt += rng.uniform(-0.5, 0.5, dim)
                pt = np.clip(pt, -5, 5)
                
                fx = call_func(pt)
                if fx is None:
                    break
                
                if fx < best_f:
                    # Try scipy Nelder-Mead from this point
                    try:
                        result = minimize(wrapped_func, pt.copy(), method='Nelder-Mead',
                                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10,
                                                   'adaptive': True})
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = np.clip(result.x, -5, 5).copy()
                    except Exception:
                        pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
