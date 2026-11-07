"""Coordinate-wise sequential search with golden-section refinement for Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Coordinate-wise sequential search with golden-section refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls_made += 1
            return val

        # Phase 1: Coordinate-wise sequential search with golden-section refinement
        # For each dimension, do a 1D optimization using golden-section search
        
        # Initial point: try origin first, then random if it's bad
        x = np.zeros(dim)
        fx = safe_call(x)
        if fx is None:
            return x, np.inf
        
        best_x = x.copy()
        best_fx = fx

        # Also try a few random starting points to avoid getting stuck
        for _ in range(3):
            x_rand = rng.uniform(-4, 4, dim)
            fx_rand = safe_call(x_rand)
            if fx_rand is None:
                break
            if fx_rand < best_fx:
                best_fx = fx_rand
                best_x = x_rand.copy()

        # Golden-section search on a single coordinate
        def golden_section_1d(x_base, coord_idx, lo, hi, max_iters=50):
            """
            Perform golden-section search on coordinate coord_idx.
            Returns the best point and its function value.
            """
            nonlocal calls_made, best_x, best_fx
            
            if hi - lo < 1e-10:
                return x_base.copy(), None
            
            gr = (np.sqrt(5) - 1) / 2  # golden ratio
            inv_gr = 1.0 / gr
            
            a = lo
            b = hi
            c = b - inv_gr * (b - a)
            d = a + inv_gr * (b - a)
            
            # Evaluate initial points
            x_c = x_base.copy()
            x_c[coord_idx] = c
            x_d = x_base.copy()
            x_d[coord_idx] = d
            
            fc = safe_call(x_c)
            if fc is None:
                return x_base.copy(), None
            fd = safe_call(x_d)
            if fd is None:
                return x_base.copy(), None
            
            best_local_x = x_base.copy()
            best_local_fx = min(fc, fd)
            if fc < fd:
                best_local_x = x_c.copy()
            else:
                best_local_x = x_d.copy()
            
            for _ in range(max_iters):
                if time.time() >= deadline:
                    break
                if calls_made >= budget - 50:  # Reserve budget for later phases
                    break
                if b - a < 1e-10:
                    break
                
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - inv_gr * (b - a)
                    x_c = x_base.copy()
                    x_c[coord_idx] = c
                    fc = safe_call(x_c)
                    if fc is None:
                        break
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + inv_gr * (b - a)
                    x_d = x_base.copy()
                    x_d[coord_idx] = d
                    fd = safe_call(x_d)
                    if fd is None:
                        break
                
                # Update local best
                if fc < best_local_fx:
                    best_local_fx = fc
                    best_local_x = x_c.copy()
                if fd < best_local_fx:
                    best_local_fx = fd
                    best_local_x = x_d.copy()
            
            return best_local_x, best_local_fx

        # Phase 1a: Multi-cycle coordinate-wise optimization
        # Do multiple passes, each pass optimizing each coordinate in sequence
        num_cycles = 5
        for cycle in range(num_cycles):
            if time.time() >= deadline:
                break
            if calls_made >= budget - 80:
                break
            
            for coord in range(dim):
                if time.time() >= deadline:
                    break
                if calls_made >= budget - 80:
                    break
                
                # Determine search range for this coordinate
                # Use a range centered on current value
                center = best_x[coord]
                # Start with a wide range, narrow down
                width = max(1.0, 5.0 - cycle)  # Shrink range each cycle
                lo = max(-5.0, center - width)
                hi = min(5.0, center + width)
                
                x_new, fx_new = golden_section_1d(best_x, coord, lo, hi, max_iters=30)
                if fx_new is not None and fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()
                elif fx_new is None:
                    # Keep best_x as is
                    pass

        # Phase 1b: Fine-tune each coordinate with tighter ranges
        for _ in range(3):
            if time.time() >= deadline:
                break
            if calls_made >= budget - 60:
                break
            
            for coord in range(dim):
                if time.time() >= deadline:
                    break
                if calls_made >= budget - 60:
                    break
                
                center = best_x[coord]
                width = 0.5
                lo = max(-5.0, center - width)
                hi = min(5.0, center + width)
                
                x_new, fx_new = golden_section_1d(best_x, coord, lo, hi, max_iters=40)
                if fx_new is not None and fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()

        # Phase 2: Nelder-Mead polish
        remaining_budget = budget - calls_made
        if remaining_budget > 20 and time.time() < deadline:
            nm_calls = [0]
            
            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining_budget:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                z_clipped = np.clip(z, -5, 5)
                return func(z_clipped)
            
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
