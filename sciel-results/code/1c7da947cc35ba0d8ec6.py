"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start coordinate descent with golden-section line search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        def time_check():
            return time.time() < deadline

        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_check():
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Golden-section search along one coordinate
        def golden_section_1d(x, axis, lo, hi, max_evals=20):
            """Minimize f along one axis using golden-section method."""
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_check():
                return x, None

            phi = (1.0 + np.sqrt(5.0)) / 2.0
            invphi = 1.0 / phi
            invphi2 = invphi * invphi

            a = lo
            b = hi
            c = b - invphi * (b - a)
            d = a + invphi * (b - a)

            # Evaluate at c and d
            x_c = x.copy()
            x_c[axis] = c
            x_d = x.copy()
            x_d[axis] = d

            fc = eval_point(x_c)
            fd = eval_point(x_d)

            if fc is None or fd is None:
                # Return the better of the two evaluated points
                if fc is not None and (fd is None or fc < fd):
                    return x_c, fc
                elif fd is not None:
                    return x_d, fd
                return x, None

            best_local = x_c if fc < fd else x_d
            best_local_f = min(fc, fd)

            for _ in range(max_evals - 2):
                if calls >= budget or not time_check():
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - invphi * (b - a)
                    x_c = x.copy()
                    x_c[axis] = c
                    fc = eval_point(x_c)
                    if fc is None:
                        break
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + invphi * (b - a)
                    x_d = x.copy()
                    x_d[axis] = d
                    fd = eval_point(x_d)
                    if fd is None:
                        break

                if fc < fd:
                    if fc < best_local_f:
                        best_local = x_c
                        best_local_f = fc
                else:
                    if fd < best_local_f:
                        best_local = x_d
                        best_local_f = fd

            return best_local, best_local_f

        # Coordinate descent: cycle through all axes
        def coordinate_descent(x, max_rounds=5):
            """Perform coordinate-wise line search cycling through all axes."""
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_check():
                return x

            fx = eval_point(x)
            if fx is None:
                return x

            for _ in range(max_rounds):
                if calls >= budget or not time_check():
                    break
                improved = False
                for axis in range(dim):
                    if calls >= budget or not time_check():
                        break
                    # Determine search interval: expand from current value
                    x_curr = x[axis]
                    # Try expanding interval to find a bracket
                    step = 1.0
                    lo = x_curr
                    hi = x_curr
                    # First, try to find a direction of improvement
                    x_plus = x.copy()
                    x_plus[axis] = min(5.0, x_curr + step)
                    x_minus = x.copy()
                    x_minus[axis] = max(-5.0, x_curr - step)
                    
                    fp = eval_point(x_plus)
                    fm = eval_point(x_minus)
                    
                    if fp is None and fm is None:
                        break
                    
                    # Determine which direction is better
                    if fp is not None and (fm is None or fp < fm):
                        # Move in positive direction
                        lo = x_curr
                        hi = x_plus[axis]
                        # Expand if still improving
                        for _exp in range(3):
                            if calls >= budget or not time_check():
                                break
                            new_hi = min(5.0, hi + step)
                            if new_hi <= hi:
                                break
                            x_new = x.copy()
                            x_new[axis] = new_hi
                            fn = eval_point(x_new)
                            if fn is None:
                                break
                            if fn < fp:
                                lo = hi
                                hi = new_hi
                                fp = fn
                            else:
                                break
                    else:
                        # Move in negative direction
                        lo = x_minus[axis]
                        hi = x_curr
                        for _exp in range(3):
                            if calls >= budget or not time_check():
                                break
                            new_lo = max(-5.0, lo - step)
                            if new_lo >= lo:
                                break
                            x_new = x.copy()
                            x_new[axis] = new_lo
                            fn = eval_point(x_new)
                            if fn is None:
                                break
                            if fn < fm:
                                hi = lo
                                lo = new_lo
                                fm = fn
                            else:
                                break

                    # Now do golden-section search in [lo, hi]
                    if hi - lo > 1e-6:
                        x_new, _ = golden_section_1d(x, axis, lo, hi, max_evals=15)
                        if x_new is not None:
                            x = x_new
                            improved = True

                if not improved:
                    break

            return x

        # Phase 1: Deterministic grid of starting points
        # Create a coarse grid in 5D
        # Use a stratified approach: center + axis-aligned points + corner-like points
        start_points = []
        
        # Center
        start_points.append(np.zeros(dim))
        
        # Axis-aligned points at ±2.5
        for axis in range(dim):
            for val in [-2.5, 2.5]:
                pt = np.zeros(dim)
                pt[axis] = val
                start_points.append(pt)
        
        # Points at ±1.0 on each axis
        for axis in range(dim):
            for val in [-1.0, 1.0]:
                pt = np.zeros(dim)
                pt[axis] = val
                start_points.append(pt)
        
        # Some diagonal-like points
        for sign in [-1, 1]:
            pt = np.full(dim, sign * 2.0)
            start_points.append(pt)
        
        # A few more spread points
        for val in [-4.0, -3.0, 3.0, 4.0]:
            pt = np.full(dim, val * 0.5)
            start_points.append(pt)

        # Deduplicate
        unique_starts = []
        for pt in start_points:
            if not any(np.allclose(pt, u) for u in unique_starts):
                unique_starts.append(pt)

        # Phase 2: For each start point, run coordinate descent
        # Budget allocation: give more budget to earlier/better starts
        for i, start_pt in enumerate(unique_starts):
            if calls >= budget or not time_check():
                break
            
            # Allocate remaining budget proportionally
            remaining_starts = len(unique_starts) - i
            if remaining_starts > 0:
                # Reserve some budget for later starts
                budget_for_this = min(50, max(10, (budget - calls) // remaining_starts))
            else:
                budget_for_this = budget - calls
            
            # Evaluate the start point
            fx = eval_point(start_pt)
            if fx is None:
                break
            
            # Run coordinate descent with limited budget
            # Track calls before and after
            calls_before = calls
            x_result = coordinate_descent(start_pt.copy(), max_rounds=3)
            calls_used = calls - calls_before
            
            # If we used very few calls, the descent converged quickly
            # If we hit budget, continue with next start

        # Phase 3: Final refinement around the best point found
        if best_x is not None and calls < budget and time_check():
            # Do a final round of coordinate descent with tighter intervals
            x = best_x.copy()
            fx = eval_point(x)
            if fx is not None:
                for _ in range(2):  # 2 rounds of fine coordinate search
                    if calls >= budget or not time_check():
                        break
                    for axis in range(dim):
                        if calls >= budget or not time_check():
                            break
                        x_curr = x[axis]
                        # Tight interval around current best
                        step = 0.1
                        lo = max(-5.0, x_curr - step * 5)
                        hi = min(5.0, x_curr + step * 5)
                        if hi - lo > 1e-8:
                            x_new, _ = golden_section_1d(x, axis, lo, hi, max_evals=10)
                            if x_new is not None:
                                x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
