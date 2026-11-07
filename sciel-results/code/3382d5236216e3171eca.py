"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid approach combining CMA-ES for basin exploration and
    coordinate descent / Nelder-Mead for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            x_clipped = np.clip(x, -self.bound, self.bound)
            calls += 1
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Quasi-random scan to find promising basins
        # Use Halton sequence for better coverage
        rng = np.random.default_rng(42)
        
        def halton_sequence(index, dim):
            point = np.zeros(dim)
            for i in range(dim):
                base = i + 2
                f = 1.0
                r = 0.0
                idx = index
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                point[i] = r
            return point

        # Scan phase: 50 points
        n_scan = min(50, budget // 10)
        for i in range(1, n_scan + 1):
            if calls >= budget or time.time() > deadline:
                break
            x = halton_sequence(i, dim) * 10 - 5
            x = np.clip(x, -5, 5)
            eval_and_track(x)

        # Phase 2: Parallel coordinate descent with golden-section 1D search
        # For each coordinate, perform a 1D golden-section search to minimize
        # along that axis, then shuffle coordinate order and repeat.
        
        def golden_section_1d(x_base, axis, f_base, lo, hi, max_evals):
            """Perform 1D golden-section search along `axis` starting from x_base."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return x_base, f_base
            
            # Golden ratio
            gr = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
            inv_gr = 2.0 - gr  # ~1.618
            
            # Initial interval: expand from current value
            x_curr = x_base.copy()
            f_curr = f_base
            
            # Try to find a bracketing interval
            step = 1.0
            for _ in range(8):
                if calls >= budget or time.time() > deadline:
                    break
                # Try expanding in both directions
                x_plus = x_curr.copy()
                x_plus[axis] = min(x_curr[axis] + step, self.bound)
                f_plus = eval_and_track(x_plus)
                
                x_minus = x_curr.copy()
                x_minus[axis] = max(x_curr[axis] - step, -self.bound)
                f_minus = eval_and_track(x_minus)
                
                if f_plus < f_curr:
                    lo = x_curr[axis]
                    hi = x_plus[axis]
                    f_lo = f_curr
                    f_hi = f_plus
                    x_curr = x_plus
                    f_curr = f_plus
                    break
                elif f_minus < f_curr:
                    lo = x_minus[axis]
                    hi = x_curr[axis]
                    f_lo = f_minus
                    f_hi = f_curr
                    x_curr = x_minus
                    f_curr = f_minus
                    break
                else:
                    step *= 2.0
                    if step > 10.0:
                        break
            
            else:
                # No improvement found, return current
                return x_base, f_base
            
            # Now perform golden-section search within [lo, hi]
            a = lo
            b = hi
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            x_c = x_curr.copy()
            x_c[axis] = c
            f_c = eval_and_track(x_c)
            
            x_d = x_curr.copy()
            x_d[axis] = d
            f_d = eval_and_track(x_d)
            
            # Iterate golden-section
            for _ in range(max_evals):
                if calls >= budget or time.time() > deadline:
                    break
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - gr * (b - a)
                    x_c = x_curr.copy()
                    x_c[axis] = c
                    f_c = eval_and_track(x_c)
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + gr * (b - a)
                    x_d = x_curr.copy()
                    x_d[axis] = d
                    f_d = eval_and_track(x_d)
                
                # Check convergence
                if abs(b - a) < 1e-4:
                    break
            
            # Take the best of c and d
            if f_c < f_d:
                x_best_local = x_c
                f_best_local = f_c
            else:
                x_best_local = x_d
                f_best_local = f_d
            
            return x_best_local, f_best_local

        # Main coordinate descent loop
        x_curr = best_x.copy()
        f_curr = best_f
        coord_order = list(range(dim))
        
        # Number of coordinate descent sweeps
        max_sweeps = 50
        evals_per_1d = 15  # Max evaluations per 1D golden-section
        
        for sweep in range(max_sweeps):
            if calls >= budget - 10 or time.time() > deadline:
                break
            
            # Shuffle coordinate order every few sweeps to escape local minima
            if sweep % 3 == 0 and sweep > 0:
                rng.shuffle(coord_order)
            
            improved = False
            for axis in coord_order:
                if calls >= budget - 10 or time.time() > deadline:
                    break
                
                # Perform 1D golden-section search along this axis
                x_new, f_new = golden_section_1d(x_curr, axis, f_curr, 
                                                  -self.bound, self.bound, 
                                                  evals_per_1d)
                
                if f_new < f_curr:
                    x_curr = x_new
                    f_curr = f_new
                    improved = True
            
            # If no improvement, we're likely at a local minimum
            # Try a random perturbation to escape
            if not improved:
                # Small random perturbation
                perturbation = rng.standard_normal(dim) * 0.5
                x_perturbed = np.clip(x_curr + perturbation, -self.bound, self.bound)
                f_perturbed = eval_and_track(x_perturbed)
                
                if f_perturbed < f_curr:
                    x_curr = x_perturbed
                    f_curr = f_perturbed
                    # Restart coordinate descent from new point
                else:
                    # Try larger perturbation
                    perturbation = rng.standard_normal(dim) * 2.0
                    x_perturbed = np.clip(x_curr + perturbation, -self.bound, self.bound)
                    f_perturbed = eval_and_track(x_perturbed)
                    
                    if f_perturbed < f_curr:
                        x_curr = x_perturbed
                        f_curr = f_perturbed
                    else:
                        # No escape found, try a few more random restarts
                        escaped = False
                        for _ in range(3):
                            if calls >= budget - 10 or time.time() > deadline:
                                break
                            perturbation = rng.standard_normal(dim) * 3.0
                            x_perturbed = np.clip(x_curr + perturbation, -self.bound, self.bound)
                            f_perturbed = eval_and_track(x_perturbed)
                            if f_perturbed < f_curr:
                                x_curr = x_perturbed
                                f_curr = f_perturbed
                                escaped = True
                                break
                        if not escaped:
                            # Give up on this local minimum, try fresh random point
                            x_random = rng.uniform(-4, 4, size=dim)
                            f_random = eval_and_track(x_random)
                            if f_random < f_curr:
                                x_curr = x_random
                                f_curr = f_random
                            else:
                                # Break if we can't improve
                                if sweep > 5:
                                    break
            
            # Phase 3: Local refinement with Nelder-Mead
            if calls < budget - 10:
                remaining = budget - calls
                # Use Nelder-Mead on the best point
                x_start = best_x.copy()
                
                class BudgetFunc:
                    def __init__(self, func, x_best, f_best, budget_remaining):
                        self.func = func
                        self.x_best = x_best
                        self.f_best = f_best
                        self.remaining = budget_remaining
                        self.count = 0
                    
                    def __call__(self, x):
                        if self.count >= self.remaining or time.time() > deadline:
                            return self.f_best
                        x_clipped = np.clip(x, -self.bound, self.bound)
                        fx = self.func(x_clipped)
                        self.count += 1
                        if fx < self.f_best:
                            self.f_best = fx
                            self.x_best = x_clipped.copy()
                        return fx
                
                bf = BudgetFunc(func, x_start, best_f, remaining)
                
                try:
                    res = minimize(bf, x_start, method='Nelder-Mead', 
                                  options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
            
            # Phase 4: Final coordinate descent polish
            if calls < budget - 5:
                # Simple coordinate descent
                x_curr = best_x.copy()
                f_curr = best_f
                step = 0.5
                for _ in range(10):
                    if calls >= budget - 5 or time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        if calls >= budget - 5 or time.time() > deadline:
                            break
                        # Try moving in both directions
                        for direction in [1, -1]:
                            x_trial = x_curr.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = eval_and_track(x_trial)
                            if f_trial < f_curr:
                                x_curr = x_trial
                                f_curr = f_trial
                                improved = True
                    
                    if not improved:
                        step *= 0.5
                        if step < 1e-4:
                            break
                    else:
                        step = max(step * 1.1, 0.1)
                
                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x_curr

        return best_x, best_f

# EVOLVE-BLOCK-END
