import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

def halton_sequence(n, dim, skip=1):
    """
    Generate n points in [0, 1]^dim using the Halton sequence.
    skip: number of initial points to skip (to avoid 0,0,...,0 which is often a bad start for Halton in some contexts, or just to center it).
    """
    points = []
    for i in range(skip, skip + n):
        point = []
        for j in range(1, dim + 1):
            f = 1.0
            r = 0.0
            i_temp = i
            while i_temp > 0:
                f /= j
                r += f * (i_temp % j)
                i_temp //= j
            point.append(r)
        points.append(point)
    return np.array(points)

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Phase 1: Global Exploration with Halton Sequence
        # Generate diverse points in [-4, 4]^5
        # We map [0, 1]^dim to [-4, 4]^dim
        n_halton = 20
        halton_pts = halton_sequence(n_halton, self.dim, skip=1)
        # Map to [-4, 4]
        halton_pts = halton_pts * 8.0 - 4.0
        
        # Add some deterministic points: Origin, and points on axes
        starts = []
        starts.append(np.zeros(self.dim))
        for i in range(self.dim):
            for val in [-3.0, 3.0]:
                p = np.zeros(self.dim)
                p[i] = val
                starts.append(p)
        
        # Add Halton points
        for p in halton_pts:
            starts.append(p)
        
        # Shuffle starts? No, Halton is already well-distributed.
        # But we want to evaluate the "cheap" deterministic ones first?
        # Actually, evaluating all is better for sorting.
        # Total starts: 1 + 2*5 + 20 = 32 points.
        # This is a significant chunk of 500, but ensures we find the basin.
        
        evaluated_starts = []
        for x0 in starts:
            if call_count >= self.budget - 50:
                break
            fx = tracked_func(x0)
            evaluated_starts.append((fx, x0.copy()))
            
        if not evaluated_starts:
            return best_x, best_fx

        # Sort by function value
        evaluated_starts.sort(key=lambda item: item[0])
        
        # Phase 2: Racing Local Refinement
        # We will try the top K candidates.
        # For each, we give a small budget to see if it converges.
        # If it does, we take the rest of the budget for final polish.
        
        top_k = min(5, len(evaluated_starts))
        winner_x = None
        winner_fx = np.inf
        
        for i in range(top_k):
            if not time_ok() or call_count >= self.budget - 20:
                break
            
            fx_start, x_start = evaluated_starts[i]
            
            # If we already found a very good solution from a previous race, stop
            if i > 0 and best_fx < 1e-2:
                break
                
            # Give this candidate a chance to converge
            # Budget for this race: 40 calls
            race_budget = 40
            if call_count + race_budget > self.budget - 20:
                race_budget = self.budget - call_count - 20
            if race_budget < 5:
                break
                
            try:
                res = minimize(
                    tracked_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxfun': race_budget,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
            except Exception:
                pass
            
            # Update winner if this run improved the global best
            # Note: tracked_func updates best_x/best_fx automatically.
            # So if this run found a better point, best_x is updated.
            
            # Check if we should stop racing
            # If best_fx is very low, we are likely in the global basin.
            if best_fx < 1e-1:
                break
            # If this run didn't improve best_fx at all (unlikely, but possible if L-BFGS-B fails immediately),
            # and we have other starts, continue.
            # But since we sorted, x_start is likely good.
        
        # Phase 3: Final Polish
        # Use remaining budget to polish the current best_x
        if time_ok() and call_count < self.budget - 5:
            remaining = self.budget - call_count - 5
            if remaining > 10:
                try:
                    minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 1000,
                            'maxfun': remaining,
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
                except Exception:
                    pass
            
            # If still budget left, try Nelder-Mead for robustness?
            # Maybe not needed if L-BFGS-B worked.
            if time_ok() and call_count < self.budget - 5:
                remaining = self.budget - call_count - 5
                if remaining > 20:
                    try:
                        minimize(
                            tracked_func,
                            best_x,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 100,
                                'maxfev': remaining,
                                'adaptive': True,
                                'xatol': 1e-10,
                                'fatol': 1e-12
                            }
                        )
                    except Exception:
                        pass

        return best_x, best_fx
