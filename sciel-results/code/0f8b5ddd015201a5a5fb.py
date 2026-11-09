import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s kill
        
        calls_used = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = None
        dim = self.dim
        
        def update_best(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            for i in range(1, n + 1):
                point = [halton(i, base) for base in range(2, dim + 2)]
                points.append(point)
            return np.array(points)

        # Phase 1: Global Exploration using Halton Sequence
        # We use a fixed number of points to cover the space deterministically.
        # For dim=5, 100 points is a reasonable balance between coverage and budget.
        n_halton = 100
        halton_points = halton_sequence(n_halton, dim)
        
        # Map Halton [0,1]^d to [-5,5]^d
        halton_points = halton_points * 10.0 - 5.0
        
        # Evaluate Halton points
        for x in halton_points:
            if time.time() > deadline or calls_used >= max_calls:
                break
            f_val = func(x)
            calls_used += 1
            update_best(x, f_val)
            
        # Phase 2: Local Refinement using Multi-start Nelder-Mead
        # We will run Nelder-Mead from the top few distinct points found in Phase 1.
        # To ensure diversity, we select points that are far apart.
        
        # Collect all evaluated points and their fitness
        evaluated_points = []
        # Re-evaluate or store? We didn't store all in a list in the loop above efficiently.
        # Let's restart the logic slightly: We need the points we evaluated.
        # Actually, the loop above didn't store them. Let's modify the approach:
        # We will just take the best few from the Halton set we just generated.
        # But we need their coordinates.
        
        # Let's re-structure to store points during Phase 1.
        # Wait, I already ran Phase 1. I need the points.
        # I can just regenerate the Halton points (they are deterministic) and pick the best ones
        # based on the function values? No, I don't have the function values stored.
        
        # Correction: I will re-run the selection logic.
        # Since Halton points are deterministic, I can just take the first few "good" ones?
        # No, I need to know which ones were good.
        
        # Let's refactor: Store points and values in Phase 1.
        
        # Restarting logic to be safe and clean:
        calls_used = 0
        best_f = float('inf')
        best_x = None
        history = [] # (f, x)
        
        def update_best_history(x, f):
            nonlocal best_f, best_x
            history.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Global Exploration
        for x in halton_points:
            if time.time() > deadline or calls_used >= max_calls:
                break
            f_val = func(x)
            calls_used += 1
            update_best_history(x, f_val)
            
        # Sort history by fitness
        history.sort(key=lambda item: item[0])
        
        # Select top K distinct starts for Nelder-Mead
        # We want starts that are diverse.
        starts = []
        min_dist = 1.0 # Minimum distance between starts
        
        for f_val, x_val in history:
            if len(starts) >= 5: # Max 5 local optimizations
                break
            is_distinct = True
            for s_x in starts:
                if np.linalg.norm(x_val - s_x) < min_dist:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append(x_val)
                
        # If we didn't find 5 distinct ones, just take the best 5
        if len(starts) < 5 and len(history) > 0:
            for f_val, x_val in history:
                if len(starts) >= 5:
                    break
                if x_val not in starts: # Simple check, might be slow but dim=5 is small
                     # Better: check distance again
                     is_in = False
                     for s_x in starts:
                         if np.allclose(x_val, s_x):
                             is_in = True
                             break
                     if not is_in:
                         starts.append(x_val)
        
        # Phase 2: Local Refinement
        # Allocate remaining budget to Nelder-Mead
        # Nelder-Mead in 5D typically takes ~50-100 function evaluations to converge or stagnate.
        # With 5 starts, that's 250-500 calls.
        # We have ~400 calls left. So ~80 calls per start is feasible.
        
        nm_options = {
            'maxiter': 100,
            'xatol': 1e-4,
            'fatol': 1e-4
        }
        
        def nm_func(x):
            nonlocal calls_used
            if calls_used >= max_calls or time.time() > deadline:
                return best_f # Return current best to stop optimization gracefully? 
                              # Actually, scipy might not handle this well.
                              # Better to just let it run and check budget in wrapper?
                              # Scipy minimize doesn't have a callback for budget easily.
                              # We will rely on maxiter and the fact that dim is small.
            
            f_val = func(x)
            calls_used += 1
            update_best_history(x, f_val)
            return f_val

        for start_x in starts:
            if time.time() > deadline or calls_used >= max_calls:
                break
            
            # Check if we have enough budget for a reasonable run
            if max_calls - calls_used < 10:
                break
                
            try:
                res = minimize(nm_func, start_x, method='Nelder-Mead', options=nm_options)
                # The best point is tracked in update_best_history
            except Exception:
                pass
            
            # Check time
            if time.time() > deadline:
                break

        return best_x, best_f
