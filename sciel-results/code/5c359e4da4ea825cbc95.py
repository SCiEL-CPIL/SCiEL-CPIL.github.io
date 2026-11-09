import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin for 300s limit
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf
        all_evaluated = []

        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            all_evaluated.append((val, x_clipped.copy()))
            return val

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Global Sampling to Initialize Simplexes ---
        # We need 5 diverse starting points for the 5 simplexes.
        # Use a mix of structured and random points.
        
        rng = np.random.default_rng(42)
        
        # Candidate starting points
        start_points = []
        
        # 1. Origin
        start_points.append(np.zeros(n))
        
        # 2. Corners of [-4, 4] (sample a few)
        # Instead of all 32, pick 4 orthogonal corners
        corners = [
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),
            np.array([4.0, 4.0, -4.0, -4.0, 4.0]),
            np.array([-4.0, -4.0, 4.0, 4.0, -4.0])
        ]
        start_points.extend(corners)
        
        # Evaluate these 5 points
        for p in start_points:
            if not time_ok() or calls_made >= total_budget:
                break
            do_call(p)

        # If we have more budget, add some random points to find better basins?
        # The directive says "5 independent simplexes are initialized at the corners... and center".
        # So we stick to these 5.
        
        # --- Phase 2: Parallel Nelder-Mead Ensemble ---
        # We will run 5 Nelder-Mead optimizations.
        # To simulate "parallel" or just run them sequentially but independently,
        # we allocate a portion of the remaining budget to each simplex.
        
        remaining_budget = total_budget - calls_made
        num_simplexes = 5
        # Allocate budget per simplex. 
        # NM in 5D needs at least n+1 = 6 points per simplex.
        # A typical run might take 50-100 iterations.
        # Let's give each simplex 1/5 of the remaining budget, with a minimum.
        
        # We'll store the state of each simplex
        # Simplex is a list of n+1 points and their function values.
        
        simplex_states = []
        
        # Initialize simplexes around the start points
        for i in range(num_simplexes):
            if i >= len(start_points):
                break
            center = start_points[i]
            # Construct simplex: center + n * epsilon * e_i
            # epsilon should be small enough to stay in bounds but large enough to explore.
            # Box is [-5, 5]. Center is in [-4, 4]. 
            # Let's use step size 0.5.
            step = 0.5
            simplex = []
            values = []
            
            # Point 0: center
            f0 = do_call(center)
            simplex.append(center.copy())
            values.append(f0)
            
            for j in range(n):
                pt = center.copy()
                # Check bounds
                if pt[j] + step > 5.0:
                    pt[j] = 5.0 - 0.1
                elif pt[j] - step < -5.0:
                    pt[j] = -5.0 + 0.1
                else:
                    pt[j] += step
                fj = do_call(pt)
                simplex.append(pt)
                values.append(fj)
                
            simplex_states.append({
                'points': simplex,
                'values': values,
                'best_val': min(values)
            })

        # If we ran out of budget during initialization, return best so far
        if calls_made >= total_budget or not time_ok():
            return best_x, best_f

        # Run the optimization loop for all simplexes
        # We will interleave steps from each simplex to ensure fair resource usage
        # and allow early termination if a simplex converges or stagnates.
        
        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Max iterations per simplex to prevent infinite loops
        max_iter_per_simplex = 200 
        # But we are limited by total budget.
        
        # We'll use a "round robin" approach.
        # Each simplex gets a turn.
        
        # To manage budget, we track calls per simplex.
        simplex_calls = [0] * len(simplex_states)
        
        while calls_made < total_budget and time_ok():
            for idx, state in enumerate(simplex_states):
                if not time_ok() or calls_made >= total_budget:
                    break
                if simplex_calls[idx] >= max_iter_per_simplex:
                    continue
                
                points = state['points']
                values = state['values']
                
                # Sort points by value (index 0 is best)
                order = np.argsort(values)
                points = [points[i] for i in order]
                values = [values[i] for i in order]
                
                # If best value is very small, we might consider converged
                # But for Rastrigin, we don't know f_opt. 
                # We'll rely on budget exhaustion.
                
                # Check for convergence: if best and worst are close
                if values[-1] - values[0] < 1e-8:
                    continue
                
                # 1. Centroid of all points except the worst
                centroid = np.sum(points[:-1], axis=0) / n
                
                # 2. Reflection
                worst_point = points[-1]
                xr = centroid + alpha * (centroid - worst_point)
                xr = np.clip(xr, -5.0, 5.0)
                fr = do_call(xr)
                simplex_calls[idx] += 1
                
                if fr < values[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - worst_point)
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = do_call(xe)
                    simplex_calls[idx] += 1
                    
                    if fe < fr:
                        points[-1] = xe
                        values[-1] = fe
                    else:
                        points[-1] = xr
                        values[-1] = fr
                elif fr < values[-2]:
                    # Accept reflection
                    points[-1] = xr
                    values[-1] = fr
                else:
                    # Try contraction
                    # Outside contraction
                    xc = centroid + rho * (worst_point - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = do_call(xc)
                    simplex_calls[idx] += 1
                    
                    if fc < fr:
                        points[-1] = xc
                        values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            points[i] = points[0] + sigma * (points[i] - points[0])
                            fi = do_call(points[i])
                            simplex_calls[idx] += 1
                            values[i] = fi
                        # Note: values[0] remains the best, points[0] unchanged
                        # But we need to re-sort? 
                        # Shrinkage moves all points towards best.
                        # We don't need to re-sort immediately if we just updated.
                        # But the order might change. 
                        # For simplicity, we will re-sort at the start of the next iteration.
                
                # Update state
                state['points'] = points
                state['values'] = values
                state['best_val'] = min(values)
                
                # If this simplex found a new global best, we might want to 
                # restart other simplexes? No, the ensemble handles that.
                # But if one simplex is way better, it might be worth focusing there.
                # However, NM is local. The ensemble is the global search mechanism.
                
                # Optional: If a simplex converges (step size too small), 
                # we could restart it around the current global best?
                # Let's check if the simplex is very small.
                # Distance between best and second best
                if len(values) > 1:
                    # Sort again to be sure
                    sorted_vals = sorted(values)
                    if sorted_vals[1] - sorted_vals[0] < 1e-6:
                         # Converged? 
                         # We can leave it be, it will just waste calls.
                         # Better to stop this simplex to save budget for others?
                         # Or restart it?
                         # Restarting is risky if we don't have a good global best.
                         # Let's just let it run out of max_iter.
                         pass

        return best_x, best_f
