import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Time guard: keep a safety margin before the 300s wall-clock limit.
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = np.zeros(n)
        best_f = float('inf')
        evals_left = budget
        
        # Store history of evaluated points and values
        history = []

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
            history.append((f, x.copy()))

        def generate_local_grid(center, radius, res):
            """Generate a grid of points in a box around center."""
            # Define bounds
            low = np.maximum(center - radius, -5.0)
            high = np.minimum(center + radius, 5.0)
            
            # If the box is degenerate (center at boundary), ensure we have points
            # Generate linspace points
            coords = []
            for i in range(n):
                if high[i] - low[i] < 1e-10:
                    coords.append(np.array([low[i]]))
                else:
                    coords.append(np.linspace(low[i], high[i], res))
            
            grids = np.meshgrid(*coords, indexing='ij')
            points = np.stack(grids, axis=-1).reshape(-1, n)
            return points

        # Phase 1: Global Coarse Grid
        # Use 3 points per dimension: 3^5 = 243 points
        global_res = 3
        num_global = global_res ** n
        
        if evals_left >= num_global:
            global_points = generate_local_grid(np.zeros(n), 5.0, global_res)
            for x in global_points:
                if evals_left <= 0 or time.time() > deadline:
                    break
                f = func(x)
                update_best(x, f)
                evals_left -= 1
        else:
            # Fallback to 2^5 if budget is tight (shouldn't happen with 500)
            global_res = 2
            num_global = global_res ** n
            global_points = generate_local_grid(np.zeros(n), 5.0, global_res)
            for x in global_points:
                if evals_left <= 0 or time.time() > deadline:
                    break
                f = func(x)
                update_best(x, f)
                evals_left -= 1

        # Phase 2: Local Refinement on Top K Candidates
        # Sort history by function value
        if len(history) > 0:
            history.sort(key=lambda item: item[0])
            top_k = 5
            candidates = [h[1] for h in history[:top_k]]
            
            # For each candidate, run a local grid search
            local_res = 2 # 2^5 = 32 points
            local_radius = 1.5 # Search in a box of size 3x3x3x3x3 around candidate
            
            for c in candidates:
                if evals_left <= 0 or time.time() > deadline:
                    break
                local_points = generate_local_grid(c, local_radius, local_res)
                for x in local_points:
                    if evals_left <= 0 or time.time() > deadline:
                        break
                    f = func(x)
                    update_best(x, f)
                    evals_left -= 1

        # Phase 3: Final Polish on Best Point
        if evals_left > 0 and time.time() < deadline and len(history) > 0:
            # Re-sort to find the absolute best
            history.sort(key=lambda item: item[0])
            best_candidate = history[0][1]
            
            # Fine grid around the best point
            fine_res = 2
            fine_radius = 0.5
            fine_points = generate_local_grid(best_candidate, fine_radius, fine_res)
            
            # Use remaining budget, but don't exceed the generated points
            # If we have more budget, we could do multiple rounds, but let's stick to one fine grid
            # 2^5 = 32 points.
            for x in fine_points:
                if evals_left <= 0 or time.time() > deadline:
                    break
                f = func(x)
                update_best(x, f)
                evals_left -= 1
                
            # If we still have budget, do an even finer grid
            if evals_left > 0 and time.time() < deadline:
                history.sort(key=lambda item: item[0])
                best_candidate = history[0][1]
                finer_res = 2
                finer_radius = 0.1
                finer_points = generate_local_grid(best_candidate, finer_radius, finer_res)
                for x in finer_points:
                    if evals_left <= 0 or time.time() > deadline:
                        break
                    f = func(x)
                    update_best(x, f)
                    evals_left -= 1

        return best_x, best_f
