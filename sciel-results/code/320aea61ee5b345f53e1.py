import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf
        rng = np.random.default_rng(42)

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
            return val

        def time_ok():
            return time.time() < deadline

        # Initialize 5 simplexes: 4 corners + 1 center
        # Corners: combinations of +/-4 to stay within [-5, 5] but cover space
        # Center: 0
        init_points = [
            np.array([4.0] * n),
            np.array([-4.0] * n),
            np.array([4.0 if i % 2 == 0 else -4.0 for i in range(n)]),
            np.array([-4.0 if i % 2 == 0 else 4.0 for i in range(n)]),
            np.zeros(n)
        ]

        # Simplex structure: list of (points, values)
        # Each simplex has n+1 points
        simplexes = []
        
        # Alpha, Gamma, Rho parameters for Nelder-Mead
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 2.0 # Expansion factor (sometimes used, but standard NM uses reflection first)
        
        # We will run a loop where we pick the simplex with the worst best-point and step it
        # This is a form of asynchronous parallel optimization.
        
        # Initialize simplexes
        for p0 in init_points:
            points = [p0.copy()]
            values = [do_call(p0)]
            
            # Generate n more points by perturbing one dimension
            for i in range(n):
                p_new = p0.copy()
                # Perturb in direction of coordinate i, magnitude 1.0
                # Ensure we stay in bounds
                step = 1.0
                if p_new[i] + step > 5.0:
                    step = -step
                if p_new[i] + step < -5.0:
                    step = -step
                p_new[i] += step
                p_new = np.clip(p_new, -5.0, 5.0)
                
                val = do_call(p_new)
                points.append(p_new)
                values.append(val)
                
                if calls_made >= total_budget or not time_ok():
                    break
            
            if len(points) == n + 1:
                simplexes.append({'points': points, 'values': values})
            else:
                # If we ran out of budget during init, just add what we have? 
                # But NM requires n+1. If we can't complete, we might just stop.
                # Given budget 500 and 5*(1+5)=30 calls for init, we are safe.
                pass

        if not simplexes:
            return best_x, best_f

        # Main Loop
        while calls_made < total_budget and time_ok():
            # Find the simplex with the worst (highest) best value
            # We want to improve the worst one to balance exploration/exploitation
            worst_idx = 0
            worst_best_val = np.inf
            for i, s in enumerate(simplexes):
                s_best = min(s['values'])
                if s_best > worst_best_val:
                    worst_best_val = s_best
                    worst_idx = i
            
            s = simplexes[worst_idx]
            points = s['points']
            values = s['values']
            
            # Sort points by value
            sorted_indices = np.argsort(values)
            points = [points[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            
            x0 = points[0]  # Best
            xw = points[-1] # Worst
            fw = values[-1]
            
            # Centroid of all but worst
            centroid = np.mean(points[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - xw)
            xr = np.clip(xr, -5.0, 5.0)
            fr = do_call(xr)
            
            if fr < values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - xw)
                xe = np.clip(xe, -5.0, 5.0)
                fe = do_call(xe)
                
                if fe < fr:
                    points[-1] = xe
                    values[-1] = fe
                else:
                    points[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                # Reflection is good enough
                points[-1] = xr
                values[-1] = fr
            else:
                # Contraction
                # Try inside contraction first
                xc = centroid + rho * (xw - centroid)
                xc = np.clip(xc, -5.0, 5.0)
                fc = do_call(xc)
                
                if fc < fw:
                    points[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    # Move all points towards x0
                    for i in range(1, n + 1):
                        points[i] = x0 + 0.5 * (points[i] - x0)
                        points[i] = np.clip(points[i], -5.0, 5.0)
                        values[i] = do_call(points[i])
                        
                        if calls_made >= total_budget or not time_ok():
                            break
                    if calls_made >= total_budget or not time_ok():
                        break

            # Update simplex in list
            simplexes[worst_idx] = {'points': points, 'values': values}
            
            # Optional: If a simplex gets very small, restart it?
            # For now, standard NM should handle it.
            # But if we detect stagnation (no improvement in values for many steps), 
            # we could re-initialize the worst simplex.
            # Given the "ensemble" nature, the worst simplex is the one being worked on.
            # If it's stuck, it stays stuck until we pick it again.
            # To improve, we can track iterations per simplex.
            
        return best_x, best_f
