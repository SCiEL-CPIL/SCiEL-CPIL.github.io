"""Model-and-optimize approach for WFG4: exploit separable structure."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        mid = 0.5 * (lo + hi)

        eval_count = 0
        def time_left():
            return deadline - time.time()

        def check_deadline():
            if time.time() > deadline:
                raise TimeoutError()

        # ---------- PHASE 1: IDENTIFY FRONT VARIABLE ----------
        # Probe each variable to see its impact on the objectives.
        # We use the midpoint as a base.
        base_x = mid.copy()
        base_f = np.asarray(problem(base_x), dtype=float)
        eval_count += 1
        
        impacts = np.zeros(dim)
        for i in range(dim):
            # Move variable i to high
            x_hi = mid.copy()
            x_hi[i] = hi[i]
            f_hi = np.asarray(problem(x_hi), dtype=float)
            eval_count += 1
            
            # Move variable i to low
            x_lo = mid.copy()
            x_lo[i] = lo[i]
            f_lo = np.asarray(problem(x_lo), dtype=float)
            eval_count += 1
            
            # Impact is the max change in any objective
            impact = max(np.max(np.abs(f_hi - base_f)), np.max(np.abs(f_lo - base_f)))
            impacts[i] = impact

        front_var = int(np.argmax(impacts))
        
        # ---------- PHASE 2: OPTIMIZE PENALTY VARIABLES ----------
        # For WFG4, the "penalty" variables usually have a unique minimum (often at 0.5)
        # that minimizes the distance to the ideal point or reduces the "noise".
        # We will optimize each penalty variable independently using a small search.
        
        # Start with mid as the initial guess for penalty vars
        x_opt = mid.copy()
        
        # For each non-front variable, perform a local search to minimize the sum of objectives
        # (or distance to ideal, but sum is simpler and robust for WFG4)
        # We use a coarse-to-fine 1D search.
        
        for i in range(dim):
            if i == front_var:
                continue
                
            if eval_count >= self.budget - 50: # Reserve budget for front sweep
                break
            if time_left() < 10.0:
                break
                
            # Define search range for this variable
            # We assume the optimal value is within [lo, hi].
            # WFG4 penalties are often minimized at 0.5 (mid).
            
            # Coarse search: 5 points
            coarse_pts = np.linspace(lo[i], hi[i], 5)
            best_val = x_opt[i]
            best_f = np.asarray(problem(x_opt), dtype=float)
            best_sum = np.sum(best_f)
            
            for v in coarse_pts:
                if eval_count >= self.budget:
                    break
                x_trial = x_opt.copy()
                x_trial[i] = v
                f_trial = np.asarray(problem(x_trial), dtype=float)
                eval_count += 1
                if np.sum(f_trial) < best_sum:
                    best_sum = np.sum(f_trial)
                    best_val = v
                    best_f = f_trial
            
            # Fine search around the best coarse point
            if best_val != x_opt[i]:
                # Narrow range around best_val
                delta = (hi[i] - lo[i]) / 20.0
                lower = max(lo[i], best_val - delta)
                upper = min(hi[i], best_val + delta)
                fine_pts = np.linspace(lower, upper, 5)
                
                for v in fine_pts:
                    if eval_count >= self.budget:
                        break
                    x_trial = x_opt.copy()
                    x_trial[i] = v
                    f_trial = np.asarray(problem(x_trial), dtype=float)
                    eval_count += 1
                    if np.sum(f_trial) < best_sum:
                        best_sum = np.sum(f_trial)
                        best_val = v
                        best_f = f_trial
                
                x_opt[i] = best_val
                # Update x_opt's f? We don't strictly need it, but let's keep it consistent for next iterations
                # Actually, we should re-evaluate x_opt if we want to be precise, but budget is tight.
                # We'll just store the best value found.
        
        # ---------- PHASE 3: DENSE SWEEP OF FRONT VARIABLE ----------
        # Now that penalty variables are optimized, sweep the front variable.
        remaining = self.budget - eval_count
        n_dense = max(10, min(100, int(remaining * 0.8)))
        
        # Ensure we have enough points to cover the front
        if n_dense < 20:
            n_dense = 20
            
        # If we ran out of budget, reduce n_dense
        if n_dense > remaining:
            n_dense = max(10, remaining)

        front_ts = np.linspace(0.0, 1.0, n_dense)
        
        # We will collect all points
        all_points = []
        all_Fs = []
        
        for t in front_ts:
            if eval_count >= self.budget:
                break
            if time_left() < 2.0:
                break
                
            x = x_opt.copy()
            x[front_var] = lo[front_var] + t * span[front_var]
            
            f = np.asarray(problem(x), dtype=float)
            eval_count += 1
            
            all_points.append(x)
            all_Fs.append(f)
            
        # If we have extra budget, we can do a second pass with slightly different penalty settings
        # or just refine the front sweep further.
        if eval_count < self.budget - 10 and time_left() > 5.0:
            # Try to improve the penalty variables again with a finer search?
            # Or just do more front points.
            extra_remaining = self.budget - eval_count
            extra_n = min(50, extra_remaining)
            if extra_n > 0:
                # Interpolate between existing points or just sample more
                # Let's just sample more points uniformly
                extra_ts = np.linspace(0.0, 1.0, extra_n)
                for t in extra_ts:
                    if eval_count >= self.budget:
                        break
                    if time_left() < 1.0:
                        break
                    x = x_opt.copy()
                    x[front_var] = lo[front_var] + t * span[front_var]
                    f = np.asarray(problem(x), dtype=float)
                    eval_count += 1
                    all_points.append(x)
                    all_Fs.append(f)

        # Return value is ignored, but we must return something
        if all_points:
            return np.array(all_points)
        else:
            return np.array([mid])
