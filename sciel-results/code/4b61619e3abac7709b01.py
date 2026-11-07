import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        bounds = [(-5.0, 5.0)] * self.dim
        lb = np.full(self.dim, -5.0)
        ub = np.full(self.dim, 5.0)
        d = self.dim

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lb, ub)
            
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
                
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy: Adaptive global sampling + priority-based local refinement with restarts
        
        # Phase 1: Global Sampling - more samples for better basin coverage
        num_global = 50
        candidates = []
        for i in range(num_global):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            candidates.append((f, x))
        
        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Priority-based local refinement with adaptive restarts
        # Maintain a list of start points to try, prioritizing diverse and promising regions
        
        # Create a priority queue of start points
        # Each entry: (priority_score, x_start, refined_flag)
        # Priority: combine fitness and distance to already-refined points
        
        refined_points = []  # List of (x, f) for points that have been locally refined
        start_queue = []
        
        # Add initial candidates to the queue
        for f, x in candidates:
            start_queue.append((f, x, False))
        
        # Sort by fitness
        start_queue.sort(key=lambda item: item[0])
        
        max_local_runs = 15
        run_count = 0
        prev_best_f = best_f
        
        while run_count < max_local_runs and calls_made < self.budget and time.time() < deadline:
            # Find the best unrefined start point that is sufficiently different from refined points
            x_start = None
            f_start = None
            
            for i, (f, x, refined) in enumerate(start_queue):
                if refined:
                    continue
                
                # Check distance to all refined points
                min_dist = np.inf
                for rx, rf in refined_points:
                    dist = np.linalg.norm(x - rx)
                    min_dist = min(min_dist, dist)
                
                # If sufficiently far from refined points, or if it's the best candidate
                if min_dist > 1.0 or len(refined_points) == 0:
                    x_start = x
                    f_start = f
                    start_queue[i] = (f, x, True)
                    break
            else:
                # All points are too close to refined points; pick the best unrefined
                for i, (f, x, refined) in enumerate(start_queue):
                    if not refined:
                        x_start = x
                        f_start = f
                        start_queue[i] = (f, x, True)
                        break
            
            if x_start is None:
                # No more start points; generate a new random one
                x_start = rng.uniform(-5.0, 5.0, d)
                f_start = call_func(x_start)
                if f_start < np.inf:
                    start_queue.append((f_start, x_start, True))
                    start_queue.sort(key=lambda item: item[0])
            
            if x_start is None:
                break
            
            # Determine maxfev for this local run
            remaining = self.budget - calls_made
            runs_left = max(1, max_local_runs - run_count)
            maxfev = max(10, min(60, remaining // runs_left))
            
            # Run L-BFGS-B first (good for smooth)
            res_x = x_start.copy()
            res_f = f_start
            try:
                res = minimize(
                    call_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxfev': maxfev,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
                if res.success or res.fun < res_f:
                    res_x = res.x
                    res_f = res.fun
            except Exception:
                pass
            
            # If we still have budget and time, polish with Nelder-Mead
            if calls_made < self.budget and time.time() < deadline:
                nm_maxfev = max(5, min(25, (self.budget - calls_made) // max(1, runs_left)))
                try:
                    res_nm = minimize(
                        call_func,
                        res_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': nm_maxfev,
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                    if res_nm.fun < res_f:
                        res_x = res_nm.x
                        res_f = res_nm.fun
                except Exception:
                    pass
            
            # Record the refined point
            if res_f < np.inf:
                refined_points.append((res_x.copy(), res_f))
            
            # Adaptive restart: if this run significantly improved the best,
            # add perturbed versions of the new best to explore nearby regions
            if best_f < prev_best_f - 1e-6 and len(refined_points) <= 8:
                # Generate 2-3 perturbed points around the new best
                n_perturb = min(3, max_local_runs - run_count - 1)
                for _ in range(n_perturb):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    # Perturb with decreasing scale
                    scale = 0.5 + 0.3 * rng.random()
                    x_perturb = best_x + scale * rng.standard_normal(d)
                    x_perturb = np.clip(x_perturb, lb, ub)
                    f_perturb = call_func(x_perturb)
                    if f_perturb < np.inf:
                        start_queue.append((f_perturb, x_perturb, False))
                        start_queue.sort(key=lambda item: item[0])
            
            prev_best_f = best_f
            run_count += 1
        
        # Phase 3: Final intensive polish
        # If we have budget left, do a final high-precision optimization from best_x
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            if remaining > 20:
                # First: L-BFGS-B with very tight tolerances
                final_maxfev = min(remaining // 2, 40)
                try:
                    res_final = minimize(
                        call_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxfev': final_maxfev,
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
                except Exception:
                    pass
                
                # Second: Nelder-Mead for final fine-tuning
                if calls_made < self.budget and time.time() < deadline:
                    nm_final_maxfev = min(self.budget - calls_made, 30)
                    if nm_final_maxfev > 5:
                        try:
                            res_nm_final = minimize(
                                call_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxfev': nm_final_maxfev,
                                    'xatol': 1e-10,
                                    'fatol': 1e-10
                                }
                            )
                        except Exception:
                            pass

        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
