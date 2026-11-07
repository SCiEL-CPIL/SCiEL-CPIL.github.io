"""Optimizer using pattern search with random restarts for BBOB f18 (Schaffers F7 ill-conditioned)."""
import numpy as np
import time

class Optimizer:
    """Pattern search with binary step-size adaptation and random restarts."""
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)
    
    def _pattern_search(self, func, x_start, budget, rng, deadline):
        """Direct pattern search with orthogonal and diagonal directions.
        
        Uses binary search on step size per direction to guarantee progress.
        Returns (best_x, best_f, calls_used).
        """
        dim = self.dim
        x = self._clip(x_start.copy())
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx
        
        # Search directions: orthogonal + diagonal
        directions = []
        for i in range(dim):
            d_pos = np.zeros(dim)
            d_pos[i] = 1.0
            directions.append(d_pos)
            d_neg = -d_pos.copy()
            directions.append(d_neg)
        # Add diagonal directions
        for i in range(dim):
            for j in range(i + 1, dim):
                d_diag = np.zeros(dim)
                d_diag[i] = 1.0
                d_diag[j] = 1.0
                directions.append(d_diag.copy())
                directions.append(-d_diag.copy())
        
        # Normalize directions
        directions = [d / np.linalg.norm(d) for d in directions]
        
        # Initial step size
        step = 1.0
        min_step = 1e-8
        max_step = 10.0
        
        # Track consecutive failures to adapt step size globally
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while calls < budget:
            if time.time() > deadline - 5.0:
                break
            
            progress_made = False
            
            for direction in directions:
                if calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                # Binary search for best step size in this direction
                lo = min_step
                hi = max_step
                best_step = None
                best_f_dir = fx
                best_x_dir = x.copy()
                
                # Try a few candidate step sizes via binary search
                for _ in range(6):  # ~64 effective steps in binary search
                    mid = (lo + hi) / 2.0
                    x_trial = self._clip(x + mid * direction)
                    fx_trial = func(x_trial)
                    calls += 1
                    
                    if fx_trial < best_f_dir:
                        best_f_dir = fx_trial
                        best_x_dir = x_trial.copy()
                        best_step = mid
                        lo = mid
                    else:
                        hi = mid
                    
                    if hi - lo < 1e-6:
                        break
                
                if best_step is not None and best_f_dir < fx:
                    # Accept the improvement
                    x = best_x_dir.copy()
                    fx = best_f_dir
                    progress_made = True
                    consecutive_failures = 0
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    # Increase step size slightly for next iteration
                    step = min(step * 1.2, max_step)
                    break  # Restart direction loop with new x
                
            if not progress_made:
                consecutive_failures += 1
                # Decrease step size when no progress
                step = max(step * 0.7, min_step)
                
                if consecutive_failures >= max_consecutive_failures:
                    # If stuck, try a larger random jump
                    jump = rng.normal(0, 0.5, size=dim)
                    x_jump = self._clip(x + jump)
                    fx_jump = func(x_jump)
                    calls += 1
                    if fx_jump < fx:
                        x = x_jump.copy()
                        fx = fx_jump
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        consecutive_failures = 0
                    else:
                        # If still stuck, give up on this local search
                        break
        
        return best_x, best_f, calls
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Random sampling to find diverse starting points
        n_initial = min(30, budget // 5)
        initial_points = []
        initial_values = []
        
        # Always include origin and a few structured points
        test_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.array([3.0, -3.0, 3.0, -3.0, 3.0] if dim == 5 else [3.0] * dim),
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0] if dim == 5 else [-3.0] * dim),
        ]
        
        for xp in test_points[:min(len(test_points), n_initial)]:
            if total_calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            xp = self._clip(xp)
            fxp = func(xp)
            total_calls += 1
            initial_points.append(xp.copy())
            initial_values.append(fxp)
            if fxp < best_f:
                best_f = fxp
                best_x = xp.copy()
        
        # Add random points
        for i in range(n_initial - len(test_points[:min(len(test_points), n_initial)])):
            if total_calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            xi = rng.uniform(-4.5, 4.5, size=dim)
            fi = func(xi)
            total_calls += 1
            initial_points.append(xi.copy())
            initial_values.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        # Phase 2: Pattern search from top-k diverse starting points
        # Sort by fitness
        order = np.argsort(initial_values)
        selected_starts = []
        for idx in order:
            if len(selected_starts) >= 5:
                break
            p = initial_points[idx]
            # Check diversity
            too_close = False
            for s in selected_starts:
                if np.linalg.norm(p - s) < 0.5:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append(p.copy())
        
        # Add a few more random diverse points
        while len(selected_starts) < 8 and total_calls < budget - 20:
            x_new = rng.uniform(-4.0, 4.0, size=dim)
            too_close = False
            for s in selected_starts:
                if np.linalg.norm(x_new - s) < 0.8:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append(x_new)
        
        calls_per_run = max(20, (budget - total_calls) // max(1, len(selected_starts)))
        
        for start_idx, x0 in enumerate(selected_starts):
            if total_calls >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            
            remaining = budget - total_calls
            max_calls_this = min(calls_per_run, remaining)
            
            if max_calls_this < 5:
                continue
            
            x_ps, f_ps, calls_ps = self._pattern_search(func, x0, max_calls_this, rng, deadline)
            total_calls += calls_ps
            
            if f_ps < best_f:
                best_f = f_ps
                best_x = x_ps.copy()
        
        return best_x, best_f
