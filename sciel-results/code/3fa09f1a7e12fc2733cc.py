"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin).
Strategy: Derivative-Free Coordinate Descent (DFCD) with multi-resolution grid
scanning per dimension to handle the bi-modal nature of Lunacek's function.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin for 300s limit
        d = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracker
        best_x = np.zeros(d)
        best_f = np.inf
        calls_made = 0

        def _time_ok():
            return (time.time() - start_time) < deadline

        def _eval(x):
            nonlocal calls_made, best_f, best_x
            calls_made += 1
            x_clip = np.clip(x, -5.0, 5.0)
            fx = func(x_clip)
            if fx < best_f:
                best_f = fx
                best_x = x_clip.copy()
            return fx

        # Multi-resolution grid scan for a single dimension
        def _grid_scan_dim(x, dim_idx, n_points, span):
            """Scan one dimension with a grid of n_points over span around x[dim_idx]."""
            nonlocal best_x, best_f, calls_made
            center = x[dim_idx]
            lo = max(-5.0, center - span / 2)
            hi = min(5.0, center + span / 2)
            if lo >= hi:
                return x, False
            
            grid = np.linspace(lo, hi, n_points)
            best_val = np.inf
            best_pt = None
            
            for val in grid:
                if not _time_ok() or calls_made >= total_budget:
                    break
                trial = x.copy()
                trial[dim_idx] = val
                f_val = _eval(trial)
                if f_val < best_val:
                    best_val = f_val
                    best_pt = val
            
            improved = False
            if best_pt is not None and best_pt != x[dim_idx]:
                x[dim_idx] = best_pt
                improved = True
            return x, improved

        # Coordinate descent pass: sweep all dimensions with a grid
        def _coord_descent_pass(x, n_points, span):
            nonlocal best_x, best_f, calls_made
            improved_any = False
            for dim_idx in range(d):
                if not _time_ok() or calls_made >= total_budget:
                    break
                x, improved = _grid_scan_dim(x, dim_idx, n_points, span)
                if improved:
                    improved_any = True
            return x, improved_any

        # Phase 1: Coarse global exploration
        # Start from multiple diverse initial points to handle bi-modality
        initial_points = [
            np.zeros(d),
            rng.uniform(-4, 4, d),
            rng.uniform(-3, 3, d),
            np.full(d, 2.0),
            np.full(d, -2.0),
        ]
        
        # Allocate budget: ~40% for coarse exploration, ~40% for medium refinement, ~20% for fine polish
        coarse_budget = int(total_budget * 0.4)
        medium_budget = int(total_budget * 0.4)
        fine_budget = total_budget - coarse_budget - medium_budget
        
        # Coarse phase: large spans, fewer points per dimension
        coarse_n_points = 7
        coarse_span = 8.0  # Covers most of [-5,5]
        
        for x0 in initial_points:
            if calls_made >= coarse_budget or not _time_ok():
                break
            x = x0.copy()
            _eval(x)  # Initial evaluation
            
            # Do a few coordinate descent passes with coarse grid
            for _ in range(3):
                if calls_made >= coarse_budget or not _time_ok():
                    break
                x, improved = _coord_descent_pass(x, coarse_n_points, coarse_span)
                if not improved:
                    break
        
        # Phase 2: Medium refinement
        medium_n_points = 11
        medium_span = 3.0
        
        if calls_made < total_budget and _time_ok():
            x = best_x.copy()
            for _ in range(5):
                if calls_made >= total_budget - fine_budget or not _time_ok():
                    break
                x, improved = _coord_descent_pass(x, medium_n_points, medium_span)
                if not improved:
                    break
        
        # Phase 3: Fine polish with adaptive step sizes
        if calls_made < total_budget and _time_ok():
            x = best_x.copy()
            fx = best_f
            step = 0.5
            
            # Adaptive (1+1)-ES style polish with coordinate-wise moves
            for _ in range(total_budget - calls_made):
                if not _time_ok():
                    break
                
                # Try small random perturbations in each dimension
                improved = False
                for dim_idx in range(d):
                    if not _time_ok() or calls_made >= total_budget:
                        break
                    
                    # Try moving in + and - directions
                    for direction in [1, -1]:
                        if not _time_ok() or calls_made >= total_budget:
                            break
                        trial = x.copy()
                        trial[dim_idx] += direction * step
                        trial = np.clip(trial, -5, 5)
                        f_val = _eval(trial)
                        if f_val < fx:
                            x = trial
                            fx = f_val
                            improved = True
                            break  # Move to next dimension
                    if improved:
                        step *= 1.05  # Slight increase on success
                        break
                else:
                    step *= 0.85  # Decrease on no improvement
                
                step = max(step, 0.001)
                step = min(step, 1.0)
        
        return best_x, best_f
