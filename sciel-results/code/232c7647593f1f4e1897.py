"""Rewritten optimizer for BBOB f20 (Schwefel) using (1+1)-ES with global restarts."""
import numpy as np
import time


class Optimizer:
    """(1+1) Evolution Strategy with global Sobol restarts for multimodal search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(fx):
                fx = np.inf
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global coverage using stratified sampling
        # Use a moderate number of global points to find promising basins
        n_global = 60
        
        # Generate stratified samples
        rng = np.random.default_rng(12345)
        n_per_dim = max(2, int(round(n_global ** (1.0 / dim))))
        
        from itertools import product
        grid_indices = list(product(range(n_per_dim), repeat=dim))
        rng.shuffle(grid_indices)
        
        global_points = []
        for idx in grid_indices[:n_global]:
            point = np.zeros(dim)
            for d in range(dim):
                lo = -5.0 + 10.0 * idx[d] / n_per_dim
                hi = -5.0 + 10.0 * (idx[d] + 1) / n_per_dim
                point[d] = rng.uniform(lo, hi)
            global_points.append(point)
        
        # Evaluate global points
        for x in global_points:
            if time.time() > deadline or call_count >= budget:
                break
            safe_call(x)

        # Phase 2: (1+1)-ES with restarts
        # Strategy: Run (1+1)-ES from the best point found, with periodic restarts
        # if stagnation is detected. Each step gives a best-value update for AOCC.
        
        remaining_budget = budget - call_count
        if remaining_budget <= 0:
            return best_x, best_f

        # Parameters for (1+1)-ES
        # Start with moderate sigma for balance between exploration and exploitation
        sigma = 0.5  # Smaller initial step for faster local convergence
        sigma_min = 1e-10
        sigma_max = 5.0
        
        # Adaptation parameters - faster adaptation
        sigma_decay = 0.98  # Faster decay when no improvement
        sigma_growth = 1.05  # More aggressive growth when improvement occurs
        
        # Restart parameters
        max_stagnant = 20  # Restart after this many stagnant steps
        stagnant_count = 0
        current_best_f = best_f
        current_x = best_x.copy()
        
        # Local search phase
        while call_count < budget and time.time() < deadline:
            # Generate offspring
            step = sigma * rng.standard_normal(dim)
            candidate = current_x + step
            candidate = np.clip(candidate, -5.0, 5.0)
            
            fx = safe_call(candidate)
            
            if not np.isfinite(fx):
                # Failed evaluation, shrink sigma
                sigma = max(sigma_min, sigma * 0.9)
                stagnant_count += 1
                continue
            
            if fx < current_best_f:
                # Improvement: move to new point, increase sigma slightly
                current_x = candidate.copy()
                current_best_f = fx
                stagnant_count = 0
                sigma = min(sigma_max, sigma * sigma_growth)
            else:
                # No improvement: keep current point, shrink sigma
                sigma = max(sigma_min, sigma * sigma_decay)
                stagnant_count += 1
            
            # Check for stagnation and restart
            if stagnant_count >= max_stagnant:
                # Try a few random restarts from different regions
                restarts_found = False
                for _ in range(8):
                    if call_count >= budget or time.time() > deadline:
                        break
                    # Random point in the box
                    restart_x = rng.uniform(-5.0, 5.0, size=dim)
                    restart_f = safe_call(restart_x)
                    if np.isfinite(restart_f) and restart_f < current_best_f:
                        current_x = restart_x.copy()
                        current_best_f = restart_f
                        stagnant_count = 0
                        sigma = 1.0  # Reset sigma to moderate value
                        restarts_found = True
                        break
                if not restarts_found:
                    # If no better restart found, try a smaller sigma and continue
                    sigma = max(sigma_min, sigma * 0.3)
                    stagnant_count = 0
            
            # If sigma is very small, try a moderate local restart
            if sigma < 0.005 and stagnant_count > 15:
                for _ in range(5):
                    if call_count >= budget or time.time() > deadline:
                        break
                    restart_x = current_x + rng.uniform(-0.5, 0.5, size=dim)
                    restart_x = np.clip(restart_x, -5.0, 5.0)
                    restart_f = safe_call(restart_x)
                    if np.isfinite(restart_f) and restart_f < current_best_f:
                        current_x = restart_x.copy()
                        current_best_f = restart_f
                        stagnant_count = 0
                        sigma = 0.3
                        break
        
        # Final polish phase: use remaining budget for very fine-tuning
        if call_count < budget and time.time() < deadline:
            polish_sigma = max(1e-6, sigma * 0.1)
            polish_count = 0
            max_polish = min(50, budget - call_count)
            
            while polish_count < max_polish and call_count < budget and time.time() < deadline:
                step = polish_sigma * rng.standard_normal(dim)
                candidate = current_x + step
                candidate = np.clip(candidate, -5.0, 5.0)
                
                fx = safe_call(candidate)
                
                if np.isfinite(fx) and fx < current_best_f:
                    current_x = candidate.copy()
                    current_best_f = fx
                    polish_sigma *= 0.8
                else:
                    polish_sigma *= 0.9
                    if polish_sigma < 1e-12:
                        break
                polish_count += 1

        # Final: ensure we have the global best
        if current_best_f < best_f:
            best_f = current_best_f
            best_x = current_x.copy()

        return best_x, best_f
