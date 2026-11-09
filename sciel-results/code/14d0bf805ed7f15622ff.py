"""Optimized Katsuura (f23) minimizer with adaptive local search and multi-restart strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB-style Katsuura function (f23).
    
    Strategy:
    1. Smart initial sampling: Mix of origin, local Gaussian, and global uniform points.
    2. Tiered local search:
       - Phase 1: Quick Nelder-Mead on top candidates to identify promising basins.
       - Phase 2: Deeper search on the best basin found.
       - Phase 3: Final high-precision polish on the global best.
    3. Time and budget guards to ensure valid results within constraints.
    
    The Katsuura function is highly oscillatory with many local minima.
    The global minimum is at the origin (0, 0, ..., 0) in the unshifted case,
    but the instances apply hidden shifts and rotations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0  # Safety margin before 330s kill
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')
        
        # Use a fixed seed for reproducibility within a single instance
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            v = func(x_clipped)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x_clipped.copy()
            return v

        # 1. Initial Diverse Sampling
        # The Katsuura function has a global minimum at the origin in the standard formulation.
        # With hidden shifts/rotations, the minimum is somewhere in [-4, 4]^5.
        # We use a mix of strategies to cover the space effectively.
        
        init_points = []
        
        # Priority 1: The origin (likely close to the global min even with shifts)
        init_points.append(np.zeros(dim))
        
        # Priority 2: Local Gaussian sampling around the origin
        # Katsuura is oscillatory, so local minima are frequent.
        # 15 points with sigma=0.3 (very local)
        for _ in range(15):
            p = rng.normal(0, 0.3, dim)
            init_points.append(np.clip(p, -5, 5))
            
        # 10 points with sigma=1.0 (medium local)
        for _ in range(10):
            p = rng.normal(0, 1.0, dim)
            init_points.append(np.clip(p, -5, 5))
            
        # Priority 3: Uniform sampling for global exploration
        # 15 points spread across the box
        for _ in range(15):
            p = rng.uniform(-5, 5, dim)
            init_points.append(p)
        
        # Evaluate initial points
        init_values = []
        for p in init_points:
            v = wrapped_func(p)
            init_values.append(v)
        
        # Sort indices by function value (ascending)
        sorted_indices = np.argsort(init_values)
        
        # 2. Tiered Local Search
        # We prioritize deep search on the best candidates first for AOCC (early progress).
        
        # Tier 1: Deep search on the top 3 candidates
        top_k_tier1 = 3
        if calls < self.budget - 80:
            for i in range(min(top_k_tier1, len(sorted_indices))):
                seed_idx = sorted_indices[i]
                if time.time() - start_time > time_limit - 10.0:
                    break
                seed = init_points[seed_idx].copy()
                try:
                    # Medium-Long Nelder-Mead to converge in the basin
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 60, 
                                       'xatol': 1e-6, 
                                       'fatol': 1e-8, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Tier 2: Medium search on the next 5 candidates (indices 3-7)
        top_k_tier2 = 5
        if calls < self.budget - 50:
            for i in range(top_k_tier1, min(top_k_tier1 + top_k_tier2, len(sorted_indices))):
                seed_idx = sorted_indices[i]
                if time.time() - start_time > time_limit - 10.0:
                    break
                seed = init_points[seed_idx].copy()
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 25, 
                                       'xatol': 1e-5, 
                                       'fatol': 1e-7, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Tier 3: Final high-precision polish on the global best
        if calls < self.budget - 5 and best_x is not None:
            try:
                # Very tight tolerances for final convergence
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={
                                   'maxiter': 150, 
                                   'xatol': 1e-10, 
                                   'fatol': 1e-12, 
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        # Additional restarts if budget remains
        # Try a few more random restarts with short local search
        if calls < self.budget - 20 and best_x is not None:
            for _ in range(3):
                if time.time() - start_time > time_limit - 10.0:
                    break
                if calls >= self.budget - 20:
                    break
                # Restart near the current best with some perturbation
                perturbation = rng.normal(0, 0.2, dim)
                seed = np.clip(best_x + perturbation, -5, 5)
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 15, 
                                       'xatol': 1e-5, 
                                       'fatol': 1e-7, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
