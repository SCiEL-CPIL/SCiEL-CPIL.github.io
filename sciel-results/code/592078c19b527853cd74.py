import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        eval_history = []

        def safe_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float).ravel(), -5.0, 5.0)
            f = func(x)
            calls_made += 1
            if np.isfinite(f):
                eval_history.append((f, x.copy()))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            return f if np.isfinite(f) else np.inf

        # Phase 1: Structure-exploiting grid search for Rastrigin
        # Rastrigin: f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The global minimum of the unshifted function is at x=0 with f=0.
        # With shift s and rotation R, the minimum is at R^{-1}(-s).
        # The function is periodic with period 2 in each dimension (before rotation).
        # Key insight: the local minima are at integer coordinates (after undoing shift/rotation).
        # Since we don't know the shift/rotation, we do a coarse grid to find the basin.

        # Coarse grid: sample at intervals that align with the period structure
        # For Rastrigin, local minima are at integer points. The global minimum is at one specific integer point.
        # A grid with spacing ~0.5 in each dimension would capture all local minima basins.
        # But that's 11^5 = 161051 points, too many.
        # Instead, use a hierarchical approach:
        # 1. Very coarse grid (spacing 1.0) to find which integer lattice point is best
        # 2. Refine around the best few points

        n_explore = int(budget * 0.6)

        # Coarse grid: spacing of 1.0 from -5 to 5, giving 11 points per dim = 11^5 = 161051
        # That's too many. Let's be smarter.
        # Use a Halton-like or Sobol-like sequence but biased toward integer coordinates.
        
        # Actually, let's use a two-level grid:
        # Level 1: coarse grid with spacing 2.0 (period of cos(2*pi*x) is 2, but local minima at integers)
        # Spacing 2.0: from -4 to 4, step 2: -4, -2, 0, 2, 4 -> 5 points per dim = 5^5 = 3125 points
        # Level 2: around the best few coarse points, refine with spacing 0.5

        # Generate coarse grid points
        coarse_coords = np.arange(-4.0, 5.0, 2.0)  # -4, -2, 0, 2, 4
        # Also add -5 and 5 boundaries
        coarse_coords = np.concatenate([[-5.0], coarse_coords, [5.0]])
        # That's -5, -4, -2, 0, 2, 4, 5 -> 7 points per dim = 7^5 = 16807, still a lot
        
        # Let's use just -4, -2, 0, 2, 4 -> 5^5 = 3125
        coarse_coords = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        
        # Generate all combinations
        from itertools import product
        coarse_points = np.array(list(product(coarse_coords, repeat=dim)))
        
        # Shuffle to explore broadly
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(coarse_points))
        coarse_points = coarse_points[perm]
        
        # Evaluate coarse grid
        for x in coarse_points:
            if calls_made >= n_explore or time.time() > deadline - 60.0:
                break
            safe_call(x)
        
        # Phase 2: Refine around the best coarse points
        if eval_history:
            eval_history.sort(key=lambda item: item[0])
            # Take top 5 best coarse points
            top_k = min(5, len(eval_history))
            best_coarse = [x for f, x in eval_history[:top_k]]
            
            # For each best coarse point, do a local grid refinement
            remaining_budget = budget - calls_made
            budget_per_point = max(20, remaining_budget // max(1, top_k))
            
            for x0 in best_coarse:
                if calls_made >= budget or time.time() > deadline - 30.0:
                    break
                
                # Local grid around x0 with spacing 0.25
                # This gives a 5x5x5x5x5 = 3125 point grid in a 1x1x1x1x1 region
                # But we only have budget_per_point, so we need to be selective
                local_spacing = 0.25
                n_local_per_dim = 5  # 5 points per dim
                local_coords = x0 + np.array([-1, -0.5, 0, 0.5, 1]) * local_spacing
                local_coords = np.clip(local_coords, -5.0, 5.0)
                
                local_points = np.array(list(product(local_coords, repeat=dim)))
                
                # Shuffle
                perm = rng.permutation(len(local_points))
                local_points = local_points[perm]
                
                local_budget = min(budget_per_point, len(local_points))
                for x in local_points[:local_budget]:
                    if calls_made >= budget or time.time() > deadline - 30.0:
                        break
                    safe_call(x)
        
        # Phase 3: Nelder-Mead polish on the best point
        if calls_made < budget and time.time() < deadline - 10.0:
            local_calls = 0
            local_budget = min(100, budget - calls_made)
            
            def nm_func(x):
                nonlocal local_calls
                if local_calls >= local_budget or calls_made >= budget or time.time() > deadline - 5.0:
                    return np.inf
                local_calls += 1
                return safe_call(x)
            
            try:
                res = minimize(
                    nm_func,
                    x0=best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
            
            # One more round of Nelder-Mead from a slightly perturbed best point
            if calls_made < budget and time.time() < deadline - 5.0:
                local_calls = 0
                local_budget = min(50, budget - calls_made)
                
                def nm_func2(x):
                    nonlocal local_calls
                    if local_calls >= local_budget or calls_made >= budget or time.time() > deadline - 5.0:
                        return np.inf
                    local_calls += 1
                    return safe_call(x)
                
                # Perturb best point slightly
                perturbed = best_x + rng.normal(0, 0.1, size=dim)
                perturbed = np.clip(perturbed, -5.0, 5.0)
                
                try:
                    res2 = minimize(
                        nm_func2,
                        x0=perturbed,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
