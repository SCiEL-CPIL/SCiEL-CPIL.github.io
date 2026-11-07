"""Rastrigin-aware Nelder-Mead with multi-start for rotated/shifted f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with Rastrigin-aware multi-start initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return float('inf')
            if time.time() > deadline - 2:
                return float('inf')
            x = np.clip(x, -5, 5)
            val = func(x)
            calls_made += 1
            return val

        best_x = None
        best_fx = float('inf')

        # --- Phase 0: Multi-start Nelder-Mead ---
        # Generate diverse starting points exploiting Rastrigin structure
        # Minima are at integer points in the rotated/shifted frame
        
        start_points = []
        
        # Center
        start_points.append(np.zeros(dim))
        
        # Integer grid: sample strategically
        # For each dimension, try integer values with others at 0
        for d in range(dim):
            for v in range(-4, 5):
                pt = np.zeros(dim)
                pt[d] = v
                start_points.append(pt)
        
        # Random integer points
        for _ in range(40):
            pt = rng.integers(-4, 5, dim).astype(float)
            start_points.append(pt)
        
        # Random continuous points
        for _ in range(20):
            pt = rng.uniform(-4, 4, dim)
            start_points.append(pt)
        
        # Deduplicate
        seen = set()
        unique_starts = []
        for p in start_points:
            key = tuple(np.round(p, 3))
            if key not in seen:
                seen.add(key)
                unique_starts.append(p)
        
        # Allocate budget: ~200 calls per start, up to 2 starts
        # But adaptively: give more budget to promising starts
        
        nm_budget_per_start = 250
        max_starts = min(len(unique_starts), 2)
        
        for start_idx in range(max_starts):
            if calls_made >= self.budget:
                break
            if time.time() > deadline - 30:
                break
            
            # Nelder-Mead simplex method
            x0 = np.clip(unique_starts[start_idx], -5, 5)
            fx0 = safe_func(x0)
            
            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()
            
            # Build initial simplex
            simplex = [x0.copy()]
            fvals = [fx0]
            
            # Expand along each axis
            for d in range(dim):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline - 20:
                    break
                
                # Adaptive step size based on function landscape
                # For Rastrigin, period is 1.0, so use 0.5 as initial step
                step = 0.5
                for trial_step in [0.5, 0.3, 0.2]:
                    x_trial = x0.copy()
                    x_trial[d] += trial_step
                    x_trial = np.clip(x_trial, -5, 5)
                    ft = safe_func(x_trial)
                    if ft < float('inf'):
                        step = trial_step
                        break
                
                x_trial = x0.copy()
                x_trial[d] += step
                x_trial = np.clip(x_trial, -5, 5)
                ft = safe_func(x_trial)
                simplex.append(x_trial.copy())
                fvals.append(ft)
                
                if ft < best_fx:
                    best_fx = ft
                    best_x = x_trial.copy()
            
            if len(simplex) < dim + 1:
                continue
            
            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink
            
            max_nm_iter = (self.budget - calls_made) // (dim + 1)
            max_nm_iter = min(max_nm_iter, 50)
            
            for it in range(max_nm_iter):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline - 10:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = safe_func(x_reflect)
                
                if f_reflect < best_fx:
                    best_fx = f_reflect
                    best_x = x_reflect.copy()
                
                if f_reflect < fvals[0]:
                    # Expand
                    x_expand = centroid + gamma * (centroid - simplex[-1])
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = safe_func(x_expand)
                    
                    if f_expand < best_fx:
                        best_fx = f_expand
                        best_x = x_expand.copy()
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    # Contract
                    if f_reflect < fvals[-1]:
                        # Outside contraction
                        x_contract = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        x_contract = centroid - rho * (simplex[-1] - centroid)
                    
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = safe_func(x_contract)
                    
                    if f_contract < best_fx:
                        best_fx = f_contract
                        best_x = x_contract.copy()
                    
                    if f_contract < min(f_reflect, fvals[-1]):
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = safe_func(simplex[i])
                        
                        if fvals[1] < best_fx:
                            best_fx = fvals[1]
                            best_x = simplex[1].copy()
                
                # Check convergence
                if len(fvals) > 0:
                    f_range = max(fvals) - min(fvals)
                    if f_range < 1e-8:
                        break
        
        # --- Phase 1: Final fine-tuning with small steps ---
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5:
            x_refine = best_x.copy()
            fx_refine = best_fx
            
            # Try small perturbations in random directions
            step_sizes = [0.05, 0.02, 0.01, 0.005]
            for ss in step_sizes:
                if calls_made >= self.budget:
                    break
                if time.time() > deadline - 3:
                    break
                
                improved = False
                for _ in range(10):
                    if calls_made >= self.budget:
                        break
                    if time.time() > deadline - 3:
                        break
                    
                    direction = rng.standard_normal(dim)
                    direction /= np.linalg.norm(direction)
                    
                    x_trial = x_refine + ss * direction
                    x_trial = np.clip(x_trial, -5, 5)
                    ft = safe_func(x_trial)
                    
                    if ft < fx_refine:
                        x_refine = x_trial
                        fx_refine = ft
                        improved = True
                        if ft < best_fx:
                            best_fx = ft
                            best_x = x_refine.copy()
                
                if not improved:
                    break
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = safe_func(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
