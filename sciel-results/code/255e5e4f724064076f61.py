"""Hybrid PSO with Nelder-Mead Polishing for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: PSO for global search + NM for local polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Track best result
        x_best = np.zeros(self.dim)
        f_best = np.inf
        calls = 0
        max_calls = self.budget

        def safe_func(x):
            nonlocal calls
            if calls >= max_calls:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            try:
                f = func(x)
            except Exception:
                f = np.inf
            calls += 1
            return f

        # Phase 1: Particle Swarm Optimization (PSO)
        # PSO is effective for multimodal landscapes like Rastrigin
        # and provides good early progress for AOCC.
        
        pop_size = 20
        # Allocate 60% of budget to PSO
        budget_pso = int(0.6 * max_calls)
        
        # Initialize particles
        # Random initialization in [-5, 5]
        positions = rng.uniform(-5, 5, size=(pop_size, self.dim))
        velocities = rng.uniform(-1, 1, size=(pop_size, self.dim))
        
        pbest = positions.copy()
        pbest_f = np.full(pop_size, np.inf)
        gbest = np.zeros(self.dim)
        gbest_f = np.inf
        
        # PSO Parameters
        w = 0.7  # Inertia weight
        c1 = 1.5 # Cognitive coefficient
        c2 = 1.5 # Social coefficient
        
        # Run PSO until budget or time limit
        # Each iteration evaluates pop_size new points
        # We need to be careful with the number of iterations
        max_iter_pso = budget_pso // pop_size
        
        for _ in range(max_iter_pso):
            if time.time() > deadline:
                break
            if calls >= max_calls:
                break
                
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                if calls >= max_calls:
                    break
                    
                # Evaluate new position
                # Note: We update position first, then evaluate
                # Standard PSO updates velocity then position
                # Here we evaluate the current position in the loop to simplify budget tracking
                # Actually, standard PSO loop:
                # 1. Update velocity
                # 2. Update position
                # 3. Evaluate position
                # 4. Update pbest/gbest
                
                # Calculate velocity update
                r1 = rng.random()
                r2 = rng.random()
                velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - positions[i]) + c2 * r2 * (gbest - positions[i])
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Clip to bounds
                positions[i] = np.clip(positions[i], -5, 5)
                
                # Evaluate
                f_val = safe_func(positions[i])
                
                if f_val < pbest_f[i]:
                    pbest_f[i] = f_val
                    pbest[i] = positions[i]
                    
                if f_val < gbest_f:
                    gbest_f = f_val
                    gbest = positions[i]
                    
            # Update global bests for tracking
            if gbest_f < f_best:
                f_best = gbest_f
                x_best = gbest.copy()
                
        # Phase 2: Nelder-Mead Polishing
        # Use the remaining budget to polish the best candidates found by PSO
        # We select the top K distinct best points from pbest
        
        if calls < max_calls and time.time() < deadline:
            # Get indices of best pbests
            # Sort by pbest_f
            sorted_idx = np.argsort(pbest_f)
            
            # Select top K distinct points
            # "Distinct" means sufficiently far apart to avoid redundant NM runs
            selected_starts = []
            selected_f = []
            min_dist = 0.5 # Threshold for distinctness
            
            for idx in sorted_idx:
                if len(selected_starts) >= 5: # Max 5 NM runs
                    break
                if pbest_f[idx] == np.inf:
                    continue
                    
                # Check distance to already selected starts
                is_distinct = True
                for s in selected_starts:
                    if np.linalg.norm(pbest[idx] - s) < min_dist:
                        is_distinct = False
                        break
                        
                if is_distinct:
                    selected_starts.append(pbest[idx].copy())
                    selected_f.append(pbest_f[idx])
            
            # If no starts selected, use gbest
            if not selected_starts:
                selected_starts.append(gbest.copy())
                selected_f.append(gbest_f)
            
            # Distribute remaining budget among NM runs
            remaining_budget = max_calls - calls
            if remaining_budget > 0:
                budget_per_nm = remaining_budget // len(selected_starts)
                
                for start_x in selected_starts:
                    if calls >= max_calls:
                        break
                    if time.time() > deadline:
                        break
                        
                    try:
                        res = minimize(
                            safe_func, 
                            start_x, 
                            method='Nelder-Mead',
                            options={'maxiter': budget_per_nm, 'xatol': 1e-8, 'fatol': 1e-8}
                        )
                        if res.fun < f_best:
                            f_best = res.fun
                            x_best = res.x
                    except Exception:
                        pass
                        
        return x_best, f_best
# EVOLVE-BLOCK-END
