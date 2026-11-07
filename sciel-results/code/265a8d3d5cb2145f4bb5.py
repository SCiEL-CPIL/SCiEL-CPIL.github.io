"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive Multi-Start Nelder-Mead with annealing-based restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget_left, rng):
        """Run Nelder-Mead from x0, using at most budget_left evaluations."""
        dim = self.dim
        calls_made = 0
        best_x = x0.copy()
        best_f = func(x0)
        calls_made += 1
        
        # Initialize simplex
        simplex = [x0.copy()]
        f_simplex = [best_f]
        
        # Delta for initial simplex spread - adaptive based on current scale
        # Start with a moderate spread
        delta = 0.5
        for i in range(dim):
            x_new = x0.copy()
            x_new[i] += delta
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls_made += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            simplex.append(x_new)
            f_simplex.append(f_new)
        
        simplex = np.array(simplex)
        f_simplex = np.array(f_simplex)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_iter = budget_left - calls_made
        
        for _ in range(max_iter):
            if calls_made >= budget_left:
                break
            
            # Sort by fitness
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # Convergence check
            if f_simplex[-1] - f_simplex[0] < 1e-12:
                break
            
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflect = x_centroid + alpha * (x_centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, -5, 5)
            f_reflect = func(x_reflect)
            calls_made += 1
            if f_reflect < best_f:
                best_f = f_reflect
                best_x = x_reflect.copy()
            
            if f_simplex[0] <= f_reflect < f_simplex[-2]:
                simplex[-1] = x_reflect
                f_simplex[-1] = f_reflect
            elif f_reflect < f_simplex[0]:
                # Expansion
                x_expand = x_centroid + gamma * (x_reflect - x_centroid)
                x_expand = np.clip(x_expand, -5, 5)
                f_expand = func(x_expand)
                calls_made += 1
                if f_expand < best_f:
                    best_f = f_expand
                    best_x = x_expand.copy()
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    f_simplex[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    f_simplex[-1] = f_reflect
            else:
                # Contraction
                if f_reflect < f_simplex[-1]:
                    x_contract = x_centroid + rho * (x_reflect - x_centroid)
                else:
                    x_contract = x_centroid - rho * (x_reflect - x_centroid)
                x_contract = np.clip(x_contract, -5, 5)
                f_contract = func(x_contract)
                calls_made += 1
                if f_contract < best_f:
                    best_f = f_contract
                    best_x = x_contract.copy()
                
                if f_contract < min(f_reflect, f_simplex[-1]):
                    simplex[-1] = x_contract
                    f_simplex[-1] = f_contract
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex[i] = func(simplex[i])
                        calls_made += 1
                        if f_simplex[i] < best_f:
                            best_f = f_simplex[i]
                            best_x = simplex[i].copy()
                    if calls_made >= budget_left:
                        break
        
        return best_x, best_f, calls_made

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        total_calls = 0
        rng = np.random.default_rng(123)
        
        # Multi-start strategy:
        # 1. First, do a coarse grid-like exploration with a few points
        # 2. Then, for each promising point, run Nelder-Mead
        # 3. Use annealing to decide restart points
        
        # Phase 1: Coarse exploration with 5-10 points
        n_initial = 8
        initial_points = []
        
        # Include center and corners
        initial_points.append(np.zeros(dim))
        
        # Add some random points
        for _ in range(n_initial - 1):
            x = rng.uniform(-4, 4, dim)
            initial_points.append(x)
        
        # Evaluate initial points
        for x in initial_points:
            if total_calls >= budget:
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
        
        # Phase 2: Adaptive Nelder-Mead with restarts
        # Use a "temperature" to decide when to restart
        # Start with the best point found
        
        remaining_budget = budget - total_calls
        
        # Run NM from best point
        if remaining_budget > 10:
            x_start = best_x.copy() if best_x is not None else np.zeros(dim)
            best_x, best_f, nm_calls = self._nelder_mead(func, x_start, remaining_budget, rng)
            total_calls += nm_calls
        
        # Check if we should do more restarts
        # If we have budget left, try a few more random restarts
        # but with smaller NM budgets
        
        n_restarts = 3
        for i in range(n_restarts):
            remaining = budget - total_calls
            if remaining < 20:
                break
            
            # Generate a new random starting point
            # Biased towards the current best region but with some exploration
            if best_x is not None:
                # Mix of local and global
                if rng.random() < 0.5:
                    x_new = best_x + rng.standard_normal(dim) * 1.0
                else:
                    x_new = rng.uniform(-4, 4, dim)
            else:
                x_new = rng.uniform(-4, 4, dim)
            
            x_new = np.clip(x_new, -5, 5)
            
            # Run NM with a portion of remaining budget
            nm_budget = min(remaining - 5, 50)
            if nm_budget < 10:
                break
            
            x_res, f_res, calls_used = self._nelder_mead(func, x_new, nm_budget, rng)
            total_calls += calls_used
            
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()
        
        # Final safety check
        if best_x is None:
            best_x = np.zeros(dim)
            if total_calls < budget:
                best_f = func(best_x)
                total_calls += 1
        
        return best_x, best_f

# EVOLVE-BLOCK-END
