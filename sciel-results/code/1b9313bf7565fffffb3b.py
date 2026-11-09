"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse grid search + Powell's method."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _powell(self, func, x0, budget, deadline, rng):
        """Powell's method with coordinate-wise quadratic interpolation."""
        dim = self.dim
        x = np.array(x0, dtype=float)
        f = func(x)
        budget -= 1
        # Initial directions: coordinate axes
        directions = [np.eye(dim, dtype=float)[i] for i in range(dim)]
        steps = np.ones(dim, dtype=float)
        
        while budget > 0:
            if time.time() > deadline:
                break
            
            old_x = x.copy()
            min_f = f
            min_dir = None
            min_step = None
            
            for i in range(dim):
                if budget <= 0 or time.time() > deadline:
                    break
                d = directions[i]
                # Find optimal step along direction d via quadratic interpolation
                # Try a few step sizes to find minimum
                step = steps[i]
                x_test = np.clip(x + step * d, -5, 5)
                f_test = func(x_test)
                budget -= 1
                if budget <= 0 or time.time() > deadline:
                    break
                
                if f_test < min_f:
                    # Try to refine with smaller step
                    # Quadratic interpolation with three points
                    # Use current x, x_test, and x - 2*step*d
                    x1 = x.copy()
                    f1 = min_f
                    x2 = x_test.copy()
                    f2 = f_test
                    x3 = np.clip(x - step * d, -5, 5)
                    f3 = func(x3)
                    budget -= 1
                    
                    if budget <= 0 or time.time() > deadline:
                        break
                    
                    # Quadratic interpolation to find minimum
                    a = f1 - 2*f2 + f3
                    b = f3 - f1
                    if abs(a) > 1e-15:
                        t_opt = -b / (2*a)
                    else:
                        t_opt = 0.5
                    
                    # t is in units of step from x1
                    if 0.2 < t_opt < 0.8:
                        x_new = np.clip(x1 + t_opt * step * d, -5, 5)
                        f_new = func(x_new)
                        budget -= 1
                        if f_new < f_test:
                            x = x_new
                            f = f_new
                            if f < min_f:
                                min_f = f
                                min_dir = d
                                min_step = step
                        else:
                            x = x_test
                            f = f_test
                            if f < min_f:
                                min_f = f
                                min_dir = d
                                min_step = step
                    else:
                        x = x_test
                        f = f_test
                        if f < min_f:
                            min_f = f
                            min_dir = d
                            min_step = step
                elif f_test > min_f:
                    # Step in opposite direction
                    x_test_neg = np.clip(x - step * d, -5, 5)
                    f_test_neg = func(x_test_neg)
                    budget -= 1
                    if budget <= 0 or time.time() > deadline:
                        break
                    if f_test_neg < min_f:
                        x = x_test_neg
                        f = f_test_neg
                        min_f = f
                        min_dir = d
                        min_step = -step
                    else:
                        # Shrink step
                        steps[i] *= 0.5
                else:
                    steps[i] *= 0.5
            
            if min_dir is not None:
                # Replace the direction that gave the largest improvement with the overall direction
                # Find direction with largest change
                max_change = 0
                max_idx = 0
                for i in range(dim):
                    change = abs(np.dot(x - old_x, directions[i]))
                    if change > max_change:
                        max_change = change
                        max_idx = i
                directions[max_idx] = (x - old_x) / (np.linalg.norm(x - old_x) + 1e-15)
            else:
                # No improvement, shrink all steps
                steps *= 0.5
                if np.max(steps) < 1e-6:
                    break
        
        return x, f, budget

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Coarse grid search to identify best basin
        # Use a stratified grid with some randomization
        grid_points = []
        # 3x3x3x3x3 = 243 points, but we'll use a smarter approach
        # Use 4 levels per dimension: -4, -1.5, 1.5, 4
        levels = [-4.0, -1.5, 1.5, 4.0]
        
        best_x = None
        best_f = float('inf')
        
        # Evaluate grid points
        for i0 in range(len(levels)):
            for i1 in range(len(levels)):
                for i2 in range(len(levels)):
                    for i3 in range(len(levels)):
                        for i4 in range(len(levels)):
                            x = np.array([levels[i0], levels[i1], levels[i2], levels[i3], levels[i4]])
                            fx = func(x)
                            budget -= 1
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            if budget <= 0 or time.time() > deadline - 100:
                                break
                        if budget <= 0 or time.time() > deadline - 100:
                            break
                    if budget <= 0 or time.time() > deadline - 100:
                        break
                if budget <= 0 or time.time() > deadline - 100:
                    break
            if budget <= 0 or time.time() > deadline - 100:
                break
        
        # Also add a few random points for diversity
        rng = np.random.default_rng(42)
        for _ in range(10):
            if budget <= 0 or time.time() > deadline - 100:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            budget -= 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            budget -= 1
        
        # Phase 2: Powell's method local optimization
        if budget > 10 and time.time() < deadline - 5:
            best_x, best_f, budget = self._powell(func, best_x, budget, deadline, rng)
        
        # Phase 3: Fine-tune with small random perturbations if budget remains
        if budget > 5 and time.time() < deadline - 2:
            sigma = 0.1
            for _ in range(min(budget, 20)):
                if time.time() > deadline - 1:
                    break
                x_test = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fx_test = func(x_test)
                budget -= 1
                if fx_test < best_f:
                    best_x = x_test
                    best_f = fx_test
                    sigma *= 1.3
                else:
                    sigma *= 0.7
        
        return best_x, best_f
# EVOLVE-BLOCK-END
