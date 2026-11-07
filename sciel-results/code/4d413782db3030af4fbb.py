"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate descent with golden-section search for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        # Golden section search along one coordinate
        def golden_search(x, coord, a, b, func, budget_remaining):
            """
            Golden section search along coordinate `coord` of x,
            searching in interval [a, b].
            Returns (best_x, best_f, num_calls)
            """
            if budget_remaining < 5:
                return x, func(x), 1
            
            phi = (1 + np.sqrt(5)) / 2
            invphi = 1 / phi
            invphi2 = invphi * invphi
            
            # Initial interior points
            x1 = b - (b - a) / phi
            x2 = a + (b - a) / phi
            
            # Evaluate x1 and x2
            x_test1 = x.copy()
            x_test1[coord] = x1
            f1 = func(x_test1)
            
            x_test2 = x.copy()
            x_test2[coord] = x2
            f2 = func(x_test2)
            
            num_calls = 2
            budget_remaining -= 2
            
            # Iterate
            while budget_remaining >= 2 and (b - a) > 1e-10:
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = b - (b - a) / phi
                    x_test1 = x.copy()
                    x_test1[coord] = x1
                    f1 = func(x_test1)
                    num_calls += 1
                    budget_remaining -= 1
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = a + (b - a) / phi
                    x_test2 = x.copy()
                    x_test2[coord] = x2
                    f2 = func(x_test2)
                    num_calls += 1
                    budget_remaining -= 1
            
            # Return best point
            if f1 < f2:
                best_val = x1
                best_f = f1
            else:
                best_val = x2
                best_f = f2
            
            x_best = x.copy()
            x_best[coord] = best_val
            return x_best, best_f, num_calls

        # Phase 1: Coarse search - do a few passes of coordinate descent with wide intervals
        # Use ~40% of budget for coarse, 60% for fine
        coarse_budget = int(budget * 0.4)
        fine_budget = budget - coarse_budget - 1  # -1 for initial call
        
        # Coarse phase: multiple passes
        pass_count = 0
        while calls < budget - fine_budget:
            for coord in range(dim):
                if calls >= budget - fine_budget:
                    break
                remaining = budget - fine_budget - calls
                if remaining < 10:
                    break
                x, fx, n_calls = golden_search(x, coord, -5.0, 5.0, func, remaining)
                calls += n_calls
        
        # Phase 2: Fine search - narrower intervals around current best
        # Estimate current ranges
        for coord in range(dim):
            if calls >= budget:
                break
            remaining = budget - calls
            if remaining < 10:
                break
            # Narrow interval around current x[coord]
            delta = 1.0
            a = max(-5.0, x[coord] - delta)
            b = min(5.0, x[coord] + delta)
            x, fx, n_calls = golden_search(x, coord, a, b, func, remaining)
            calls += n_calls
        
        # Phase 3: Even finer search
        for coord in range(dim):
            if calls >= budget:
                break
            remaining = budget - calls
            if remaining < 10:
                break
            delta = 0.1
            a = max(-5.0, x[coord] - delta)
            b = min(5.0, x[coord] + delta)
            x, fx, n_calls = golden_search(x, coord, a, b, func, remaining)
            calls += n_calls
        
        # Use any remaining budget for the most critical coordinates
        # (typically the one with the steepest gradient, but we'll just cycle)
        coord = 0
        while calls < budget:
            remaining = budget - calls
            if remaining < 5:
                break
            delta = 0.01
            a = max(-5.0, x[coord] - delta)
            b = min(5.0, x[coord] + delta)
            x, fx, n_calls = golden_search(x, coord, a, b, func, remaining)
            calls += n_calls
            coord = (coord + 1) % dim
        
        return x, fx
# EVOLVE-BLOCK-END
