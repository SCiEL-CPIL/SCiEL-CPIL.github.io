"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for Büche-Rastrigin f4 on [-5,5]^5.
    
    Combines multi-start exploration with adaptive (1+1)-ES refinement.
    The Büche-Rastrigin function is highly multimodal. We first sample a set of
    random points to find the best basin, then apply a sequence of adaptive
    (1+1)-ES runs from the top candidates to refine the solution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _adaptive_es(self, func, x_start, fx_start, budget_remaining, dim, rng, deadline, x_best, f_best):
        """
        Run adaptive (1+1)-ES starting from x_start.
        Returns updated x_best, f_best, and remaining budget.
        """
        x = x_start.copy()
        fx = fx_start
        sigma = 1.5
        success_window = []
        window_size = 10
        min_sigma = 0.01
        max_sigma = 10.0
        budget_used = 0

        while budget_used < budget_remaining and time.time() < deadline:
            # Generate trial point
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = float(func(y))
            budget_used += 1
            
            if fy < fx:
                x, fx = y, fy
                success_window.append(1)
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
            else:
                success_window.append(0)
            
            # Adapt sigma based on 1/5th success rule
            if len(success_window) >= window_size:
                success_rate = np.mean(success_window[-window_size:])
                if success_rate > 0.2:
                    sigma *= 1.05
                elif success_rate < 0.1:
                    sigma *= 0.95
                
                if len(success_window) > window_size:
                    success_window.pop(0)
                
                sigma = max(min_sigma, min(sigma, max_sigma))
        
        return x_best, f_best, budget_remaining - budget_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        x_best = np.zeros(dim)
        f_best = float('inf')
        budget_used = 0

        # Phase 1: Multi-Start Exploration
        # Sample a diverse set of points to find promising basins.
        n_starts = 15
        candidates = []
        candidate_f = []

        for _ in range(n_starts):
            if budget_used >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = float(func(x))
            budget_used += 1
            candidates.append(x)
            candidate_f.append(fx)
            
            if fx < f_best:
                f_best = fx
                x_best = x.copy()

        if not candidates:
            return x_best, f_best

        # Sort candidates by fitness
        order = np.argsort(candidate_f)
        # Keep the top few for refinement
        n_refine = min(5, len(candidates))
        
        # Phase 2: Adaptive (1+1)-ES Refinement from top candidates
        # Allocate remaining budget to refinement runs
        remaining_budget = budget - budget_used
        # Give each refinement run a portion of the budget
        # We'll do sequential refinements, each with a capped budget
        
        for idx in range(n_refine):
            if budget_used >= budget or time.time() >= deadline:
                break
            
            start_idx = order[idx]
            x_start = candidates[start_idx].copy()
            fx_start = candidate_f[start_idx]
            
            # Allocate budget for this refinement: 
            # More budget for better starting points
            # First candidate gets 40% of remaining, others get 15% each
            if idx == 0:
                alloc = int(remaining_budget * 0.4)
            else:
                alloc = int(remaining_budget * 0.15)
            
            alloc = min(alloc, budget - budget_used)
            if alloc <= 0:
                break
            
            x_best, f_best, alloc_used = self._adaptive_es(
                func, x_start, fx_start, alloc, dim, rng, deadline, x_best, f_best
            )
            budget_used += alloc_used
        
        return x_best, f_best
# EVOLVE-BLOCK-END
