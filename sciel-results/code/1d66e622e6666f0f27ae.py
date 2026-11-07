"""Multi-start adaptive (1+1)-ES with deadline guard for BBOB Weierstrass (f16)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        # Fixed seed for the optimizer's internal randomness to ensure consistent behavior
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0  # Safety margin below the 330s kill

        # ---- Phase 1: cheap multi-start to find a good basin early ----
        # Strategy: 1 zero seed + 5 random seeds.
        # This balances the cost of exploration with the need for early progress.
        # Weierstrass has a broad basin, so randoms are effective.
        n_starts = 6
        seeds = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            seeds.append(rng.uniform(-4.5, 4.5, dim))

        best_x = None
        best_f = np.inf
        
        # Evaluate seeds
        for s in seeds:
            if time.time() > deadline:
                break
            x_try = np.clip(s, -5.0, 5.0)
            fx = func(x_try)
            if fx < best_f:
                best_f = fx
                best_x = x_try.copy()

        if best_x is None:
            # Fallback if deadline hit immediately (unlikely)
            x0 = np.zeros(dim)
            best_f = func(x0)
            best_x = x0.copy()

        # ---- Phase 2: adaptive (1+1)-ES from the best start ----
        x = best_x.copy()
        fx = best_f
        
        # Start with a larger sigma to escape local minima in Weierstrass
        # Weierstrass is multimodal and rugged; large steps help jump out.
        sigma = 2.0  
        min_sigma, max_sigma = 1e-7, 8.0
        target = 1.0 / 5.0
        c_sigma = 0.35  # Faster adaptation rate as seen in higher-scoring inspiration

        n_calls = len(seeds)
        remaining = self.budget - n_calls
        
        for i in range(remaining):
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                # Success: increase sigma
                sigma *= np.exp(c_sigma * (1.0 - target))
            else:
                # Failure: decrease sigma
                sigma *= np.exp(c_sigma * (0.0 - target))
            
            # Clamp sigma
            sigma = min(max(sigma, min_sigma), max_sigma)

        return x, fx
# EVOLVE-BLOCK-END
