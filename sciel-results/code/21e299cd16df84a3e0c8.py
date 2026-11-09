"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid multi-start CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin for 300s limit

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start exploration with diverse initial points
        # Generate several starting points: random, zero, small random
        n_starts = 5
        starts = []
        # Zero
        starts.append(np.zeros(dim))
        # Small random
        starts.append(rng.uniform(-0.5, 0.5, dim))
        # Random in box
        for _ in range(n_starts - 2):
            starts.append(rng.uniform(-5, 5, dim))

        # Evaluate all starting points
        for x0 in starts:
            if time.time() > deadline:
                break
            fx = eval_func(np.clip(x0, -5, 5))

        # Phase 2: Run CMA-ES from each starting point with allocated budget
        # Allocate budget proportionally: more budget to better starts
        # But first, let's run a short CMA-ES from each start to see which is promising
        short_budget = max(20, budget // (n_starts + 1))
        remaining_budget = budget - n_starts

        # Run CMA-ES from each starting point
        for x0 in starts:
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # CMA-ES parameters
            x = np.clip(x0.copy(), -5, 5)
            fx = eval_func(x) if x is not None else best_f
            # Actually we already evaluated x0, so let's use that
            # Re-evaluate to be safe (but we already have it)
            # Let's just use the best_f from the starting point evaluation
            
            # CMA-ES implementation
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            sigma = 1.0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            cs = (pop_size + 2) / (dim + 4)
            cmu = min(0.05, (1 - cs) / 2)
            cmuc = 0.0
            damps = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cmu
            dampc = (4 + dim / 4) / (dim + 4 + 2)
            cc = (1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1)) / (dim + 1)

            # Covariance matrix adaptation
            # For simplicity, use a simplified version
            # Actually, let's use a simpler (1+1)-ES with sigma adaptation for speed
            
            # Let's use a simpler approach: (1+1)-ES with success-rate adaptation
            sigma = 1.0
            success_count = 0
            window = 10
            
            for iteration in range(short_budget):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                
                # Adapt sigma based on success rate
                if (iteration + 1) % window == 0:
                    success_rate = success_count / window
                    if success_rate > 0.5:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9
                    success_count = 0

        # Phase 3: Final polish with smaller sigma on the best solution found
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - calls_made
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
