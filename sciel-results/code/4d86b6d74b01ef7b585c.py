"""Portfolio search optimizer for BBOB f20 (Schwefel) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start (1+1)-ES with adaptive step size, population-based exploration,
    and a local refinement phase. Designed for BBOB f20 (Schwefel) on [-5,5]^5
    with budget 500 and a 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start exploration with 8 diverse initial points
        # Phase 2: Local refinement around the best point found

        # Generate diverse initial points using a quasi-random (Halton-like) approach
        # plus some random points for robustness
        initial_points = []

        # Use stratified sampling: divide the box into sub-regions
        # For dim=5, use 8 points spread across the space
        n_starts = 8

        # Create a grid of points
        for i in range(n_starts):
            # Use a combination of deterministic and random offsets
            # Point i is biased towards a different region
            x = np.zeros(dim)
            for d in range(dim):
                # Spread points across the range with some randomness
                base = (i + 0.5) / n_starts * 10.0 - 5.0
                x[d] = base + rng.uniform(-1.0, 1.0)
            x = np.clip(x, -5.0, 5.0)
            initial_points.append(x)

        # Add a few fully random points for diversity
        for _ in range(4):
            x = rng.uniform(-5.0, 5.0, dim)
            initial_points.append(x)

        best_x = None
        best_fx = np.inf
        calls = 0

        # Evaluate all initial points
        for x0 in initial_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            fx = func(x0)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x0.copy()

        # Phase 2: For each promising starting point, run a short (1+1)-ES
        # Focus more evaluations on the best few starts
        # Sort initial points by their function value
        evaluated = []
        for x0 in initial_points:
            if calls >= budget:
                break
            fx = func(x0)
            calls += 1
            evaluated.append((fx, x0))
        evaluated.sort(key=lambda t: t[0])

        # Allocate remaining budget: give more to the top 3 starts
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_fx

        # Top 3 get 40% each, next 5 get 4% each (if budget allows)
        top_k = min(3, len(evaluated))
        mid_k = min(5, len(evaluated) - top_k)

        allocs = []
        for i in range(top_k):
            allocs.append(int(remaining * 0.4))
        for i in range(mid_k):
            allocs.append(max(1, int(remaining * 0.04)))

        # Distribute any leftover
        total_alloc = sum(allocs)
        leftover = remaining - total_alloc
        if leftover > 0 and allocs:
            allocs[0] += leftover

        for idx, (fx, x0) in enumerate(evaluated):
            if idx >= len(allocs):
                break
            alloc = allocs[idx]
            if alloc <= 0:
                continue
            if time.time() > deadline:
                break

            # Run (1+1)-ES from this starting point
            x = x0.copy()
            fx_cur = fx
            sigma = 1.5  # Initial step size
            no_improve = 0

            for _ in range(alloc):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1

                if fy < fx_cur:
                    x = y
                    fx_cur = fy
                    no_improve = 0
                    sigma *= 1.3
                else:
                    no_improve += 1
                    sigma *= 0.9

                # Update global best
                if fx_cur < best_fx:
                    best_fx = fx_cur
                    best_x = x.copy()

                # Adaptive sigma: if stuck for too long, increase exploration
                if no_improve > 15:
                    sigma *= 1.5
                    no_improve = 0

            # After local search, if this point improved the best, do fine-tuning
            if fx_cur < best_fx + 1e-6:
                # Fine-tuning phase with smaller sigma
                sigma = 0.1
                x_fine = x.copy()
                fx_fine = fx_cur
                fine_budget = min(20, budget - calls)
                for _ in range(fine_budget):
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    y = x_fine + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls += 1
                    if fy < fx_fine:
                        x_fine = y
                        fx_fine = fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
                    if fx_fine < best_fx:
                        best_fx = fx_fine
                        best_x = x_fine.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
