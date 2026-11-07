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

        def remaining_time():
            return deadline - time.time()

        # Phase 1: Multi-start exploration with diverse initial points
        # Use a combination of deterministic spread and random sampling
        n_starts = 12
        initial_points = []

        # Stratified/deterministic spread
        for i in range(n_starts):
            x = np.zeros(dim)
            for d in range(dim):
                # Use different patterns for diversity
                if i < 4:
                    # Corners and centers
                    base = (i % 2) * 10.0 - 5.0
                    if i >= 2:
                        base = 0.0
                    x[d] = base + rng.uniform(-1.5, 1.5)
                elif i < 8:
                    # Mid-range spread
                    base = (i - 4 + 0.5) / 4.0 * 10.0 - 5.0
                    x[d] = base + rng.uniform(-1.0, 1.0)
                else:
                    # Fully random
                    x[d] = rng.uniform(-5.0, 5.0)
            x = np.clip(x, -5.0, 5.0)
            initial_points.append(x)

        best_x = None
        best_fx = np.inf
        calls = 0

        # Evaluate all initial points
        for x0 in initial_points:
            if calls >= budget or remaining_time() < 1.0:
                break
            fx = func(x0)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x0.copy()

        # Sort initial points by function value
        evaluated = []
        for x0 in initial_points:
            if calls >= budget or remaining_time() < 1.0:
                break
            fx = func(x0)
            calls += 1
            evaluated.append((fx, x0))
        evaluated.sort(key=lambda t: t[0])

        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_fx

        # Allocate budget: top 3 get more, others get less
        top_k = min(3, len(evaluated))
        mid_k = min(5, len(evaluated) - top_k)

        allocs = []
        for i in range(top_k):
            allocs.append(int(remaining * 0.35))
        for i in range(mid_k):
            allocs.append(max(1, int(remaining * 0.05)))

        total_alloc = sum(allocs)
        leftover = remaining - total_alloc
        if leftover > 0 and allocs:
            allocs[0] += leftover

        for idx, (fx, x0) in enumerate(evaluated):
            if idx >= len(allocs) or remaining_time() < 1.0:
                break
            alloc = allocs[idx]
            if alloc <= 0:
                continue

            # Run (1+1)-ES from this starting point with adaptive sigma
            x = x0.copy()
            fx_cur = fx
            sigma = 2.0
            no_improve = 0
            sigma_history = [sigma]

            for _ in range(alloc):
                if calls >= budget or remaining_time() < 1.0:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1

                if fy < fx_cur:
                    x = y
                    fx_cur = fy
                    no_improve = 0
                    # Increase sigma on success, but cap it
                    sigma = min(sigma * 1.2, 5.0)
                else:
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma *= 0.95

                # Update global best
                if fx_cur < best_fx:
                    best_fx = fx_cur
                    best_x = x.copy()

                # If stuck, try a larger jump
                if no_improve > 20:
                    sigma = min(sigma * 2.0, 5.0)
                    no_improve = 0

            # Fine-tuning phase for the best point
            if fx_cur <= best_fx + 1e-6 and remaining_time() > 5.0:
                sigma_fine = 0.5
                x_fine = x.copy()
                fx_fine = fx_cur
                fine_budget = min(30, budget - calls)
                
                for _ in range(fine_budget):
                    if calls >= budget or remaining_time() < 1.0:
                        break
                    y = x_fine + sigma_fine * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls += 1
                    if fy < fx_fine:
                        x_fine = y
                        fx_fine = fy
                        sigma_fine = min(sigma_fine * 1.1, 1.0)
                    else:
                        sigma_fine *= 0.9
                    if fx_fine < best_fx:
                        best_fx = fx_fine
                        best_x = x_fine.copy()

        # Final polish: if we have time and budget, do a very fine local search
        if best_x is not None and remaining_time() > 10.0 and calls < budget:
            sigma_polish = 0.1
            x_polish = best_x.copy()
            fx_polish = best_fx
            polish_budget = min(50, budget - calls)
            
            for _ in range(polish_budget):
                if calls >= budget or remaining_time() < 1.0:
                    break
                y = x_polish + sigma_polish * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < fx_polish:
                    x_polish = y
                    fx_polish = fy
                    sigma_polish = min(sigma_polish * 1.05, 0.5)
                else:
                    sigma_polish *= 0.95
                if fx_polish < best_fx:
                    best_fx = fx_polish
                    best_x = x_polish.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
