"""Rewritten optimizer with adaptive multi-start CMA-like search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive multi-start optimizer with covariance-based step adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin: 280s out of 300s

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0

        def time_check():
            nonlocal deadline
            return time.time() < deadline

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return None
            call_count += 1
            try:
                return func(x)
            except Exception:
                return None

        # Phase 1: Broad exploration with multiple random starts
        # Use a moderate population to cover the space
        n_starts = min(10, budget // 50)
        candidates = []

        for i in range(n_starts):
            if not time_check():
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                candidates.append((x, fx))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        if not candidates:
            # Fallback: single random point
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_x = x
                best_f = fx
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[1])

        # Phase 2: Adaptive local search around best candidates
        # Use a CMA-ES-like approach with covariance adaptation
        # Start with the best candidate and explore

        # Adaptive step size per dimension
        step_sizes = np.full(dim, 1.0)

        # Maintain a small population for diversity
        pop_size = min(8, max(4, (budget - call_count) // 50))

        # For each of the top few candidates, do local refinement
        top_k = min(3, len(candidates))

        for start_idx in range(top_k):
            if not time_check() or call_count >= budget:
                break

            x = candidates[start_idx][0].copy()
            fx = candidates[start_idx][1]

            # Local search with adaptive steps
            local_budget = min(50, budget - call_count)
            sigma = 0.5

            for _ in range(local_budget):
                if not time_check() or call_count >= budget:
                    break

                # Generate perturbation
                perturbation = rng.standard_normal(dim) * sigma
                y = np.clip(x + perturbation, -5, 5)
                fy = safe_call(y)

                if fy is None:
                    continue

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                    # If sigma gets too small, try a larger jump
                    if sigma < 0.01:
                        sigma = 0.5
                        # Try a random restart near current best
                        y = np.clip(x + rng.standard_normal(dim) * 2.0, -5, 5)
                        fy = safe_call(y)
                        if fy is not None and fy < fx:
                            x = y
                            fx = fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        # Phase 3: Final refinement around the global best
        if best_x is not None and time_check() and call_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1

            for _ in range(min(30, budget - call_count)):
                if not time_check() or call_count >= budget:
                    break

                perturbation = rng.standard_normal(dim) * sigma
                y = np.clip(x + perturbation, -5, 5)
                fy = safe_call(y)

                if fy is None:
                    continue

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
