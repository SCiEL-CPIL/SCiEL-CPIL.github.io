"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size, diverse starts,
    and a final local refinement phase. Includes wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit
        calls = 0

        def can_continue():
            return calls < budget and time.time() < deadline

        def remaining():
            return budget - calls

        def make_func_counter():
            nonlocal calls
            def f(x):
                nonlocal calls
                calls += 1
                return func(x)
            return f

        f_count = make_func_counter()

        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        rng = np.random.default_rng(0)

        # Generate diverse starting points
        starts = []
        # Random uniform in box
        for _ in range(8):
            starts.append(rng.uniform(-5, 5, dim))
        # Near origin
        starts.append(np.zeros(dim))
        starts.append(rng.normal(0, 1.0, dim))
        starts.append(rng.normal(0, 2.0, dim))
        # Corners and mid-range
        starts.append(np.array([5.0] * dim))
        starts.append(np.array([-5.0] * dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-3, 3, dim))
        starts.append(rng.uniform(-2, 2, dim))

        # Evaluate all starting points
        start_results = []
        for x0 in starts:
            if not can_continue():
                break
            x0 = np.clip(x0, -5, 5)
            fx = f_count(x0)
            update_best(x0, fx)
            start_results.append((x0, fx))

        if not can_continue():
            return best_x, best_f

        # Sort starts by fitness (best first)
        start_results.sort(key=lambda t: t[1])

        def run_cma_es(x_init, fx_init, max_steps, sigma_init=1.5):
            """Run a simplified CMA-ES-like algorithm from x_init."""
            x = x_init.copy()
            fx = fx_init
            sigma = sigma_init
            n = dim
            # Step size adaptation parameters
            c_sigma = 1 / (n + 3)
            d_sigma = 1 + 2 * max(0, (n + 1) / (n + 3) - 1)
            # Success rate tracking
            successes = 0
            window = 10

            for step in range(max_steps):
                if not can_continue():
                    break
                
                # Generate mutation
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = f_count(y)
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    successes += 1
                    update_best(x, fx)
                else:
                    successes = max(0, successes - 1)
                
                # Adapt sigma based on success rate over a window
                if (step + 1) % window == 0:
                    success_rate = successes / window
                    # 1/5 rule
                    if success_rate > 0.2:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    successes = 0
                
                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.001, 10.0)

            return x, fx

        # Phase 1: Run CMA-ES from the best few starting points
        # Allocate budget: first run gets ~40%, second ~30%, third ~20%
        n_es_runs = min(3, len(start_results))
        allocations = []
        for i in range(n_es_runs):
            if i == 0:
                allocations.append(min(remaining(), 200))
            elif i == 1:
                allocations.append(min(remaining(), 150))
            else:
                allocations.append(min(remaining(), 100))

        for i in range(n_es_runs):
            if not can_continue():
                break
            x0, fx0 = start_results[i]
            run_cma_es(x0, fx0, allocations[i])

        # Phase 2: If budget remains, do a few random restarts
        n_restarts = 3
        for _ in range(n_restarts):
            if not can_continue():
                break
            x0 = rng.uniform(-5, 5, dim)
            x0 = np.clip(x0, -5, 5)
            fx0 = f_count(x0)
            update_best(x0, fx0)
            if can_continue():
                run_cma_es(x0, fx0, min(remaining(), 40), sigma_init=1.0)

        # Phase 3: Final local refinement with decreasing sigma
        if can_continue() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            n_refine = remaining()
            for i in range(n_refine):
                if not can_continue():
                    break
                # Gradually reduce sigma
                sigma = max(0.005, 0.3 * (0.98 ** i))
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = f_count(y)
                if fy < fx:
                    x = y
                    fx = fy
                    update_best(x, fx)
                else:
                    sigma *= 0.95

        return best_x, best_f

# EVOLVE-BLOCK-END
