"""Program for BBOB f18 optimization using multistart CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import cma

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')

        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def time_ok():
            return time.time() < deadline

        def calls_ok(needed=1):
            return n_calls + needed <= budget

        # Phase 1: Initial sampling to find a good starting region
        rng = np.random.default_rng(12345)
        n_initial = min(20, budget // 10)
        
        # Mix of uniform random points and a few near-origin points
        for i in range(n_initial):
            if not (time_ok() and calls_ok()):
                break
            if i < 5:
                # Points near origin with varying scales
                scale = 2.0 * (i + 1) / 5
                x = rng.uniform(-scale, scale, dim)
            elif i < 10:
                # Moderate spread
                x = rng.uniform(-3, 3, dim)
            else:
                # Full range
                x = rng.uniform(-4.5, 4.5, dim)
            timed_call(x)

        if not (time_ok() and calls_ok()):
            return best_x, best_f

        # Phase 2: Multistart CMA-ES with adaptive restarts
        # We'll run several CMA-ES instances, each with a portion of the budget
        remaining = budget - n_calls
        if remaining < 30:
            return best_x, best_f

        # Determine number of restarts based on remaining budget
        # Each restart needs at least ~20 evaluations to be useful
        n_restarts = max(1, min(8, remaining // 25))
        
        # For f18 (multimodal), we want diverse starting points
        # Use a combination of: best found, perturbations, and new random points
        start_points = []
        sigmas = []
        
        # Start 1: best point with moderate sigma
        start_points.append(best_x.copy())
        sigmas.append(1.0)
        
        # Start 2: perturbation of best
        start_points.append(np.clip(best_x + 0.5 * rng.standard_normal(dim), -4.5, 4.5))
        sigmas.append(0.5)
        
        # Starts 3+: new random points
        for i in range(2, n_restarts):
            if i % 2 == 0:
                x0 = rng.uniform(-3, 3, dim)
                s0 = 1.0
            else:
                x0 = rng.uniform(-4, 4, dim)
                s0 = 1.5
            start_points.append(x0)
            sigmas.append(s0)

        # Budget allocation: give more budget to earlier restarts (they're more promising)
        # Use a decreasing allocation
        weights = np.array([1.0 / (i + 1) for i in range(n_restarts)])
        weights /= weights.sum()
        budgets = (remaining * weights).astype(int)
        # Ensure minimum budget
        for i in range(n_restarts):
            budgets[i] = max(20, budgets[i])
        # Adjust to not exceed remaining
        total_alloc = budgets.sum()
        if total_alloc > remaining:
            # Scale down proportionally
            budgets = (budgets * (remaining / total_alloc)).astype(int)
            for i in range(n_restarts):
                budgets[i] = max(10, budgets[i])

        for r in range(n_restarts):
            if not (time_ok() and calls_ok()):
                break
            
            x0 = start_points[r]
            sigma0 = sigmas[r]
            max_evals = budgets[r]
            
            if max_evals < 10:
                continue

            # CMA-ES options
            opts = {
                'maxfevals': max_evals,
                'seed': 42 + r,
                'verbose': -9,
                'bounds': [-5.0] * dim,
                'tolfun': 1e-10,
                'tolx': 1e-8,
                'tolstall': 30,  # Restart after 30 evals without improvement
                'CMA_adaptation_of_diagonal_cov': True,
                'CMA_adaptation_of_full_cov': True,
            }
            
            try:
                es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
                while not es.stop() and time_ok() and calls_ok():
                    # Generate batch
                    batch_size = min(es.popsize, budget - n_calls)
                    if batch_size < 1:
                        break
                    es.ask(batch_size)
                    for x in es.pop:
                        if not (time_ok() and calls_ok()):
                            break
                        f = timed_call(x)
                        es.tell(x, f)
            except Exception:
                # If CMA-ES fails, try a simple random hill-climb as fallback
                try:
                    x_cur = x0.copy()
                    f_cur = best_f if np.allclose(x_cur, best_x) else timed_call(x_cur)
                    step = sigma0
                    for _ in range(max_evals):
                        if not (time_ok() and calls_ok()):
                            break
                        # Random perturbation
                        x_try = np.clip(x_cur + step * rng.standard_normal(dim), -5, 5)
                        f_try = timed_call(x_try)
                        if f_try < f_cur:
                            x_cur = x_try
                            f_cur = f_try
                            step *= 1.1
                        else:
                            step *= 0.9
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
