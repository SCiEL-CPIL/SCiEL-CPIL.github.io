import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        dim = self.dim
        budget = self.budget
        deadline = start_time + 290.0  # Safety margin before 330s kill

        def elapsed_ok():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or not elapsed_ok():
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global scan with Sobol for early progress
        scan_budget = min(80, budget // 4)
        sampler = qmc.Sobol(d=dim, scramble=True, seed=123)
        points = sampler.random(scan_budget)
        points = points * 10.0 - 5.0

        for p in points:
            if not elapsed_ok() or calls_made >= budget:
                break
            fx = eval_func(p)
            if fx is None:
                break

        if best_x is None:
            # Fallback
            fx = eval_func(np.zeros(dim))
            if best_x is None:
                return np.zeros(dim), np.inf

        # Phase 2: Adaptive (1,lambda)-ES with restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        lam = 4  # population size
        remaining = budget - calls_made
        stagnation_window = 20
        no_improve_count = 0
        max_stagnation = 30

        while remaining > 0 and elapsed_ok():
            # Generate lam candidates
            candidates = []
            for _ in range(lam):
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5.0, 5.0)
                candidates.append(y)

            improved = False
            for y in candidates:
                if not elapsed_ok() or calls_made >= budget:
                    break
                fy = eval_func(y)
                if fy is None:
                    break
                if fy < fx:
                    x = y
                    fx = fy
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            remaining = budget - calls_made

            # Adapt sigma using 1/5 success rule approximation
            # Count successes in this generation
            # We need to track successes; let's redo with proper tracking
            # For simplicity, use a window-based adaptation
            if no_improve_count > max_stagnation:
                # Restart: jump to a new random point or increase sigma significantly
                sigma *= 3.0
                sigma = min(sigma, 8.0)
                no_improve_count = 0
                # Optionally restart from a new random location
                if rng.random() < 0.3:
                    x = rng.uniform(-5, 5, size=dim)
                    fx = eval_func(x)
                    if fx is None:
                        break
            else:
                # Gentle adaptation
                if improved:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-3, 10.0)

        return best_x, best_f
