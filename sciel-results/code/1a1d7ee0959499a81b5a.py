"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid global-local optimizer for BBOB f7 (step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        n_calls = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def check_time():
            return time.time() < deadline and n_calls < budget

        def evaluate(x):
            nonlocal n_calls, best_x, best_f
            if not check_time():
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (d,) or not np.all(np.isfinite(x)):
                return None
            fx = func(x)
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration - sample diverse points
        # Use a mix of uniform random and structured points
        n_global = min(80, budget // 4)
        
        # Start with origin and some structured points
        structured_points = [
            np.zeros(d),
            np.full(d, 0.0),
            np.array([2.0] * d),
            np.array([-2.0] * d),
            np.array([4.0] * d),
            np.array([-4.0] * d),
        ]
        
        for p in structured_points:
            if not check_time():
                break
            evaluate(p)

        # Random sampling in the box
        for _ in range(n_global):
            if not check_time():
                break
            x = rng.uniform(-5, 5, d)
            evaluate(x)

        # Phase 2: Local refinement using CMA-ES-like strategy
        # We'll implement a simple evolution strategy with adaptive step size
        current_x = best_x.copy()
        current_f = best_f
        sigma = 1.5  # initial step size
        
        # Number of iterations for local search
        n_local_remaining = budget - n_calls
        max_local_iters = min(200, n_local_remaining)
        
        for it in range(max_local_iters):
            if not check_time():
                break
            
            # Generate candidate: perturb current best
            # Use a mixture: sometimes large jumps, sometimes small
            if it < 20:
                # Early: larger steps to escape local minima
                step = sigma * 2.0
            elif it < 60:
                step = sigma
            else:
                # Late: smaller steps for fine-tuning
                step = sigma * 0.5
            
            candidate = current_x + step * rng.standard_normal(d)
            candidate = np.clip(candidate, -5, 5)
            
            fc = evaluate(candidate)
            if fc is None:
                break
            
            if fc < current_f:
                current_x = candidate.copy()
                current_f = fc
                sigma = max(sigma * 1.2, 0.01)  # increase step on success
            else:
                sigma = max(sigma * 0.85, 0.001)  # decrease step on failure
            
            # Adaptive: if no improvement for a while, try a random restart
            if it > 0 and it % 30 == 0 and sigma < 0.05:
                # Try a fresh random point
                fresh = rng.uniform(-5, 5, d)
                ff = evaluate(fresh)
                if ff is not None and ff < current_f:
                    current_x = fresh.copy()
                    current_f = ff
                    sigma = 1.0

        # Phase 3: Final fine-tuning with very small steps
        if check_time():
            sigma = 0.05
            n_fine = min(50, budget - n_calls)
            for _ in range(n_fine):
                if not check_time():
                    break
                candidate = current_x + sigma * rng.standard_normal(d)
                candidate = np.clip(candidate, -5, 5)
                fc = evaluate(candidate)
                if fc is None:
                    break
                if fc < current_f:
                    current_x = candidate.copy()
                    current_f = fc
                    sigma = max(sigma * 1.1, 0.001)
                else:
                    sigma = max(sigma * 0.9, 0.0005)

        return best_x, best_f
# EVOLVE-BLOCK-END
