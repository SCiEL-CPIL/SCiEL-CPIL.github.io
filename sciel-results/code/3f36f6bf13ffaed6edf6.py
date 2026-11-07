"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with coordinate descent polish and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Counting wrapper
        call_count = [0]
        best_x = None
        best_f = np.inf

        def counting_func(x):
            nonlocal best_x, best_f
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f if best_f < np.inf else 0.0
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Generate multiple starting points
        # Mix of random points and points near the expected minimum region
        starts = []
        
        # Some random uniform points
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, dim))
        
        # Points biased towards center (since minimizer is in [-4,4]^5)
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, dim))
        
        # A few points very close to origin
        for _ in range(5):
            starts.append(rng.uniform(-2, 2, dim))

        # Shuffle starts
        rng.shuffle(starts)

        bounds = [(-5.0, 5.0)] * dim

        # Try each start with L-BFGS-B
        for start in starts:
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    counting_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': budget - call_count[0]}
                )
            except Exception:
                continue

            # If L-BFGS-B finished, do a quick coordinate descent polish
            if call_count[0] < budget and time.time() < deadline:
                x = result.x.copy()
                fx = counting_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Simple coordinate descent with line search
                for _ in range(5):  # A few sweeps
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    for i in range(dim):
                        if call_count[0] >= budget or time.time() > deadline:
                            break
                        # Try moving along axis i
                        step = 0.5
                        improved = True
                        while improved and step > 1e-6 and call_count[0] < budget and time.time() < deadline:
                            improved = False
                            for direction in [1, -1]:
                                trial = x.copy()
                                trial[i] = np.clip(trial[i] + direction * step, -5, 5)
                                f_trial = counting_func(trial)
                                if f_trial < fx:
                                    x = trial
                                    fx = f_trial
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                                    improved = True
                                    break
                            step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
