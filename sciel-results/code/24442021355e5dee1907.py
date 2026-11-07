"""Optimizer for BBOB f16 (Weierstrass) with coordinate-wise adaptive ES + gradient polish."""
import numpy as np
import time


class Optimizer:
    """Coordinate-wise adaptive (1+1)-ES with finite-difference gradient polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return float(func(x))

        # Phase 1: Global random sampling
        rng = np.random.default_rng(7)
        n_global = min(30, budget)
        for _ in range(n_global):
            if calls_made >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # Phase 2: Coordinate-wise adaptive (1+1)-ES
        # Each dimension has its own step size, updated based on success rate
        x = best_x.copy()
        sigma = np.ones(dim) * 0.8  # per-dimension step sizes
        window = 10  # recent moves window per dimension
        successes = [0] * dim  # count of successful moves per dimension in window
        total = [0] * dim     # total moves per dimension in window
        rate = 1.0 / (5.0 * dim)  # adaptation rate

        es_budget = budget - 80  # reserve 80 for gradient polish
        es_max = calls_made + es_budget

        while calls_made < es_max and time.time() < deadline:
            # Generate candidate: perturb each dimension independently
            y = x.copy()
            for i in range(dim):
                delta = sigma[i] * rng.standard_normal()
                y[i] = x[i] + delta

            y = np.clip(y, -5.0, 5.0)

            fx = safe_func(x)
            fy = safe_func(y)

            if fy < fx:
                # Success: update best if needed
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                x = y.copy()
                # Track success per dimension
                for i in range(dim):
                    successes[i] += 1
                    total[i] += 1
            else:
                # Failure
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                for i in range(dim):
                    total[i] += 1

            # Adapt step sizes based on success rate in window
            for i in range(dim):
                if total[i] >= window:
                    success_rate = successes[i] / total[i]
                    # Target ~1/3 success rate (optimal for (1+1)-ES)
                    target = 1.0 / 3.0
                    if success_rate > target:
                        sigma[i] *= np.exp(rate * (success_rate - target) * 2)
                    else:
                        sigma[i] *= np.exp(rate * (success_rate - target) * 2)
                    # Reset window
                    successes[i] = 0
                    total[i] = 0

                # Clamp sigma
                sigma[i] = np.clip(sigma[i], 1e-8, 10.0)

            # Early stopping if all sigmas are tiny
            if np.all(sigma < 1e-6):
                break

        # Phase 3: Finite-difference gradient descent polish
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            # Estimate gradient via central differences
            h = 1e-4
            for step in range(200):
                if calls_made >= budget or time.time() >= deadline:
                    break

                # Compute gradient
                grad = np.zeros(dim)
                for i in range(dim):
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] = min(xp[i] + h, 5.0)
                    xm[i] = max(xm[i] - h, -5.0)
                    fp = safe_func(xp)
                    fm = safe_func(xm)
                    grad[i] = (fp - fm) / (xp[i] - xm[i]) if xp[i] != xm[i] else 0.0

                # Check if we've exhausted budget
                if calls_made >= budget or time.time() >= deadline:
                    break

                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-10:
                    break

                # Adaptive step size
                lr = 0.01 / (1.0 + step * 0.05)
                direction = -grad / gnorm
                # Try a few step sizes
                found = False
                for ls in [lr, lr * 0.5, lr * 0.25, lr * 0.1]:
                    xn = x + ls * direction
                    xn = np.clip(xn, -5.0, 5.0)
                    fn = safe_func(xn)
                    if fn < best_f:
                        best_f = fn
                        best_x = xn.copy()
                        x = xn.copy()
                        found = True
                        break
                    if calls_made >= budget or time.time() >= deadline:
                        break

                if not found:
                    # Try smaller steps
                    for ls in [lr * 0.05, lr * 0.01, lr * 0.001]:
                        xn = x + ls * direction
                        xn = np.clip(xn, -5.0, 5.0)
                        fn = safe_func(xn)
                        if fn < best_f:
                            best_f = fn
                            best_x = xn.copy()
                            x = xn.copy()
                            found = True
                            break
                        if calls_made >= budget or time.time() >= deadline:
                            break

                    if not found:
                        break

        return best_x, best_f
