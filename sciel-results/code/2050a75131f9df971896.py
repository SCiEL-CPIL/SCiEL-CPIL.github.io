import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim

        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        all_points = []

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError()
            if calls_made >= budget:
                raise ValueError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            all_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def halton_sequence(n, dim):
            """Generate n Halton sequence points in [0,1)^dim."""
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(dim):
                    prime = primes[d % len(primes)]
                    f = 1.0
                    ip = i
                    while ip > 0:
                        f /= prime
                        ip = int(ip / prime)
                    point.append(f)
                points.append(point)
            return np.array(points)

        # Phase 1: Deterministic Halton sequence exploration
        # Slightly increased from 40 to 50 to improve basin discovery probability
        n_halton = 50
        halton_pts = halton_sequence(n_halton, dim)
        # Map from [0,1)^dim to [-5,5]^dim
        halton_pts_mapped = -5.0 + 10.0 * halton_pts

        for i in range(n_halton):
            if time.time() > deadline - 20 or calls_made >= budget - 50:
                break
            eval_func(halton_pts_mapped[i])

        # Sort by fitness
        all_points.sort(key=lambda p: p[0])

        # Get top diverse starting points
        start_points = []
        min_dist = 2.0
        for f, x in all_points:
            if len(start_points) >= 5:
                break
            if all(np.linalg.norm(x - sp) > min_dist for sp in start_points):
                start_points.append(x)

        # Fill if needed
        while len(start_points) < 5 and len(all_points) > len(start_points):
            f, x = all_points[len(start_points)]
            if all(np.linalg.norm(x - sp) > 0.5 for sp in start_points):
                start_points.append(x)

        # Phase 2: Multi-start Powell refinement
        # Powell's method is often more efficient than Nelder-Mead for smooth,
        # low-dimensional continuous functions as it avoids simplex geometry issues.
        remaining_budget = budget - calls_made
        n_starts = len(start_points)
        if n_starts == 0:
            n_starts = 1
            start_points = [best_x.copy()]

        powell_calls_per_start = max(20, remaining_budget // (n_starts + 1))

        for i, sp in enumerate(start_points):
            if time.time() > deadline - 15 or calls_made >= budget - 10:
                break

            try:
                minimize(
                    eval_func,
                    sp,
                    method='Powell',
                    options={
                        'maxiter': min(powell_calls_per_start, 100),
                        'xtol': 1e-6,
                        'ftol': 1e-8,
                        'maxfev': min(powell_calls_per_start, 100)
                    }
                )
            except Exception:
                continue

        # Phase 3: Final refinement with Nelder-Mead
        # NM is excellent for the final "polish" to squeeze out last bits of accuracy
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline - 5:
            try:
                minimize(
                    eval_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining - 2, 50),
                        'xatol': 1e-7,
                        'fatol': 1e-9,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return best_x, best_f
