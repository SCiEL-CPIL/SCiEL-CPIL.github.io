"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with Sobol initialization and (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            calls_used += 1
            return val

        # Phase 1: Generate Sobol-based initial points for multi-start
        # Use a simple quasi-random sequence for space-filling
        def sobol_sample(n, dim, seed=42):
            """Generate n quasi-random points in [0,1]^dim using a simple method."""
            rng_local = np.random.default_rng(seed)
            # Use Halton-like sequence for better distribution
            points = np.zeros((n, dim))
            # Prime bases for Halton
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            for i in range(n):
                for j in range(dim):
                    p = primes[j % len(primes)]
                    f = 1.0 / p
                    result = 0.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= p
                        result += f * (i_val % p)
                        i_val //= p
                    points[i, j] = result
            return points

        # Generate 8 initial points using Sobol/Halton in [-5, 5]^dim
        n_starts = 8
        halton_pts = sobol_sample(n_starts, dim, seed=42)
        starts = halton_pts * 10.0 - 5.0  # map to [-5, 5]

        # Evaluate all starts
        best_x = None
        best_f = np.inf
        start_results = []
        for s in starts:
            fs = safe_func(s)
            start_results.append((fs, s.copy()))
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        # Phase 2: Run Nelder-Mead from the best start point
        # Initialize simplex around best_x
        x0 = best_x
        f0 = best_f
        simplex = [x0.copy()]
        # Add perturbed points
        for j in range(dim):
            dx = x0.copy()
            # Use a reasonable step size
            step = 0.5
            if dx[j] + step > 5.0:
                step = -step
            dx[j] += step
            simplex.append(dx)

        # Evaluate simplex
        fvals = [f0]
        for i in range(1, dim + 1):
            fi = safe_func(simplex[i])
            fvals.append(fi)

        # Nelder-Mead iterations
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        max_nm_iter = 100
        min_step = 1e-8

        for nm_iter in range(max_nm_iter):
            if time.time() > deadline:
                break
            if budget - calls_used < 2:
                break

            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            fspread = max(fvals) - min(fvals)
            if fspread < 1e-10:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_func(xr)

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_func(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fvals[i] = safe_func(simplex[i])

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()

        # Phase 3: Polish with (1+1)-ES from the best point
        remaining = budget - calls_used
        if remaining > 0 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_es = 0.1  # smaller step for polishing

            for i in range(remaining):
                if time.time() > deadline:
                    break
                if budget - calls_used <= 0:
                    break

                y = x + sigma_es * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    best_f = fx
                    best_x = x.copy()
                    sigma_es *= 1.2
                else:
                    sigma_es *= 0.9

                sigma_es = np.clip(sigma_es, 1e-6, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
