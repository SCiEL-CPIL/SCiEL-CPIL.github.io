import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0

        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x))
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # --- Phase 1: Space-filling exploration to identify the minimum region ---
        # Katsuura f23: f(x) = prod_{i=1}^{d} [1 + (x_i - 1)^2 * something]
        # Standard minimum at x = (1,1,...,1). With hidden affine transform:
        #   f_actual(x) = f_katsuura(R * (x - c) + s) + offset
        # where R is rotation, c is shift, s is scale, offset is additive.
        # The function has increasing oscillation frequency near the minimum.

        # Halton sequence for space-filling
        def halton_seq(n, dim):
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            points = []
            for i in range(1, n + 1):
                pt = []
                for d in range(dim):
                    base = primes[d % len(primes)]
                    f = 1.0
                    r = 0.0
                    idx = i
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    pt.append(r)
                points.append(pt)
            return np.array(points)

        # Generate Halton points mapped to [-5, 5]^dim
        n_halton = 60
        halton_pts = halton_seq(n_halton, dim) * 10.0 - 5.0

        # Also add some structured points
        structured = [
            np.zeros(dim),
            np.ones(dim),
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
        ]

        all_initial = structured + halton_pts.tolist()

        # Evaluate all initial points
        initial_results = []
        for pt in all_initial:
            if eval_count >= budget or time.time() > deadline:
                break
            val = counting_func(pt)
            if val != np.inf:
                initial_results.append((val, np.asarray(pt, dtype=float)))

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0
            return best_x, best_f

        # Sort by value
        initial_results.sort(key=lambda x: x[0])

        # --- Phase 2: Estimate affine transformation using Katsuura structure ---
        # The Katsuura function has a characteristic property: its minimum is at
        # x = (1,1,...,1) in standard coordinates. The oscillation frequency
        # increases as we approach this point. We can use the observed values
        # to estimate where the "effective" minimum is in our coordinate system.

        # Use the best few points to estimate the shift direction.
        # If we have enough good points, we can do a simple linear fit.
        top_k = min(10, len(initial_results))
        if top_k >= 3:
            top_points = np.array([p for _, p in initial_results[:top_k]])
            top_vals = np.array([v for v, _ in initial_results[:top_k]])

            # The Katsuura function is approximately quadratic near the minimum
            # (the oscillations are small relative to the main quadratic term).
            # Fit a quadratic model: f(x) ≈ a + b^T x + x^T Q x
            # The minimum of this quadratic is at x* = -0.5 * Q^{-1} b

            # Simple approach: use the centroid of top points as an initial
            # estimate of the minimum location
            min_estimate = np.mean(top_points, axis=0)

            # Refine with a local search around this estimate
            # Use CMA-ES with a moderate budget
            remaining = budget - eval_count
            if remaining > 100:
                # Run CMA-ES from the estimated minimum location
                try:
                    res = scipy.optimize.minimize(
                        counting_func,
                        min_estimate,
                        method='CMA',
                        options={
                            'maxiter': min(remaining // 15, 50),
                            'maxfev': min(remaining - 50, 300),
                            'popsize': 15,
                            'sigma0': 1.0,
                            'bounds': [(-5.0, 5.0)] * dim,
                            'verbose': 0,
                            'tol': 1e-8
                        }
                    )
                except Exception:
                    pass

        # --- Phase 3: Multi-start local optimization from diverse good points ---
        remaining = budget - eval_count
        if remaining > 50 and len(initial_results) >= 5:
            # Select top diverse points for multi-start
            selected = [initial_results[0][1].copy()]  # Best point
            for val, pt in initial_results[1:]:
                if len(selected) >= 4:
                    break
                # Check diversity: minimum distance to already selected
                min_dist = min(np.linalg.norm(pt - s) for s in selected)
                if min_dist > 1.0:
                    selected.append(pt.copy())

            # Allocate remaining budget among starts
            n_starts = len(selected)
            alloc_per_start = max(20, (remaining - n_starts * 5) // n_starts)

            for i, x_start in enumerate(selected):
                if eval_count >= budget or time.time() > deadline:
                    break
                local_budget = min(alloc_per_start, budget - eval_count)
                if local_budget < 15:
                    break

                # Use Nelder-Mead for local refinement
                try:
                    res_nm = scipy.optimize.minimize(
                        counting_func,
                        x_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': local_budget,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass

        # --- Phase 4: Final high-precision polish ---
        remaining = budget - eval_count
        if remaining > 20 and best_x is not None:
            # Use L-BFGS-B with numerical gradient for final polish
            try:
                res_lbfgs = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': remaining,
                        'ftol': 1e-15,
                        'gtol': 1e-15,
                        'maxls': 20
                    }
                )
            except Exception:
                pass

            # One more Nelder-Mead pass with very tight tolerance
            remaining = budget - eval_count
            if remaining > 10:
                try:
                    res_final = scipy.optimize.minimize(
                        counting_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass

        # --- Phase 5: Coordinate-wise fine-tuning if budget remains ---
        remaining = budget - eval_count
        if remaining > 5 and best_x is not None:
            for _ in range(3):  # Up to 3 passes
                if eval_count >= budget or time.time() > deadline:
                    break
                improved = False
                for coord in range(dim):
                    if eval_count >= budget or time.time() > deadline:
                        break
                    for step in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                        if eval_count >= budget or time.time() > deadline:
                            break
                        pt = best_x.copy()
                        pt[coord] += step
                        pt = np.clip(pt, -5, 5)
                        val = counting_func(pt)
                        if val < best_f:
                            improved = True
                if not improved:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0

        return best_x, best_f
