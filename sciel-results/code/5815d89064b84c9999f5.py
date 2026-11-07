"""Optimizer for BBOB-style f17 (Schaffers F7) using CMA-ES with adaptive restarts and SciPy polish."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES based optimizer with adaptive initial sampling and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_left():
            return deadline - time.time()

        def can_continue():
            return calls_used < budget and time_left() > 1.0

        # Phase 1: Initial sampling using Halton sequence for better coverage
        # Halton sequence is a low-discrepancy sequence, good for initial space filling.
        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for d in range(dim):
                    f = 1.0
                    i_val = i + 1
                    base = d + 2  # Base 2 for first dim, 3 for second, etc.
                    while i_val > 0:
                        f /= base
                        i_val = int(i_val / base)
                    points[i, d] = f
            # Map [0, 1] to [-5, 5]
            return -5 + 10 * points

        n_initial = 40
        halton_pts = halton_sequence(n_initial, dim)
        # Add some random points to break any symmetry or bias in Halton
        random_pts = rng.uniform(-5, 5, (n_initial, dim))
        
        # Combine and evaluate
        # We evaluate in batches to potentially speed up if func is vectorized? 
        # The contract says func(x) returns a float for a vector x. It doesn't explicitly say it's vectorized.
        # "func(x) clips a finite length-5 vector... and returns a float." 
        # It's safer to assume scalar calls.
        
        for i in range(n_initial):
            if not can_continue():
                break
            x = halton_pts[i]
            f = func(x)
            calls_used += 1
            update_best(x, f)
            
        for i in range(n_initial):
            if not can_continue():
                break
            x = random_pts[i]
            f = func(x)
            calls_used += 1
            update_best(x, f)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: CMA-ES
        def run_cmaes(x_start, sigma_init, max_calls):
            nonlocal calls_used, best_x, best_f

            # CMA-ES parameters for 5D
            lam = 12  # Population size
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            # Step size adaptation
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (2 + mueff / dim) / (dim + cc + 2)
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) + cc

            # Covariance matrix adaptation
            c1 = (mueff - 2 + 1 / dim) / ((dim + 2) ** 2 - mueff)
            cmu = min(1 - cc, 2 / ((dim + 1.3) ** 2 + mueff))
            cmu = min(cmu, (1 - 2 * c1) / 2)

            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # State
            x_mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            no_improve = 0
            best_local_f = best_f
            local_calls = 0

            while local_calls < max_calls:
                if not can_continue():
                    break

                # Sample new points
                ys = rng.standard_normal((lam, dim))
                xs = np.array([x_mean + sigma * (ys[i] @ C_sqrt) for i in range(lam)])
                xs = np.clip(xs, -5, 5)

                # Evaluate
                fs = np.zeros(lam)
                for i in range(lam):
                    if not can_continue():
                        break
                    fs[i] = func(xs[i])
                    calls_used += 1
                    local_calls += 1
                    update_best(xs[i], fs[i])

                if not can_continue():
                    break

                # Sort
                order = np.argsort(fs)
                xs = xs[order]
                fs = fs[order]
                ys = ys[order]

                # Check improvement
                if fs[0] < best_local_f - 1e-10:
                    no_improve = 0
                    best_local_f = fs[0]
                else:
                    no_improve += 1
                    if no_improve > 15: # Slightly more aggressive restart
                        # Restart with larger sigma
                        x_mean = best_x.copy()
                        sigma = max(sigma * 1.5, 0.5)
                        C = np.eye(dim)
                        C_sqrt = np.eye(dim)
                        pc = np.zeros(dim)
                        ps = np.zeros(dim)
                        no_improve = 0
                        continue

                # Update mean
                x_mean_new = np.sum(weights[:, None] * xs[:mu], axis=0)

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt.T @ ((x_mean_new - x_mean) / sigma))
                sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-7, 8.0)

                # Update pc
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma

                # Update covariance
                ys_parents = ys[:mu]
                S_w = np.sum(weights[:, None, None] * (ys_parents[:, :, None] * ys_parents[:, None, :]), axis=0)
                C_new = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * C) + cmu * S_w
                C_new = (C_new + C_new.T) / 2.0

                try:
                    C_sqrt = np.linalg.cholesky(C_new)
                except np.linalg.LinAlgError:
                    C_new += 1e-8 * np.eye(dim)
                    try:
                        C_sqrt = np.linalg.cholesky(C_new)
                    except:
                        C_sqrt = np.eye(dim)
                        break

                C = C_new
                x_mean = x_mean_new

            return local_calls

        # Run main CMA-ES
        remaining = budget - calls_used
        run_cmaes(best_x.copy(), 1.5, remaining)

        # Phase 3: Local polish using SciPy Nelder-Mead
        if best_x is not None and can_continue():
            # Use Nelder-Mead for final refinement
            # Bounds are handled by clipping in func, but Nelder-Mead doesn't natively support bounds well.
            # We can use Powell or just Nelder-Mead and rely on clipping.
            # Nelder-Mead is robust.
            try:
                result = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 100, 'maxfev': budget - calls_used}
                )
                calls_used += result.nfev
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_used < budget:
                best_f = func(best_x)
                calls_used += 1

        return best_x, best_f
