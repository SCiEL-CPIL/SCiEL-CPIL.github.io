"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with periodic Newton-style refinement exploiting quadratic structure."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(12345)

        def elapsed_ok():
            return time.time() < deadline

        # --- Phase 1: Nelder-Mead simplex search ---
        # Initialize simplex around origin (shifted minimizer is in [-4,4]^5)
        step = 1.5
        simplex = np.zeros((dim + 1, dim))
        for i in range(dim):
            simplex[i + 1, i] = step
        simplex[0] = 0.0

        # Evaluate simplex
        fvals = np.array([func(simplex[i]) for i in range(dim + 1)])
        calls = dim + 1
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5

        nm_iter = 0
        while calls < budget and elapsed_ok():
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            best_idx = 0
            worst_idx = dim
            best_f = fvals[best_idx]

            if fvals[worst_idx] - fvals[best_idx] < 1e-10:
                break

            # Centroid of all but worst
            centroid = simplex[:dim].mean(axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[worst_idx])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1

            if fr < fvals[best_idx]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[worst_idx])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[worst_idx] = xe
                    fvals[worst_idx] = fe
                else:
                    simplex[worst_idx] = xr
                    fvals[worst_idx] = fr
            elif fr < fvals[dim - 1]:
                simplex[worst_idx] = xr
                fvals[worst_idx] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[worst_idx] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < fvals[worst_idx]:
                    simplex[worst_idx] = xc
                    fvals[worst_idx] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        if calls < budget and elapsed_ok():
                            fvals[i] = func(simplex[i])
                            calls += 1
                    # Re-evaluate point 0 if needed (it wasn't changed)
                    # Actually point 0 wasn't changed, so fvals[0] is still valid

            # Track best
            cur_best_idx = np.argmin(fvals)
            if fvals[cur_best_idx] < best_f:
                best_f = fvals[cur_best_idx]
                best_x = simplex[cur_best_idx].copy()

            nm_iter += 1
            if nm_iter > 5000:
                break

        # --- Phase 2: Local Newton-style refinement ---
        # Fit a local quadratic model at the best point and take Newton steps
        if calls < budget and elapsed_ok():
            x = best_x.copy()
            fx = best_f
            h = 0.05  # finite difference step

            for newton_iter in range(20):
                if calls >= budget or not elapsed_ok():
                    break

                # Compute gradient and Hessian via central differences
                grad = np.zeros(dim)
                H = np.zeros((dim, dim))

                for i in range(dim):
                    # Gradient component i
                    xp = x.copy()
                    xp[i] += h
                    xp = np.clip(xp, -5, 5)
                    xm = x.copy()
                    xm[i] -= h
                    xm = np.clip(xm, -5, 5)
                    fp = func(xp)
                    fm = func(xm)
                    calls += 2
                    grad[i] = (fp - fm) / (2 * h)

                    # Hessian row i
                    for j in range(i, dim):
                        xpp = x.copy()
                        xpp[i] += h
                        xpp[j] += h
                        xpp = np.clip(xpp, -5, 5)
                        xpm = x.copy()
                        xpm[i] += h
                        xpm[j] -= h
                        xpm = np.clip(xpm, -5, 5)
                        xmp = x.copy()
                        xmp[i] -= h
                        xmp[j] += h
                        xmp = np.clip(xmp, -5, 5)
                        xmm = x.copy()
                        xmm[i] -= h
                        xmm[j] -= h
                        xmm = np.clip(xmm, -5, 5)
                        fpp = func(xpp)
                        fpm = func(xpm)
                        fmp = func(xmp)
                        fmm = func(xmm)
                        calls += 4
                        H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h * h)
                        H[j, i] = H[i, j]

                # Solve (H + lambda*I) dx = -grad for Newton step
                # Add damping to ensure positive definiteness
                lambda_val = 1e-6
                for attempt in range(10):
                    try:
                        A = H + lambda_val * np.eye(dim)
                        dx = np.linalg.solve(A, -grad)
                        break
                    except np.linalg.LinAlgError:
                        lambda_val *= 10
                else:
                    # Fallback: steepest descent
                    dx = -grad / (np.linalg.norm(grad) + 1e-12)

                # Limit step size
                step_norm = np.linalg.norm(dx)
                if step_norm > 1.0:
                    dx = dx / step_norm

                x_new = np.clip(x + dx, -5, 5)
                fx_new = func(x_new)
                calls += 1

                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Step didn't help; try half step
                    x_half = np.clip(x + 0.5 * dx, -5, 5)
                    fx_half = func(x_half)
                    calls += 1
                    if fx_half < fx:
                        x = x_half
                        fx = fx_half
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
