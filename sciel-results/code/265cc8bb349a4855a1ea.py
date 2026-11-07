"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic-aware optimizer for Sphere: Finite-difference Newton step + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        t_start = time.time()
        deadline = 290.0  # safety margin below the 300s wall-clock limit

        # Count function evaluations against the budget.
        calls = [0]

        def f(x):
            calls[0] += 1
            return func(x)

        # ---- Phase 1: Quadratic Model Estimation ----
        # Assume f(x) ≈ f0 + g^T x + 0.5 * lambda * ||x||^2
        # This holds for Sphere functions (possibly scaled/shifted/rotated) 
        # because the Hessian of a sphere is 2*I (or 2*scalar*I).
        
        x0 = np.zeros(dim)
        f0 = f(x0)
        best_x = x0.copy()
        best_fx = f0
        
        if calls[0] >= budget:
            return best_x, best_fx

        h = 0.1
        g = np.zeros(dim)
        curv_sum = 0.0
        
        # Central differences for gradient and curvature
        for i in range(dim):
            if calls[0] >= budget: break
            ei = np.zeros(dim)
            ei[i] = h
            x_plus = x0 + ei
            x_minus = x0 - ei
            
            fp = f(x_plus)
            if calls[0] >= budget: break
            fm = f(x_minus)
            
            g[i] = (fp - fm) / (2 * h)
            curv_sum += (fp + fm - 2 * f0) / (h * h)
            
            # Update best if we found something better during probing
            if fp < best_fx:
                best_fx, best_x = fp, x_plus.copy()
            if fm < best_fx:
                best_fx, best_x = fm, x_minus.copy()

        if curv_sum > 1e-10:
            lambda_est = curv_sum / dim
            # Newton step: z = x0 - g / lambda
            z_est = x0 - g / lambda_est
            z_est = np.clip(z_est, -5.0, 5.0)
            
            if calls[0] < budget:
                fz = f(z_est)
                if fz < best_fx:
                    best_fx, best_x = fz, z_est.copy()
        
        # ---- Phase 2: Nelder-Mead Polish ----
        # Start NM from the best point found so far
        x = best_x.copy()
        fx = best_fx
        
        # Build initial simplex around best_x
        # Use a small spread since we are likely close
        simplex = [x.copy()]
        for i in range(dim):
            v = x.copy()
            v[i] = x[i] + 0.5
            simplex.append(np.clip(v, -5.0, 5.0))
            
        fvals = [fx]
        for i in range(1, dim + 1):
            if calls[0] >= budget:
                break
            fvals.append(f(simplex[i]))
            if fvals[-1] < best_fx:
                best_fx, best_x = fvals[-1], simplex[i].copy()

        alpha, gamma, rho, sigma = 1.0, 2.0, -1.0, 0.5

        def update_best(s, fv):
            nonlocal best_fx, best_x
            if fv < best_fx:
                best_fx, best_x = fv, s.copy()

        # Nelder-Mead iterations
        while calls[0] < budget and (time.time() - t_start) < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            xw = simplex[-1]     # worst
            xc = np.mean(simplex[:-1], axis=0)  # centroid of all but worst
            fw = fvals[-1]
            fb = fvals[0]

            if calls[0] >= budget:
                break

            # Reflection
            xr = np.clip(xc + alpha * (xc - xw), -5.0, 5.0)
            fr = f(xr)
            update_best(xr, fr)

            if fb <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fb:
                # Expansion
                if calls[0] >= budget:
                    break
                xe = np.clip(xc + gamma * (xc - xw), -5.0, 5.0)
                fe = f(xe)
                update_best(xe, fe)
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                # Contraction
                if fr < fw:
                    xs = np.clip(xc + rho * (xw - xc), -5.0, 5.0)
                else:
                    xs = np.clip(xc + sigma * (xw - xc), -5.0, 5.0)
                if calls[0] >= budget:
                    break
                fs = f(xs)
                update_best(xs, fs)
                if fs < min(fw, fr):
                    simplex[-1] = xs
                    fvals[-1] = fs
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5.0, 5.0)
                        if calls[0] >= budget:
                            break
                        fvals[i] = f(simplex[i])
                        update_best(simplex[i], fvals[i])
                    if calls[0] >= budget:
                        break
                        
        return best_x, best_fx
# EVOLVE-BLOCK-END
