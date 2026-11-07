"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic-aware optimizer for Sphere: finite-difference Newton + Nelder-Mead fallback."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        t_start = time.time()
        deadline = 285.0  # safety margin below the 300s wall-clock limit

        calls = [0]

        def f(x):
            calls[0] += 1
            return func(x)

        def update_best(x, fx, best_x, best_fx):
            if fx < best_fx:
                return x.copy(), fx
            return best_x, best_fx

        # Start at center (minimizer is in [-4,4]^5)
        x0 = np.zeros(dim)
        fx0 = f(x0)
        best_x = x0.copy()
        best_fx = fx0

        # ---- Phase 1: Estimate gradient and Hessian via finite differences ----
        # For a quadratic f(x) = 0.5*(x-c)^T A (x-c) + f0, we can estimate
        # the minimizer c and matrix A from function evaluations.
        
        # Step 1: Estimate gradient at x0 using central differences with step h
        h = 0.01
        grad = np.zeros(dim)
        for i in range(dim):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += h
            x_minus[i] -= h
            x_plus = np.clip(x_plus, -5.0, 5.0)
            x_minus = np.clip(x_minus, -5.0, 5.0)
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
            if calls[0] >= budget:
                break

        # If we've used up budget, return what we have
        if calls[0] >= budget:
            return best_x, best_fx

        # Step 2: Estimate Hessian using central second differences
        # For quadratic, Hessian is constant, so one estimate suffices
        hess = np.zeros((dim, dim))
        h2 = 0.1  # larger step for second derivatives
        for i in range(dim):
            for j in range(i, dim):
                if calls[0] >= budget:
                    break
                xi = x0.copy()
                xj = x0.copy()
                xii = x0.copy()
                xjj = x0.copy()
                
                xi[i] += h2
                xj[j] += h2
                xii[i] -= h2
                xjj[j] -= h2
                
                xi = np.clip(xi, -5.0, 5.0)
                xj = np.clip(xj, -5.0, 5.0)
                xii = np.clip(xii, -5.0, 5.0)
                xjj = np.clip(xjj, -5.0, 5.0)
                
                # f(x0 + h2*e_i + h2*e_j) + f(x0 - h2*e_i - h2*e_j) - f(x0 + h2*e_i - h2*e_j) - f(x0 - h2*e_i + h2*e_j)
                # This is 2*h2^2 * Hessian[i,j] for a quadratic
                
                xpp = x0.copy()
                xpp[i] += h2
                xpp[j] += h2
                xpp = np.clip(xpp, -5.0, 5.0)
                
                xmm = x0.copy()
                xmm[i] -= h2
                xmm[j] -= h2
                xmm = np.clip(xmm, -5.0, 5.0)
                
                xpm = x0.copy()
                xpm[i] += h2
                xpm[j] -= h2
                xpm = np.clip(xpm, -5.0, 5.0)
                
                xmp = x0.copy()
                xmp[i] -= h2
                xmp[j] += h2
                xmp = np.clip(xmp, -5.0, 5.0)
                
                if calls[0] >= budget:
                    break
                fpp = f(xpp)
                fmm = f(xmm)
                fpm = f(xpm)
                fmp = f(xmp)
                
                hess_val = (fpp + fmm - fpm - fmp) / (4 * h2 * h2)
                hess[i, j] = hess_val
                hess[j, i] = hess_val
        
        if calls[0] >= budget:
            return best_x, best_fx

        # Step 3: Solve for minimizer: x* = x0 - H^{-1} * grad
        # For quadratic: grad(x) = H(x - c), so c = x0 - H^{-1} grad(x0)
        try:
            # Regularize Hessian for numerical stability
            eigvals, eigvecs = np.linalg.eigh(hess)
            # Clip eigenvalues to be positive (Sphere should have positive definite Hessian)
            eigvals_clipped = np.maximum(eigvals, 1e-6)
            hess_inv = eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T
            x_star = x0 - hess_inv @ grad
            x_star = np.clip(x_star, -5.0, 5.0)
            
            fx_star = f(x_star)
            best_x, best_fx = update_best(x_star, fx_star, best_x, best_fx)
        except np.linalg.LinAlgError:
            pass

        if calls[0] >= budget:
            return best_x, best_fx

        # ---- Phase 2: Nelder-Mead refinement from best point ----
        # Build simplex around best_x
        x = best_x.copy()
        simplex = [x.copy()]
        for i in range(dim):
            v = x.copy()
            v[i] += 1.0
            v = np.clip(v, -5.0, 5.0)
            simplex.append(v)
        
        fvals = []
        for s in simplex:
            if calls[0] >= budget:
                break
            fv = f(s)
            fvals.append(fv)
            best_x, best_fx = update_best(s, fv, best_x, best_fx)
        
        if len(fvals) != dim + 1:
            return best_x, best_fx

        alpha, gamma, rho, sigma = 1.0, 2.0, -1.0, 0.5

        while calls[0] < budget and (time.time() - t_start) < deadline:
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            xw = simplex[-1]
            xc = np.mean(simplex[:-1], axis=0)
            fw = fvals[-1]
            fb = fvals[0]

            if calls[0] >= budget:
                break

            xr = np.clip(xc + alpha * (xc - xw), -5.0, 5.0)
            fr = f(xr)
            best_x, best_fx = update_best(xr, fr, best_x, best_fx)

            if fb <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fb:
                if calls[0] >= budget:
                    break
                xe = np.clip(xc + gamma * (xc - xw), -5.0, 5.0)
                fe = f(xe)
                best_x, best_fx = update_best(xe, fe, best_x, best_fx)
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                if fr < fw:
                    xs = np.clip(xc + rho * (xw - xc), -5.0, 5.0)
                else:
                    xs = np.clip(xc + sigma * (xw - xc), -5.0, 5.0)
                if calls[0] >= budget:
                    break
                fs = f(xs)
                best_x, best_fx = update_best(xs, fs, best_x, best_fx)
                if fs < min(fw, fr):
                    simplex[-1] = xs
                    fvals[-1] = fs
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5.0, 5.0)
                        if calls[0] >= budget:
                            break
                        fvals[i] = f(simplex[i])
                        best_x, best_fx = update_best(simplex[i], fvals[i], best_x, best_fx)
                    if calls[0] >= budget:
                        break

        # ---- Phase 3: (1+1)-ES polish ----
        rng = np.random.default_rng(42)
        x = best_x.copy()
        fx = best_fx
        sigma = 0.1
        while calls[0] < budget and (time.time() - t_start) < deadline:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = f(y)
            if fy < fx:
                x, fx = y, fy
                best_x, best_fx = x.copy(), fy
                sigma *= 1.2
            else:
                sigma *= 0.8
                if sigma < 1e-6:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
