"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 170.0  # Safety margin before 180s limit

    # Hermite polynomial values at 0
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 1039500.0

    # Effective n for <x^2> of psi_n: n + 1/2
    n0 = 0.5
    n4 = 4.5
    n8 = 8.5
    n12 = 12.5

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    # ------------------------------------------------------------------
    # L_infinity objective: minimize the peak of the spectral density
    #   S(x) = sum_i w_i * (Hermite-Gaussian profile)
    # We map to a bounded interval via u = x^2 (u in [0, inf)) and use
    # a log-sum-exp soft-max as a smooth surrogate for the max over a
    # finite grid of u.  The residual polynomial after the u-substitution
    # is what we drive to L_infinity minimality (Remez spirit).
    # ------------------------------------------------------------------
    # Grid of u = x^2 values on a bounded interval [0, Umax]
    Umax = 30.0
    u_grid = np.linspace(0.0, Umax, 2001)

    # Precompute the Hermite-Gaussian spectral weight for each mode as a
    # function of u.  For a normalized Hermite function psi_n, the
    # position-space density is |psi_n(x)|^2 ~ H_n(x)^2 e^{-x^2}.
    # In terms of u = x^2, e^{-x^2} = e^{-u}.  H_n(x)^2 for even n is a
    # polynomial in u of degree n/2.  We evaluate on the grid.
    def _hermite_even_sq(n, x):
        # Physicists' Hermite H_n(x) for even n, squared, as poly in u=x^2
        # Use explicit low-degree forms to stay fast and exact.
        u = x**2
        if n == 0:
            return np.ones_like(u)
        if n == 4:
            # H_4(x) = 16x^4 - 48x^2 + 12
            h4 = 16*u**2 - 48*u + 12
            return h4**2
        if n == 8:
            # H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 53760x^2 + 1680
            h8 = 256*u**4 - 4480*u**3 + 26880*u**2 - 53760*u + 1680
            return h8**2
        if n == 12:
            # H_12(x): use recurrence H_{n+1} = 2x H_n - 2n H_{n-1}
            # Build up to 12 on the grid (x = sqrt(u), sign irrelevant for square)
            x = np.sqrt(u)
            H = np.zeros((13,) + u.shape, dtype=float)
            H[0] = 1.0
            H[1] = 2.0 * x
            for k in range(1, 12):
                H[k+1] = 2.0 * x * H[k] - 2.0 * k * H[k-1]
            return H[12]**2
        raise ValueError("unsupported n")

    # Precompute profile P_i(u) = H_i(x)^2 * exp(-u)  on the grid
    P0 = _hermite_even_sq(0, np.sqrt(u_grid)) * np.exp(-u_grid)
    P4 = _hermite_even_sq(4, np.sqrt(u_grid)) * np.exp(-u_grid)
    P8 = _hermite_even_sq(8, np.sqrt(u_grid)) * np.exp(-u_grid)
    P12 = _hermite_even_sq(12, np.sqrt(u_grid)) * np.exp(-u_grid)

    # Soft-max temperature for the smooth L_infinity surrogate
    def soft_max(vals, beta):
        m = np.max(vals)
        return m + np.log(np.mean(np.exp(beta * (vals - m)))) / beta

    def objective(coeffs, beta=80.0):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        w0 = c0**2
        w4 = c4**2
        w8 = c8**2
        w12 = c12**2
        total_w = w0 + w4 + w8 + w12
        if total_w < 1e-15:
            return 1e15
        # Normalize weights so the sum is 1 (scale invariant)
        w0 /= total_w; w4 /= total_w; w8 /= total_w; w12 /= total_w
        # Spectral density on grid
        S = w0 * P0 + w4 * P4 + w8 * P8 + w12 * P12
        return soft_max(S, beta)

    # Seed coefficients
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    best_coeffs = seed_coeffs.copy()
    best_score = objective(best_coeffs)

    try:
        from scipy.optimize import minimize
        from scipy.optimize import differential_evolution

        # 1) Global search with differential evolution on the smooth surrogate
        if time.time() < deadline - 60:
            bounds = [(-1.5, 1.5), (-0.5, 0.5), (-0.1, 0.1)]
            de_result = differential_evolution(
                objective, bounds,
                maxiter=600, tol=1e-8,
                popsize=15, seed=42,
                polish=True
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x.copy()

        # 2) Remez-style exchange: iteratively raise beta and re-localize,
        #    which sharpens the soft-max toward the true L_infinity peak.
        if time.time() < deadline - 15:
            for beta in [80.0, 200.0, 500.0, 1200.0]:
                if time.time() > deadline - 20:
                    break
                def obj_beta(c, b=beta):
                    return objective(c, beta=b)
                res = minimize(obj_beta, best_coeffs, method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-13, 'fatol': 1e-15})
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x.copy()

        # 3) Multi-start local refinement around the best point
        if time.time() < deadline - 10:
            start_points = [best_coeffs.copy(), seed_coeffs.copy()]
            for scale in [0.01, 0.05, 0.1]:
                for _ in range(3):
                    noise = np.random.normal(0, scale, size=3)
                    start_points.append(best_coeffs + noise)
            for x0 in start_points:
                if time.time() > deadline - 15:
                    break
                result = minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 20000, 'xatol': 1e-13, 'fatol': 1e-15})
                if result.fun < best_score:
                    best_score = result.fun
                    best_coeffs = result.x.copy()

    except Exception:
        pass

    # Final high-precision polish with a very sharp soft-max
    if time.time() < deadline - 2:
        try:
            def obj_final(c):
                return objective(c, beta=3000.0)
            final_polish = minimize(obj_final, best_coeffs, method='Nelder-Mead',
                                    options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-17})
            if final_polish.fun < best_score:
                best_score = final_polish.fun
                best_coeffs = final_polish.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
