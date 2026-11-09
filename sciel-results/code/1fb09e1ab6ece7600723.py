import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 200.0  # 200s limit, keep safety margin before 210s kill

    # Pre-compute Hermite polynomial coefficients using numpy.polynomial.hermite
    from numpy.polynomial.hermite import herm2poly

    H0_poly = herm2poly(0)
    H4_poly = herm2poly(4)
    H8_poly = herm2poly(8)
    H12_poly = herm2poly(12)

    # Values at 0 for the constraint P(0) = 0
    H0_0 = H0_poly[0]
    H4_0 = H4_poly[0]
    H8_0 = H8_poly[0]
    H12_0 = H12_poly[0]

    def get_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def eval_poly(coeffs, x):
        val = 0.0
        for i in range(len(coeffs)):
            val += coeffs[i] * (x ** i)
        return val

    def g_func(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        h0 = 1.0
        h4 = eval_poly(H4_poly, x)
        h8 = eval_poly(H8_poly, x)
        h12 = eval_poly(H12_poly, x)
        p = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
        return np.exp(-x**2 / 2.0) * p

    def norm_4_sq(c0, c4, c8):
        def integrand(x):
            val = g_func(x, c0, c4, c8)
            return val**4
        val, err = quad(integrand, -np.inf, np.inf, limit=200)
        return val

    def fourier_g(xi, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        # For even k, (-i)^k = (-1)^(k/2).
        # k=0: 1, k=4: 1, k=8: 1, k=12: 1.
        # So the polynomial in the Fourier domain is the same combination.
        p_hat = c0 * 1.0 + c4 * eval_poly(H4_poly, xi) + c8 * eval_poly(H8_poly, xi) + c12 * eval_poly(H12_poly, xi)
        return np.exp(-xi**2 / 2.0) * p_hat

    def norm_4_sq_hat(c0, c4, c8):
        def integrand(xi):
            val = fourier_g(xi, c0, c4, c8)
            return val**4
        val, err = quad(integrand, -np.inf, np.inf, limit=200)
        return val

    def objective(c):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = c
        n1 = norm_4_sq(c0, c4, c8)
        n2 = norm_4_sq_hat(c0, c4, c8)
        if n1 < 0 or n2 < 0:
            return 1e10
        return (n1 * n2) ** 0.25

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    best_result = x0
    best_val = objective(x0)

    # Check time
    if time.time() - start_time > deadline - 30:
        return best_result

    # Local optimization from seed
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    except Exception:
        pass

    # Try a few diverse starting points to escape local minima
    if time.time() < deadline - 15:
        rng = np.random.default_rng(42)
        # Scaled random restarts
        for _ in range(10):
            if time.time() > deadline:
                break
            x_rand = x0 + rng.normal(0, 0.1, 3)
            try:
                res_r = minimize(objective, x_rand, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                if res_r.fun < best_val:
                    best_val = res_r.fun
                    best_result = res_r.x
            except Exception:
                pass

    # Final high-precision polish pass
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(objective, best_result, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_result = res_polish.x
        except Exception:
            pass

    return best_result
