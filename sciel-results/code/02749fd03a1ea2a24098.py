"""Rewritten program: numerical optimization of Hermite coefficients for uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin for 210s wall-clock limit

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def hermite_p(x, n):
        """Compute physicists' Hermite polynomial H_n(x) for n in {0,4,8,12}."""
        if n == 0:
            return np.ones_like(x)
        elif n == 4:
            return 16.0 * x**4 - 48.0 * x**2 + 12.0
        elif n == 8:
            return 256.0 * x**8 - 2304.0 * x**6 + 5760.0 * x**4 - 4320.0 * x**2 + 504.0
        elif n == 12:
            return (4096.0 * x**12 - 55296.0 * x**10 + 303104.0 * x**8
                    - 589824.0 * x**6 + 408192.0 * x**4 - 77760.0 * x**2 + 10395.0)
        else:
            raise ValueError(f"Unsupported n={n}")

    def objective(coeffs):
        """
        Compute the upper bound on C4 for the given Hermite coefficients.
        
        The test function is g(x) = P(x) * exp(-x^2/2) where
        P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
        
        The uncertainty bound involves the L4 norm of g and its Fourier transform.
        For the Fourier transform, we use the property that F[H_n(x)*exp(-x^2/2)]
        = i^n * H_n(x) * exp(-x^2/2) (up to normalization).
        
        The bound on C4 is proportional to ||g||_4 * ||F(g)||_4 / (||g||_2 * ||F(g)||_2)
        or similar. Let's use a standard form.
        """
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)

        # Define g(x) and F(g)(x) on the real line
        # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
        # F(g)(x) = (a0*H0 + a4*i^4*H4 + a8*i^8*H8 + a12*i^12*H12) * exp(-x^2/2)
        # Note: i^4 = 1, i^8 = 1, i^12 = -1
        # So F(g)(x) = (a0*H0 + a4*H4 + a8*H8 - a12*H12) * exp(-x^2/2)

        def g_func(x):
            H0 = 1.0
            H4 = 16.0 * x**4 - 48.0 * x**2 + 12.0
            H8 = 256.0 * x**8 - 2304.0 * x**6 + 5760.0 * x**4 - 4320.0 * x**2 + 504.0
            H12 = (4096.0 * x**12 - 55296.0 * x**10 + 303104.0 * x**8
                   - 589824.0 * x**6 + 408192.0 * x**4 - 77760.0 * x**2 + 10395.0)
            poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
            return poly * np.exp(-x**2 / 2.0)

        def fg_func(x):
            H0 = 1.0
            H4 = 16.0 * x**4 - 48.0 * x**2 + 12.0
            H8 = 256.0 * x**8 - 2304.0 * x**6 + 5760.0 * x**4 - 4320.0 * x**2 + 504.0
            H12 = (4096.0 * x**12 - 55296.0 * x**10 + 303104.0 * x**8
                   - 589824.0 * x**6 + 408192.0 * x**4 - 77760.0 * x**2 + 10395.0)
            poly = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12
            return poly * np.exp(-x**2 / 2.0)

        # Compute L2 norms
        try:
            norm2_g_sq, _ = quad(lambda x: g_func(x)**2, -np.inf, np.inf, limit=200)
            norm2_fg_sq, _ = quad(lambda x: fg_func(x)**2, -np.inf, np.inf, limit=200)
        except Exception:
            return 1e10

        if norm2_g_sq < 1e-15 or norm2_fg_sq < 1e-15:
            return 1e10

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_fg = np.sqrt(norm2_fg_sq)

        # Compute L4 norms
        try:
            norm4_g_4, _ = quad(lambda x: g_func(x)**4, -np.inf, np.inf, limit=200)
            norm4_fg_4, _ = quad(lambda x: fg_func(x)**4, -np.inf, np.inf, limit=200)
        except Exception:
            return 1e10

        if norm4_g_4 < 1e-15 or norm4_fg_4 < 1e-15:
            return 1e10

        norm4_g = norm4_g_4 ** 0.25
        norm4_fg = norm4_fg_4 ** 0.25

        # The bound on C4 is related to the product of L4 norms divided by product of L2 norms
        # This is a standard form for uncertainty-type bounds
        bound = (norm4_g * norm4_fg) / (norm2_g * norm2_fg)
        
        return bound

    best_coeffs = seed.copy()
    best_val = objective(seed)

    # Multi-start local optimization
    n_starts = 20
    rng = np.random.RandomState(42)

    for i in range(n_starts):
        if time.time() > deadline:
            break

        if i == 0:
            x0 = seed.copy()
        else:
            # Random start in a reasonable neighborhood
            scale = 0.5 / (1.0 + i * 0.2)
            x0 = seed + rng.normal(0, scale, size=3)

        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Refine the best solution with more iterations
    if time.time() < deadline - 20.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
