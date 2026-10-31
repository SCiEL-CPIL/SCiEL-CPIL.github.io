"""Optimize Hermite-Gaussian test function coefficients to reduce uncertainty bound."""
import numpy as np
from scipy.optimize import minimize
import time

# Physicist's Hermite polynomials evaluated at x
def _H(n, x):
    if n == 0: return np.ones_like(x)
    if n == 1: return 2*x
    # recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    Hn_1 = np.ones_like(x)
    Hn = 2*x
    for k in range(1, n):
        Hn_2 = Hn_1
        Hn_1 = Hn
        Hn = 2*x*Hn - 2*k*Hn_1
    return Hn

def _coeffs_to_g(c, x):
    c0, c4, c8 = c
    # H12(0) value
    H12_0 = 47900160.0
    c12 = -(c0 + 12.0*c4 + 1680.0*c8) / H12_0
    H0 = _H(0, x); H4 = _H(4, x); H8 = _H(8, x); H12 = _H(12, x)
    poly = c0*H0 + c4*H4 + c8*H8 + c12*H12
    return poly * np.exp(-0.5*x**2)

def _objective(c):
    x = np.linspace(-30, 30, 20001)
    dx = x[1]-x[0]
    g = _coeffs_to_g(c, x)
    # Fourier transform: for H_n e^{-x^2/2}, FT is (i)^n H_n e^{-xi^2/2} (unitary convention)
    # |g_hat(xi)| = |poly(xi)| e^{-xi^2/2}  (same magnitude)
    gh = _coeffs_to_g(c, x)  # same form by eigenfunction property
    # A(f) ~ (integral |g|^4)^{1/4} type; bound on C4 uses L4 norms
    norm_g = np.sum(np.abs(g)**4) * dx
    norm_gh = np.sum(np.abs(gh)**4) * dx
    # C4 bound proportional to (norm_g * norm_gh)^{1/4} or similar; use product of L4 norms
    return (norm_g * norm_gh) ** 0.25

def construct():
    start = time.time()
    DEADLINE = 180.0
    SAFETY = 20.0
    best_c = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _objective(best_c)

    # Multi-start local optimization
    rng = np.random.default_rng(0)
    starts = [best_c.copy()]
    # Add perturbed starts
    for scale in [0.5, 1.0, 2.0]:
        for _ in range(8):
            starts.append(best_c + scale * rng.normal(size=3))

    for s in starts:
        if time.time() - start > DEADLINE - SAFETY:
            break
        try:
            res = minimize(_objective, s, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol':1e-8, 'fatol':1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass

    # Final polish with tighter tolerance
    if time.time() - start < DEADLINE - SAFETY:
        try:
            res = minimize(_objective, best_c, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol':1e-10, 'fatol':1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass

    return np.array(best_c)
