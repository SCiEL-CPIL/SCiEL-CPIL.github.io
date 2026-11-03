"""Minimize the uncertainty upper bound C4 for a Hermite-Gaussian test function."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import eval_hermite

# Deadline guard: kill before 210s wall-clock, keep safety margin.
_START = time.time()
_DEADLINE = 205.0  # 5s margin before 210s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _a12_from_p0(a0, a4, a8):
    """Derive a12 so that g(0) = 0."""
    # H_n(0) for even n: H_0(0)=1, H_4(0)=-3, H_8(0)=105, H_12(0)=-10395
    # g(0) = a0*1 + a4*(-3) + a8*105 + a12*(-10395) = 0
    return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0


def _build_g(a0, a4, a8, x):
    """Build g(x) = (a0 H_0 + a4 H_4 + a8 H_8 + a12 H_12) * exp(-x^2/2)."""
    a12 = _a12_from_p0(a0, a4, a8)
    gauss = np.exp(-0.5 * x * x)
    return (a0 * eval_hermite(0, x) + a4 * eval_hermite(4, x)
            + a8 * eval_hermite(8, x) + a12 * eval_hermite(12, x)) * gauss


def _c4_bound(a0, a4, a8):
    """Compute C4 = ||g||_1 * ||g_hat||_1 with ||g||_2 = 1 normalization."""
    N = 4096
    x = np.linspace(-12.0, 12.0, N)
    dx = x[1] - x[0]
    
    g = _build_g(a0, a4, a8, x)
    
    # Normalize so ||g||_2 = 1
    norm2 = np.sqrt(np.sum(g * g) * dx)
    if norm2 < 1e-15:
        return 1e10
    g = g / norm2

    # L1 norm of g
    l1_g = np.sum(np.abs(g)) * dx

    # Fourier transform via FFT
    gft = np.fft.fftshift(np.fft.fft(g))
    # Unitary FFT scaling
    gft = gft * (dx / np.sqrt(2.0 * np.pi))
    l1_gft = np.sum(np.abs(gft)) * (2.0 * np.pi / N / dx)

    return l1_g * l1_gft


def _roots_to_coeffs(roots):
    """
    Convert roots of a cubic polynomial to Hermite coefficients.
    
    Parameterization: Let the three free coefficients be the roots of
    a monic cubic P(t) = (t - r1)(t - r2)(t - r3) = t^3 - s1 t^2 + s2 t - s3.
    We map (s1, s2, s3) to (a0, a4, a8) via a scaling that keeps values reasonable.
    
    Actually, the directive says: parameterize as coefficients of P(t) = t^3 + a t^2 + b t + c
    and minimize over the polynomial roots. Let me interpret this as:
    We use three parameters (r1, r2, r3) which are roots of P(t) = (t-r1)(t-r2)(t-r3).
    Then expand: P(t) = t^3 - (r1+r2+r3) t^2 + (r1 r2 + r1 r3 + r2 r3) t - r1 r2 r3
    So a = -(r1+r2+r3), b = r1 r2 + r1 r3 + r2 r3, c = -r1 r2 r3.
    
    We need a mapping from (a, b, c) or (r1, r2, r3) to (a0, a4, a8).
    A natural choice: use the elementary symmetric polynomials directly as the coefficients,
    scaled appropriately. Let's use:
      a0 = c  (constant term, up to sign)
      a4 = b  (linear coefficient)
      a8 = a  (quadratic coefficient)
    But we need to scale these to get reasonable magnitudes.
    
    Alternative interpretation: The roots themselves are the optimization variables,
    and we map them to coefficients via some fixed transformation. Let's use:
      a0 = r1
      a4 = r2
      a8 = r3
    with bounds on the roots. This is just a reparameterization but the key is
    that differential evolution works in the root space which may have better
    landscape properties.
    
    Let me try a more principled approach: use the roots to define the coefficients
    through the elementary symmetric polynomials with appropriate scaling.
    """
    r1, r2, r3 = roots
    # Elementary symmetric polynomials
    s1 = r1 + r2 + r3  # sum
    s2 = r1 * r2 + r1 * r3 + r2 * r3  # sum of pairwise products
    s3 = r1 * r2 * r3  # product
    
    # Map to Hermite coefficients with scaling to keep in reasonable range
    # The seed uses a0 ~ -1.13, a4 ~ 0.04, a8 ~ 0.0003
    # Scale factors to bring symmetric polynomials into that range
    a0 = s3 * 10.0  # product tends to be small
    a4 = s2 * 0.1   # pairwise products
    a8 = s1 * 0.01  # sum
    
    return a0, a4, a8


def _objective_roots(params):
    """Objective function in terms of polynomial roots."""
    try:
        a0, a4, a8 = _roots_to_coeffs(params)
        return _c4_bound(a0, a4, a8)
    except Exception:
        return 1e10


def construct():
    # Initial best from seed
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _c4_bound(*best_params)
    
    # Differential evolution over the root space
    # Bounds for roots: chosen to cover a reasonable range
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    try:
        de_result = differential_evolution(
            _objective_roots,
            bounds=bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True,
            callback=None
        )
        
        if de_result.fun < best_val:
            # Convert back to Hermite coefficients
            a0, a4, a8 = _roots_to_coeffs(de_result.x)
            best_val = de_result.fun
            best_params = np.array([a0, a4, a8])
    except Exception:
        pass
    
    # Refine with local optimization if time permits
    if _time_left() > 20.0:
        try:
            from scipy.optimize import minimize
            res = minimize(_c4_bound, best_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
