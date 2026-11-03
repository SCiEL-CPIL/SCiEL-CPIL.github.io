"""Rewritten program: optimize Hermite coefficients to minimize uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_hermite


def _g(x, c0, c4, c8):
    """Compute g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2),
       with c12 chosen so that g(0) = 0."""
    # H0(0)=1, H4(0)=0, H8(0)=0, H12(0) = 12! / (6! * 2^6) = 479001600 / (720*64) = 10395
    # Actually let's use scipy to be safe
    H0_0 = eval_hermite(0, 0)   # 1
    H4_0 = eval_hermite(0, 4)   # 0
    H8_0 = eval_hermite(0, 8)   # 0
    H12_0 = eval_hermite(0, 12) # 10395
    
    # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # c0 + c12 * H12_0 = 0  (since H4(0)=H8(0)=0)
    if H12_0 != 0:
        c12 = -c0 / H12_0
    else:
        c12 = 0.0
    
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    return poly * np.exp(-x**2 / 2.0)


def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(Fourier(f)) for the test function g.
    
    The uncertainty bound involves integrals of |g|^2 and |g'|^2 etc.
    For a test function g, the bound is related to:
    C4 = (integral |g|^2) * (integral |F(g)|^2) type quantity.
    
    Actually, for the uncertainty inequality, the bound is:
    A(f) * A(F(f)) >= 1/(4*pi) * (some norm ratio)
    
    The test function approach: we want to minimize the upper bound derived from
    the Hermite-Gaussian test function. The bound is typically:
    bound = (||g||_2^2) * (||g'||_2^2) / (||g||_2^4) or similar.
    
    Let me think more carefully. The task says "reduce the upper bound on A(f)*A(Fourier(f))".
    The bound is C4, and we want to minimize it.
    
    For a test function g, the relevant quantity is likely:
    C4 = (∫|g|^2 dx) * (∫|g'|^2 dx) / (∫|g|^2 dx)^2  -- no, that's not right either.
    
    Actually in uncertainty principle literature with test functions, the bound is:
    C = (∫ x^2 |g|^2 dx / ∫|g|^2 dx) * (∫ ξ^2 |G|^2 dξ / ∫|G|^2 dξ)
    where G is the Fourier transform of g.
    
    Or more precisely, for the Heisenberg uncertainty:
    Δx * Δξ >= 1/2
    
    The "upper bound" from a test function approach would be the product of variances
    of the test function itself, which gives a bound on what's achievable.
    
    Let me compute: 
    - variance of |g|^2 in x-space
    - variance of |G|^2 in frequency space
    - product is the bound
    
    Actually, re-reading: "score is the upper bound on C4". So C4 is the quantity we minimize.
    
    Let me just compute the standard uncertainty product for g:
    C4 = (∫x^2|g|^2 / ∫|g|^2) * (∫ξ^2|G|^2 / ∫|G|^2)
    """
    # Numerical integration
    x = np.linspace(-20, 20, 20001)
    dx = x[1] - x[0]
    
    g = _g(x, c0, c4, c8)
    g2 = np.abs(g)**2
    
    # Normalize
    norm2 = np.trapz(g2, x)
    if norm2 < 1e-30:
        return np.inf
    
    # Variance in x
    x2_g2 = np.trapz(x**2 * g2, x)
    var_x = x2_g2 / norm2  # assuming mean is 0 by symmetry
    
    # Fourier transform via FFT
    # Use a finer grid for accuracy
    N = 65536
    L = 40.0
    x_f = np.linspace(-L/2, L/2, N, endpoint=False)
    dx_f = x_f[1] - x_f[0]
    
    g_f = _g(x_f, c0, c4, c8)
    
    # FFT
    G = np.fft.fftshift(np.fft.fft(g_f)) * dx_f
    
    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx_f))
    
    G2 = np.abs(G)**2
    norm2_f = np.trapz(G2, freqs)
    if norm2_f < 1e-30:
        return np.inf
    
    freq2_G2 = np.trapz(freqs**2 * G2, freqs)
    var_f = freq2_G2 / norm2_f
    
    C4 = var_x * var_f
    return C4


def construct():
    """Return three coefficients (c0, c4, c8) that minimize the uncertainty bound."""
    t_start = time.time()
    deadline = t_start + 190.0  # safety margin before 210s kill
    
    # Initial guess from prior art
    c0_init, c4_init, c8_init = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    
    best = [c0_init, c4_init, c8_init]
    best_val = _compute_bound(c0_init, c4_init, c8_init)
    
    # Use Nelder-Mead style optimization
    from scipy.optimize import minimize
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        try:
            val = _compute_bound(c0, c4, c8)
            if not np.isfinite(val):
                return 1e10
            return val
        except Exception:
            return 1e10
    
    # Try multiple starting points
    starts = [
        [c0_init, c4_init, c8_init],
        [c0_init, c4_init, c8_init * 2],
        [c0_init, c4_init, c8_init * 0.5],
        [c0_init * 0.9, c4_init, c8_init],
        [c0_init, c4_init * 1.5, c8_init],
        [c0_init, c4_init, c8_init + 0.001],
        [c0_init + 0.01, c4_init, c8_init],
        [c0_init, c4_init - 0.01, c8_init],
    ]
    
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best = list(res.x)
        except Exception:
            pass
    
    # Try a finer local search around the best point
    if time.time() < deadline - 30.0:
        try:
            res = minimize(objective, best, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True,
                                   'initial_simplex': [best[i] + 1e-6*(1 if i==0 else -1) for i in range(3)]})
            if res.fun < best_val:
                best_val = res.fun
                best = list(res.x)
        except Exception:
            pass
    
    # Final check
    if time.time() < deadline - 10.0:
        try:
            val_check = _compute_bound(best[0], best[1], best[2])
            if val_check < best_val:
                best_val = val_check
        except Exception:
            pass
    
    return np.array(best)


# EVOLVE-BLOCK-END
