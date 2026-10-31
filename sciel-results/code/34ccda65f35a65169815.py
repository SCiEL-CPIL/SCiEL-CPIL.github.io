"""Rewritten program: optimize Hermite coefficients for uncertainty inequality bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, dual_annealing
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 190.0  # 20s safety margin before 210s kill


def _hermite_even(n, x):
    """Compute probabilist's Hermite polynomial He_n(x) for even n."""
    # Probabilist's Hermite polynomials: He_0=1, He_2=x^2-1, He_4=x^4-6x^2+3, He_8=x^8-28x^6+210x^4-420x^2+105
    # He_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    if n == 0:
        return np.ones_like(x)
    elif n == 2:
        return x**2 - 1
    elif n == 4:
        return x**4 - 6*x**2 + 3
    elif n == 6:
        return x**6 - 15*x**4 + 45*x**2 - 15
    elif n == 8:
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    elif n == 10:
        return x**10 - 45*x**8 + 630*x**6 - 3150*x**4 + 3150*x**2 - 675
    elif n == 12:
        return x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
    else:
        # Recursive: He_{n} = x*He_{n-1} - (n-1)*He_{n-2}
        if n < 2:
            return np.ones_like(x)
        he_prev2 = np.ones_like(x)
        he_prev1 = x.copy()
        for k in range(2, n+1):
            he_curr = x * he_prev1 - (k-1) * he_prev2
            he_prev2 = he_prev1
            he_prev1 = he_curr
        return he_prev1


def _g(x, c0, c4, c8):
    """
    g(x) = (c0*He0 + c4*He4 + c8*He8 + c12*He12) * exp(-x^2/2)
    where c12 is determined by g(0) = 0.
    """
    he0 = np.ones_like(x)
    he4 = _hermite_even(4, x)
    he8 = _hermite_even(8, x)
    he12 = _hermite_even(12, x)
    
    # g(0) = (c0 + c4*3 + c8*105 + c12*945) * 1 = 0
    # => c12 = -(c0 + 3*c4 + 105*c8) / 945
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    
    poly = c0*he0 + c4*he4 + c8*he8 + c12*he12
    return poly * np.exp(-x**2 / 2.0)


def _fourier_g(x, c0, c4, c8):
    """
    Fourier transform of g. For probabilist's Hermite polynomials with exp(-x^2/2),
    the Fourier transform (unitary convention) maps He_n(x) -> (-i)^n He_n(x).
    So F[g](x) = (c0*He0 + c4*(-i)^4 He4 + c8*(-i)^8 He8 + c12*(-i)^12 He12) * exp(-x^2/2)
    Since 4, 8, 12 are multiples of 4, (-i)^n = 1 for all.
    So F[g] = g! The Fourier transform is the same function.
    """
    # Actually let me reconsider. The unitary Fourier transform is:
    # F[f](xi) = (1/sqrt(2*pi)) * integral f(x) exp(-i*x*xi) dx
    # For f(x) = He_n(x) * exp(-x^2/2), F[f](xi) = (-i)^n He_n(xi) * exp(-xi^2/2)
    # Since n=0,4,8,12 are all divisible by 4, (-i)^n = 1.
    # So F[g] = g.
    return _g(x, c0, c4, c8)


def _compute_moments(c0, c4, c8):
    """
    Compute A(f) and A(F) where f(x) = g(x)^2 / ||g||_2^2 (normalized L2).
    A(f) = max_x |f(x)| (L-infinity norm of normalized f)
    A(F) = max_x |F(f)(x)| (L-infinity norm of Fourier transform of normalized f)
    
    Actually, let me reconsider the problem. The uncertainty principle bound is:
    A(f) * A(F) >= C4 where A(f) = sup |f| and A(F) = sup |F(f)|.
    The test function approach: we choose g, set f = g/||g||_2 (or similar),
    and compute the actual A(f)*A(F) which gives an upper bound on C4... 
    
    Wait, re-reading: "reduce the upper bound on A(f)*A(Fourier(f))"
    So we want to MINIMIZE the product A(f)*A(F) over our family of test functions.
    The minimum over our family gives the best upper bound we can certify.
    
    Let me think about this more carefully. The standard approach:
    - We have a family of functions parameterized by coefficients
    - For each function f (normalized appropriately), compute A(f)*A(F)
    - We want to minimize this product
    - The minimum value is our "score" (lower is better)
    
    The normalization: typically f is normalized so that ||f||_1 = 1 or ||f||_2 = 1.
    Given the context of uncertainty inequalities, likely ||f||_1 = 1 (probability density)
    or the L2 normalization. Let me assume L2 normalization: f = g / ||g||_2.
    
    Actually, for the uncertainty principle A(f)*A(F) >= C, the standard setup uses
    f in L1 with ||f||_1 = 1, and A(f) = ||f||_inf.
    
    Let me use: f(x) = g(x) / ||g||_1 (if g is non-negative) or more generally
    we need to be careful. Let me just compute both L1 and L2 normalizations and see.
    
    For now, let me use L2 normalization as it's more standard in Fourier analysis.
    """
    # Use a grid for numerical computation
    x = np.linspace(-8, 8, 4096)
    dx = x[1] - x[0]
    
    g_vals = _g(x, c0, c4, c8)
    
    # L2 norm
    norm2 = np.sqrt(np.trapz(g_vals**2, x))
    if norm2 < 1e-15:
        return 1e10  # Invalid
    
    f = g_vals / norm2
    A_f = np.max(np.abs(f))
    
    # Fourier transform of f
    # Use FFT for efficiency
    # f(x) on uniform grid, use numpy FFT
    # The grid is from -8 to 8 with 4096 points
    # FFT gives us F(f)(xi) at discrete frequencies
    
    # For proper FFT: f_k = sum f(x_j) exp(-2*pi*i*k*j/N)
    # We need to shift so that x=0 is at the center
    n = len(x)
    # x goes from -8 to 8, so x[0] = -8
    # FFT assumes input starts at index 0 corresponding to x=0
    # We need to fftshift the input
    f_shifted = np.fft.fftshift(f)
    F_f = np.fft.fft(f_shifted) * dx
    # The frequency axis
    xi = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    
    A_F = np.max(np.abs(F_f))
    
    return A_f * A_F


def _compute_moments_fast(c0, c4, c8):
    """Faster version using fewer grid points."""
    x = np.linspace(-6, 6, 2048)
    dx = x[1] - x[0]
    
    g_vals = _g(x, c0, c4, c8)
    
    norm2 = np.sqrt(np.trapz(g_vals**2, x))
    if norm2 < 1e-15:
        return 1e10
    
    f = g_vals / norm2
    A_f = np.max(np.abs(f))
    
    n = len(x)
    f_shifted = np.fft.fftshift(f)
    F_f = np.fft.fft(f_shifted) * dx
    A_F = np.max(np.abs(F_f))
    
    return A_f * A_F


def _objective(params):
    """Objective function: product A(f)*A(F)."""
    c0, c4, c8 = params
    try:
        return _compute_moments_fast(c0, c4, c8)
    except Exception:
        return 1e10


def _objective_high_res(params):
    """High-resolution objective for final evaluation."""
    c0, c4, c8 = params
    try:
        return _compute_moments(c0, c4, c8)
    except Exception:
        return 1e10


def construct():
    """
    Find optimal Hermite coefficients minimizing A(f)*A(F).
    Returns array of [c0, c4, c8].
    """
    # Starting point: the seed program values
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Evaluate starting point
    best_val = _compute_moments_fast(x0[0], x0[1], x0[2])
    best_params = x0.copy()
    
    # Strategy: Use log-magnitude and sign parameterization for global search
    # a_i = sign_i * exp(m_i)
    # This allows exploration of the full non-linear parameter space
    
    def _log_sign_objective(ls_params):
        """Convert log-sign params to coefficients and evaluate."""
        s0, m0, s4, m4, s8, m8 = ls_params
        c0 = np.sign(s0) * np.exp(m0) if abs(s0) > 1e-10 else 0.0
        c4 = np.sign(s4) * np.exp(m4) if abs(s4) > 1e-10 else 0.0
        c8 = np.sign(s8) * np.exp(m8) if abs(s8) > 1e-10 else 0.0
        return _compute_moments_fast(c0, c4, c8)
    
    # Multi-start optimization with different initial points
    # Start with the known good point and several random perturbations
    
    # Convert initial to log-sign space
    def _to_log_sign(c0, c4, c8):
        s0 = 1.0 if c0 >= 0 else -1.0
        m0 = np.log(abs(c0)) if abs(c0) > 1e-15 else -5.0
        s4 = 1.0 if c4 >= 0 else -1.0
        m4 = np.log(abs(c4)) if abs(c4) > 1e-15 else -5.0
        s8 = 1.0 if c8 >= 0 else -1.0
        m8 = np.log(abs(c8)) if abs(c8) > 1e-15 else -5.0
        return np.array([s0, m0, s4, m4, s8, m8])
    
    def _from_log_sign(ls):
        c0 = np.sign(ls[0]) * np.exp(ls[1]) if abs(ls[0]) > 1e-10 else 0.0
        c4 = np.sign(ls[2]) * np.exp(ls[3]) if abs(ls[2]) > 1e-10 else 0.0
        c8 = np.sign(ls[4]) * np.exp(ls[5]) if abs(ls[4]) > 1e-10 else 0.0
        return np.array([c0, c4, c8])
    
    # Generate multiple starting points
    np.random.seed(42)
    starts = [x0.copy()]
    
    # Add random starts in coefficient space
    for _ in range(15):
        scale = np.array([1.0, 0.1, 0.01])
        random_c = x0 + np.random.randn(3) * scale
        starts.append(random_c)
    
    # Also add some structured starts
    for c0_val in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
        for c4_val in [-0.1, -0.05, 0.05, 0.1]:
            starts.append(np.array([c0_val, c4_val, 0.001]))
    
    # Optimize from each start
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
        
        # Try Nelder-Mead first (robust for small problems)
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Then try L-BFGS-B from the Nelder-Mead result
        if time.time() > _DEADLINE - 10:
            break
        try:
            res2 = minimize(_objective, best_params if res.fun < best_val else start, 
                          method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Additional refinement with log-sign parameterization
    if time.time() < _DEADLINE - 15:
        ls_start = _to_log_sign(best_params[0], best_params[1], best_params[2])
        try:
            res_ls = minimize(_log_sign_objective, ls_start, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-10})
            if res_ls.fun < best_val:
                new_params = _from_log_sign(res_ls.x)
                best_val = res_ls.fun
                best_params = new_params
        except Exception:
            pass
    
    # Final high-resolution evaluation
    if time.time() < _DEADLINE - 5:
        final_val = _compute_moments_high_res(best_params[0], best_params[1], best_params[2])
        if final_val < best_val:
            best_val = final_val
    
    # One more local refinement at high resolution
    if time.time() < _DEADLINE - 5:
        try:
            res_final = minimize(_objective_high_res, best_params, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
