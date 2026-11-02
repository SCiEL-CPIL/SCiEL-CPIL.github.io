"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize
import time

def _hermite_coeffs():
    """Precompute Hermite polynomials H0, H4, H8, H12."""
    H0 = np.array([1.0])
    H4 = np.array([12, 0, -12, 0, 1])  # H4(x) = x^4 - 6x^2 + 3 -> coefficients [3, 0, -6, 0, 1]
    # Actually, physicists' Hermite: H4(x) = x^4 - 6x^2 + 3
    # Let's use eval_hermite for correctness
    H0 = np.array([1.0])
    H4 = np.array([3.0, 0.0, -6.0, 0.0, 1.0])
    H8 = np.array([105.0, 0.0, -420.0, 0.0, 420.0, 0.0, -168.0, 0.0, 28.0, 0.0, 1.0])
    H12 = np.array([10395.0, 0.0, -69300.0, 0.0, 138600.0, 0.0, -138600.0, 0.0, 79800.0, 0.0, -27720.0, 0.0, 5544.0, 0.0, -792.0, 0.0, 55.0, 0.0, 1.0])
    return H0, H4, H8, H12

def _poly_eval(coeffs, x):
    """Evaluate polynomial with given coefficients at x."""
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        if c != 0:
            result += c * x**i
    return result

def _g_and_constraint(a0, a4, a8):
    """
    Given a0, a4, a8, compute a12 such that g(0) = 0.
    g(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    So a12 = -(a0 + 3*a4 + 105*a8) / 10395
    """
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 10395.0
    return a12

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty bound C4 for the given coefficients.
    
    The bound involves the ratio ||g||_1 / min|ĝ| or similar structure.
    For the Fourier uncertainty inequality with Hermite-Gaussian test functions:
    
    A(f) * A(F(f)) >= C4 * ||f||_1^2 / ||f||_2^2 or similar
    
    Let me think about this more carefully. The task says "upper bound on A(f)*A(Fourier(f))"
    and "score is the upper bound on C4".
    
    For a test function g, the uncertainty constant can be bounded by:
    C4 <= (||g||_1 / ||g||_2) * (||ĝ||_1 / ||ĝ||_2) / something
    
    Actually, for the Beurling-type uncertainty:
    A(f) * A(ĝ) >= ||f||_1 * ||ĝ||_1 / (||f||_2 * ||ĝ||_2) * C
    
    Let me use a direct numerical approach: compute the bound as a function of the coefficients.
    
    The key quantity is likely:
    bound = ||g||_1 * ||ĝ||_1 / (||g||_2^2 * ||ĝ||_2^2) or similar
    
    For Gaussian-weighted Hermite functions, let's compute numerically.
    """
    # Use a grid to compute norms
    x = np.linspace(-8, 8, 4000)
    dx = x[1] - x[0]
    
    # Compute g(x)
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12 = x**12 - 42*x**10 + 840*x**8 - 7560*x**6 + 30240*x**4 - 45360*x**2 + 10395
    
    a12 = _g_and_constraint(a0, a4, a8)
    
    g = a0*H0 + a4*H4 + a8*H8 + a12*H12
    
    # Gaussian weight (for the test function f = g * exp(-x^2/4) or similar)
    # Standard Hermite-Gaussian: phi_n(x) = H_n(x) * exp(-x^2/4)
    # Let's use the standard convention where the Fourier transform of exp(-x^2/4)H_n(x) 
    # is related to exp(-x^2/4)H_n(x) up to constants
    
    # Actually, let's use f(x) = g(x) * exp(-x^2/4)
    gauss = np.exp(-x**2 / 4.0)
    f = g * gauss
    
    # Compute ||f||_1, ||f||_2
    f1 = np.abs(f)
    f2 = f**2
    
    norm1 = np.sum(f1) * dx
    norm2_sq = np.sum(f2) * dx
    norm2 = np.sqrt(norm2_sq) if norm2_sq > 0 else 1e-30
    
    # Compute Fourier transform numerically
    # Use FFT with appropriate scaling
    n = len(x)
    dx_val = x[1] - x[0]
    
    # Pad for better FFT
    nfft = 1 << (n + 1).bit_length()
    f_padded = np.zeros(nfft, dtype=complex)
    f_padded[:n] = f
    f_padded[n:] = f[:nfft-n]  # wrap-around
    
    F = np.fft.fftshift(np.fft.fft(f_padded)) * dx_val
    
    # Frequency grid
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=dx_val))
    
    # Compute ||F||_1, ||F||_2
    F1 = np.abs(F)
    F2 = F.real**2 + F.imag**2
    
    F_norm1 = np.sum(F1) * (2*np.pi / nfft) * nfft / (2*np.pi)  # need proper scaling
    # Actually, for FFT: F[k] = sum f[j] * exp(-2πi jk/n) * dx
    # So ||F||_1 ≈ sum |F[k]| * df where df = 1/(n*dx)
    df = 1.0 / (nfft * dx_val)
    F_norm1 = np.sum(F1) * df
    F_norm2_sq = np.sum(F2) * df
    F_norm2 = np.sqrt(F_norm2_sq) if F_norm2_sq > 0 else 1e-30
    
    # The uncertainty bound C4 is related to:
    # C4 = ||f||_1 * ||F||_1 / (||f||_2 * ||F||_2)
    # This is the ratio that appears in Beurling's uncertainty principle
    
    if norm2 > 1e-15 and F_norm2 > 1e-15:
        bound = (norm1 * F_norm1) / (norm2 * F_norm2)
    else:
        bound = 1e10
    
    return bound

def _compute_bound_fast(a0, a4, a8):
    """Faster computation using fewer grid points for optimization."""
    x = np.linspace(-6, 6, 2000)
    dx = x[1] - x[0]
    
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12 = x**12 - 42*x**10 + 840*x**8 - 7560*x**6 + 30240*x**4 - 45360*x**2 + 10395
    
    a12 = _g_and_constraint(a0, a4, a8)
    
    g = a0*H0 + a4*H4 + a8*H8 + a12*H12
    gauss = np.exp(-x**2 / 4.0)
    f = g * gauss
    
    f1 = np.abs(f)
    f2 = f**2
    
    norm1 = np.sum(f1) * dx
    norm2_sq = np.sum(f2) * dx
    norm2 = np.sqrt(norm2_sq) if norm2_sq > 0 else 1e-30
    
    n = len(x)
    dx_val = x[1] - x[0]
    nfft = 1 << (n + 1).bit_length()
    
    f_padded = np.zeros(nfft, dtype=complex)
    f_padded[:n] = f
    f_padded[n:] = f[:nfft-n]
    
    F = np.fft.fftshift(np.fft.fft(f_padded)) * dx_val
    F1 = np.abs(F)
    F2 = F.real**2 + F.imag**2
    
    df = 1.0 / (nfft * dx_val)
    F_norm1 = np.sum(F1) * df
    F_norm2_sq = np.sum(F2) * df
    F_norm2 = np.sqrt(F_norm2_sq) if F_norm2_sq > 0 else 1e-30
    
    if norm2 > 1e-15 and F_norm2 > 1e-15:
        bound = (norm1 * F_norm1) / (norm2 * F_norm2)
    else:
        bound = 1e10
    
    return bound

def construct():
    """
    Find optimal coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses angular parameterization + grid search + local refinement.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit, keep safety margin
    
    # The bound is scale-invariant (ratio of norms), so we can fix the scale
    # and search over directions in the 3D coefficient space.
    # Parameterize: (a0, a4, a8) = r * (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    # Since the bound is scale-invariant, r doesn't matter.
    
    best_bound = 1e10
    best_coeffs = np.array([-113/100, 1/25, 1/3240])  # initial best
    
    # Phase 1: Coarse grid search over angles
    n_theta = 40
    n_phi = 40
    
    theta_vals = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_vals = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    for i, theta in enumerate(theta_vals):
        if time.time() > deadline - 30:
            break
        st = np.sin(theta)
        ct = np.cos(theta)
        for phi in phi_vals:
            if time.time() > deadline - 30:
                break
            a0 = st * np.cos(phi)
            a4 = st * np.sin(phi)
            a8 = ct
            try:
                b = _compute_bound_fast(a0, a4, a8)
                if b < best_bound:
                    best_bound = b
                    best_coeffs = np.array([a0, a4, a8])
            except:
                pass
    
    # Phase 2: Local refinement using scipy.optimize
    if time.time() < deadline - 20:
        def objective(params):
            a0, a4, a8 = params
            return _compute_bound_fast(a0, a4, a8)
        
        # Try refinement from the best grid point
        x0 = best_coeffs.copy()
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except:
            pass
        
        # Try a few additional starting points
        if time.time() < deadline - 10:
            random_starts = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([1.0, 1.0, 1.0]),
                np.array([-1.0, 1.0, -1.0]),
            ]
            for rs in random_starts:
                if time.time() > deadline - 10:
                    break
                try:
                    result = minimize(objective, rs, method='Nelder-Mead',
                                   options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                    if result.fun < best_bound:
                        best_bound = result.fun
                        best_coeffs = result.x
                except:
                    pass
    
    # Phase 3: High-precision evaluation of the best coefficients
    if time.time() < deadline - 5:
        try:
            final_bound = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
        except:
            final_bound = best_bound
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
