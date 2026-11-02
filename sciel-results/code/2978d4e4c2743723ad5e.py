"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import factorial

# Deadline guard: 210s limit, use 195s for safety
_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

def _time_left():
    return _DEADLINE - time.time()

def _hermite_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n(x) as a list [c_0, c_1, ..., c_n] where H_n(x) = sum c_k x^k."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    # Use recurrence: H_n(x) = 2x*H_{n-1}(x) - 2(n-1)*H_{n-2}(x)
    h_nm2 = [1.0]  # H_0
    h_nm1 = [0.0, 2.0]  # H_1
    for k in range(2, n + 1):
        # H_k = 2x * H_{k-1} - 2(k-1) * H_{k-2}
        # 2x * H_{k-1}: shift coefficients up by 1 and multiply by 2
        term1 = [0.0] + [2.0 * c for c in h_nm1]
        # -2(k-1) * H_{k-2}
        term2 = [-2.0 * (k - 1) * c for c in h_nm2]
        # Pad to same length
        max_len = max(len(term1), len(term2))
        term1 = term1 + [0.0] * (max_len - len(term1))
        term2 = term2 + [0.0] * (max_len - len(term2))
        h_n = [term1[i] + term2[i] for i in range(max_len)]
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1

def _hermite_at_zero(n):
    """Return H_n(0)."""
    coeffs = _hermite_coeffs(n)
    return coeffs[0] if len(coeffs) > 0 else 0.0

# Precompute H_n(0) values
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 3024.0
_H12_0 = 10321920.0

def _c4_bound_from_coeffs(a0, a4, a8):
    """
    Compute the C4 upper bound.
    
    The task mentions "upper bound on A(f)*A(Fourier(f))" and "score is the upper bound on C4".
    In the context of uncertainty inequalities and Hermite-Gaussian test functions,
    C4 is likely related to the product of norms or a specific constant in the inequality.
    
    Based on the seed value and typical structures, let's assume the objective is to minimize
    a quantity derived from the L2 norms or L1 norms.
    
    Let's try the L2 norm squared product first, as it's analytically tractable.
    g(x) = P(x) exp(-x^2/2)
    ||g||_2^2 = integral P(x)^2 exp(-x^2) dx
    Since P(x) = sum a_n H_n(x), and H_n are orthogonal w.r.t exp(-x^2),
    ||g||_2^2 = sqrt(pi) * sum a_n^2 2^n n!
    
    By Plancherel, ||F[g]||_2 = ||g||_2.
    So if A(f) = ||f||_2, then A(f)A(F(f)) = ||g||_2^2.
    Minimizing this is trivial (set all to 0), so that's not it.
    
    Maybe A(f) is the L1 norm? Or maybe it's the uncertainty principle bound:
    sigma_x sigma_p >= 1/2.
    For a normalized wavefunction, sigma_x^2 = <x^2> - <x>^2.
    If g is even (which it is, since H_n for even n are even, and exp is even), <x> = 0.
    So sigma_x^2 = <x^2>.
    Similarly for momentum.
    
    Let's compute sigma_x^2 and sigma_p^2 for g(x) = P(x) exp(-x^2/2) / ||g||_2.
    sigma_x^2 = (1/||g||_2^2) integral x^2 P(x)^2 exp(-x^2) dx
    sigma_p^2 = (1/||g||_2^2) integral P'(x)^2 exp(-x^2) dx  (up to constants)
    
    Actually, let's look at the seed: [-1.13, 0.04, 0.000308642].
    Let's try to compute the "C4" bound numerically using L1 norms, as that's a common
    uncertainty inequality variant (Beckner inequality involves Lp norms).
    
    Beckner inequality: ||F[f]||_q >= C_{p,q} ||f||_p
    For p=q=2, it's Plancherel.
    
    Let's assume the score is simply the value of some function of the coefficients.
    Since I don't have the exact definition, I will use a robust numerical approach
    that computes the L1 norm of g and its Fourier transform, and returns the product.
    This is a standard uncertainty measure.
    """
    # Compute a12 from P(0)=0
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    # Numerical integration for L1 norm
    x = np.linspace(-10, 10, 5000)
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 4608*x**6 + 23040*x**4 - 34560*x**2 + 10080
    H12 = (4096*x**12 - 147456*x**10 + 2090112*x**8 - 
           12902400*x**6 + 34406400*x**4 - 34406400*x**2 + 10321920)
    
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = poly * np.exp(-x**2 / 2.0)
    
    if hasattr(np, 'trapezoid'):
        A_g = np.trapezoid(np.abs(g), x)
    else:
        A_g = np.trapz(np.abs(g), x)
    
    if A_g < 1e-10 or not np.isfinite(A_g):
        return 1e10
        
    # Numerical Fourier Transform
    # Use FFT for speed
    N = 16384
    x_ft = np.linspace(-10, 10, N)
    dx = x_ft[1] - x_ft[0]
    
    H0f = np.ones_like(x_ft)
    H4f = 16*x_ft**4 - 48*x_ft**2 + 12
    H8f = 256*x_ft**8 - 4608*x_ft**6 + 23040*x_ft**4 - 34560*x_ft**2 + 10080
    H12f = (4096*x_ft**12 - 147456*x_ft**10 + 2090112*x_ft**8 - 
            12902400*x_ft**6 + 34406400*x_ft**4 - 34406400*x_ft**2 + 10321920)
    
    poly_f = a0 * H0f + a4 * H4f + a8 * H8f + a12 * H12f
    g_ft = poly_f * np.exp(-x_ft**2 / 2.0)
    
    G = np.fft.fftshift(np.fft.fft(g_ft)) * dx
    # Note: FFT assumes frequency domain, but for L1 norm we just need |G|
    # The scaling might be off, but we are comparing relative values.
    # To be consistent, let's use the same grid for frequency.
    # Actually, for the product A(f)A(F(f)), the scaling of FT matters.
    # Standard FT: F(x) = int f(t) e^{-ixt} dt.
    # FFT approximates this with scaling dx.
    
    if hasattr(np, 'trapezoid'):
        A_G = np.trapezoid(np.abs(G), x_ft) # Using x_ft as proxy for freq grid spacing
    else:
        A_G = np.trapz(np.abs(G), x_ft)
        
    # The frequency grid is actually 2*pi*k/(N*dx).
    # Let's do it properly.
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    A_G = np.trapezoid(np.abs(G), freqs)
    
    bound = A_g * A_G
    if not np.isfinite(bound) or bound < 0:
        return 1e10
    return bound

def construct():
    # Seed result
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_result = seed_result.copy()
    best_bound = _c4_bound_from_coeffs(*seed_result)
    
    # Define bounds for optimization
    # Based on seed: a0 ~ -1.13, a4 ~ 0.04, a8 ~ 0.0003
    bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
    
    # Strategy: Multi-start Differential Evolution + Local Polish
    # 1. Global search with DE
    if _time_left() > 30:
        try:
            de_result = differential_evolution(
                _c4_bound_from_coeffs,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                polish=False,
                init='sobol',
                popsize=15
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_result = de_result.x
        except Exception:
            pass
    
    # 2. Local refinement with Nelder-Mead from multiple starts
    starts = [
        best_result,
        seed_result,
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-1.1, 0.04, 0.0003]),
        np.array([-1.15, 0.035, 0.00031]),
    ]
    
    for start in starts:
        if _time_left() < 10:
            break
        try:
            res = minimize(
                _c4_bound_from_coeffs,
                start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # 3. Final high-precision polish if time permits
    if _time_left() > 5:
        try:
            res = minimize(
                _c4_bound_from_coeffs,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
