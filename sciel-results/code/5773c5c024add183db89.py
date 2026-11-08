"""Rewritten program: global optimization of Hermite coefficients for uncertainty inequality bound."""
import time
import numpy as np
from scipy.special import factorial
from scipy.optimize import differential_evolution

# Deadline guard: 210s wall-clock limit, use 195s as safety margin
_DEADLINE = 195.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _hermite_polynomial(n, x):
    """Compute probabilist's Hermite polynomial He_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h_nm2 = np.ones_like(x)
    h_nm1 = x.copy()
    for k in range(2, n + 1):
        h_n = x * h_nm1 - (k - 1) * h_nm2
        h_nm2 = h_nm1
        h_nm1 = h_n
    return h_nm1


def _compute_c4(c0, c4, c8):
    """
    Compute the upper bound C4 for the uncertainty inequality.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is fixed so that g(0) = 0.
    
    For probabilist's Hermite polynomials:
    H0(x) = 1
    H4(x) = x^4 - 6x^2 + 3
    H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 30240x^2 + 10395
    
    g(0) = (c0*1 + c4*3 + c8*105 + c12*10395) * 1 = 0
    => c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    The Fourier transform of H_n(x)*exp(-x^2/2) is (-i)^n * H_n(x) * exp(-x^2/2) (up to constants)
    for probabilist's Hermite. Actually, for the Fourier transform convention, we need to be careful.
    
    Let's use the standard result: if g(x) = sum c_n H_n(x) e^{-x^2/2}, then
    the L2 norm squared is sum |c_n|^2 * n! (for probabilist's Hermite with this normalization).
    
    Actually, let me think more carefully. The probabilist's Hermite polynomials satisfy:
    ∫ H_m(x) H_n(x) e^{-x^2/2} dx = √(2π) * n! * δ_{mn}
    
    Wait, let me use the standard orthonormal basis. The functions ψ_n(x) = (1/√(2^n n! √π)) H_n(x) e^{-x^2/4}
    are orthonormal in L2(R) for the physicist's convention. Let me just work with the raw integrals.
    
    For g(x) = P(x) e^{-x^2/2} where P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12:
    
    ||g||_2^2 = ∫ |P(x)|^2 e^{-x^2} dx
    
    Using orthogonality: ∫ H_m(x) H_n(x) e^{-x^2/2} dx = √(2π) n! δ_{mn}
    
    But we have e^{-x^2} not e^{-x^2/2}. Let me reconsider.
    
    Actually, the standard result for the uncertainty principle with Gaussian-weighted Hermite:
    If g(x) = Σ c_n H_n(x) e^{-x^2/2}, then:
    ||g||_2^2 = Σ |c_n|^2 √(2π) n!
    
    And the Fourier transform (with convention F[f](ξ) = ∫ f(x) e^{-2πixξ} dx) of H_n(x)e^{-x^2/2} is...
    
    Let me just numerically compute the bound. The upper bound C4 relates to:
    C4 ≥ ||g||_2^2 * ||g'||_2^2 / ||g||_2^4 ... no.
    
    Actually, for the uncertainty inequality, the bound is:
    A(f) * A(ĝ) ≥ C
    
    where A(f) = (∫ x^2 |f|^2 dx)^{1/2} (spread in x) and A(ĝ) = (∫ ξ^2 |ĝ|^2 dξ)^{1/2} (spread in freq).
    
    The "upper bound on C4" likely refers to minimizing the product of spreads for our test function, which gives a lower bound on the constant. But the task says "reduce the upper bound on A(f)*A(Fourier(f))" - so we want to MINIMIZE the product.
    
    Let me just compute numerically:
    - g(x) = P(x) * exp(-x^2/2) on a grid
    - Compute ||g||_2^2, ∫ x^2|g|^2 dx, and the Fourier transform numerically
    - Compute ∫ ξ^2|ĝ|^2 dξ
    - The product is sqrt(∫x^2|g|^2 dx / ∫|g|^2 dx) * sqrt(∫ξ^2|ĝ|^2 dξ / ∫|ĝ|^2 dξ)
    
    Actually, re-reading: "score is the upper bound on C4" and "raw value (lower is better)". So we minimize the product of the spreads.
    """
    # Fix c12 so g(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0
    
    # Create grid
    x = np.linspace(-15, 15, 4096)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    H0 = _hermite_polynomial(0, x)
    H4 = _hermite_polynomial(4, x)
    H8 = _hermite_polynomial(8, x)
    H12 = _hermite_polynomial(12, x)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = P * np.exp(-0.5 * x**2)
    
    # Check if g is trivially zero
    norm2 = np.sum(g**2) * dx
    if norm2 < 1e-30:
        return 1e10
    
    # Compute x-spread: sqrt(∫ x^2 |g|^2 dx / ∫ |g|^2 dx)
    x2_weighted = np.sum(x**2 * g**2) * dx
    x_spread = np.sqrt(x2_weighted / norm2)
    
    # Compute Fourier transform numerically
    # Use FFT with appropriate scaling
    # g(x) on grid x, FFT gives sum g(x_k) exp(-2πi k x_k / L)
    # We need the continuous FT: ĝ(ξ) = ∫ g(x) exp(-2πixξ) dx
    # Using trapezoidal rule approximation via FFT
    
    N = len(x)
    x0 = x[0]
    L = N * dx
    
    # Shift to make grid start at 0 for FFT
    # g(x) where x goes from -15 to 15
    # FFT of g sampled at x_k = x0 + k*dx
    # The DFT is: G[k] = sum_j g[j] exp(-2πi jk/N)
    # The continuous FT at frequencies ξ_k = k/(N*dx) is approx: dx * G[k]
    
    G = np.fft.fft(g)
    # Frequencies: ξ_k = k/(N*dx) for k=0..N/2, and negative for k>N/2
    freqs = np.fft.fftfreq(N, d=dx)
    g_hat = dx * G
    
    # Compute freq-spread: sqrt(∫ ξ^2 |ĝ|^2 dξ / ∫ |ĝ|^2 dξ)
    # ∫ |ĝ|^2 dξ ≈ sum |g_hat|^2 * dξ where dξ = 1/(N*dx)
    dxi = 1.0 / (N * dx)
    freq_norm2 = np.sum(g_hat**2) * dxi
    freq2_weighted = np.sum(freqs**2 * g_hat**2) * dxi
    
    if freq_norm2 < 1e-30:
        return 1e10
    
    freq_spread = np.sqrt(freq2_weighted / freq_norm2)
    
    # The product of spreads
    product = x_spread * freq_spread
    
    return product


def construct():
    """
    Find coefficients (c0, c4, c8) that minimize the uncertainty product.
    Due to scale invariance, we can fix c0 = 1 and optimize c4, c8.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Due to scale invariance of the product of spreads, fix c0 = 1
    # Optimize over c4 and c8
    
    def objective(params):
        c4, c8 = params
        try:
            return _compute_c4(1.0, c4, c8)
        except Exception:
            return 1e10
    
    # Initial best from seed
    c4_seed = 1.0 / 25.0
    c8_seed = 1.0 / 3240.0
    best_val = objective([c4_seed, c8_seed])
    best_params = [c4_seed, c8_seed]
    
    # Use differential evolution for global optimization
    # Bounds: based on the nature of the problem, coefficients shouldn't be too large
    # If coefficients are too large, the polynomial dominates and the Gaussian doesn't suppress
    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    
    # Check if we have enough time for DE
    elapsed = time.time() - start_time
    if elapsed < 150.0:  # Leave 60s margin
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol',
                updating='deferred',
                mutation=(0.3, 1.0),
                recombination=0.9,
                workers=-1  # Use all cores if possible
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = list(result.x)
        except Exception:
            pass
    
    # If we still have time, try local refinement
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 30:
        from scipy.optimize import minimize
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = list(res.x)
        except Exception:
            pass
    
    # Return the three coefficients (c0, c4, c8)
    # c0 is fixed to 1 by our normalization
    return np.array([1.0, best_params[0], best_params[1]])


# For safety, also handle the case where construct is called and we need to return
# the coefficients in the original format. The seed returned [-113/100, 1/25, 1/3240]
# which has c0 = -113/100. But since the product of spreads is scale-invariant,
# we can normalize c0 to any non-zero value. Let's keep c0 = 1.

# Actually, wait - let me reconsider. The seed has c0 = -113/100, not 1.
# The scale invariance means we can multiply all coefficients by a constant and the
# product of spreads doesn't change. So returning [1, c4, c8] should be fine.

# But let me double-check by also trying with the original c0 sign convention.
# Actually, the product of spreads is invariant under g -> a*g for any non-zero a,
# so the sign and magnitude of c0 don't matter for the score. c0=1 is fine.
