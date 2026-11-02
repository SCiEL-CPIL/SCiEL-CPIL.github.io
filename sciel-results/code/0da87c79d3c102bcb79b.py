"""Optimize Fourier-invariant Hermite-Gaussian test function for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import factorial
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 200.0  # safety margin before 210s kill

def _hermite_phys(n, x):
    """Physicist's Hermite polynomials H_n(x) via explicit formula."""
    # H_n(x) = n! * sum_{k=0}^{floor(n/2)} (-1)^k (2x)^(n-2k) / (k! (n-2k)!)
    result = np.zeros_like(x, dtype=float)
    for k in range(n // 2 + 1):
        coeff = factorial(n) * ((-1) ** k) / (factorial(k) * factorial(n - 2 * k))
        result += coeff * (2 * x) ** (n - 2 * k)
    return result

def _hermite_vals(x):
    """Return H0, H4, H8, H12 evaluated at x."""
    H0 = np.ones_like(x, dtype=float)
    H4 = 16 * x**4 - 48 * x**2 + 12
    H8 = 256 * x**8 - 1920 * x**6 + 3360 * x**4 - 1680 * x**2 + 168
    H12 = 4096 * x**12 - 46080 * x**10 + 230400 * x**8 - 403200 * x**6 + 302400 * x**4 - 120960 * x**2 + 10080
    return H0, H4, H8, H12

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound C4 for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12 - a12) * exp(-x^2/2)
    where a12 is chosen so that g(0) = 0, i.e., P(0) = 0.
    
    P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) - a12 = 0
    H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=10080
    So: a0 + 12*a4 + 168*a8 + a12*10080 - a12 = 0
    => a0 + 12*a4 + 168*a8 + a12*10079 = 0
    => a12 = -(a0 + 12*a4 + 168*a8) / 10079
    """
    a12 = -(a0 + 12.0 * a4 + 168.0 * a8) / 10079.0
    
    # Build g(x) on a grid
    x = np.linspace(-15, 15, 20000)
    H0, H4, H8, H12 = _hermite_vals(x)
    
    P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12 - a12
    g = P * np.exp(-x**2 / 2)
    
    # Check g(0) ≈ 0
    # A(f) for Fourier-invariant: we need to understand the exact bound
    # For the uncertainty inequality A(f)*A(F(f)) <= C4
    # Typically A(f) = ||f||_1 or similar, and the bound involves integrals
    #
    # From the context of Goncalves et al., the bound C4 is:
    # C4 = (||g||_1 * ||g'||_1) / (||g||_2^2) or similar ratio
    #
    # Actually, for the Fourier uncertainty inequality in the form:
    # ||f||_1 * ||f̂||_1 >= ||f||_2^2
    # The constant C4 is the best such that ||f||_1 * ||f̂||_1 >= C4 * ||f||_2^2
    #
    # For a test function g, we compute the ratio:
    # C4_candidate = ||g||_1 * ||ĝ||_1 / ||g||_2^2
    # and we want to MINIMIZE this to get a better (lower) upper bound
    #
    # Wait - re-reading: "reduce the upper bound on A(f)*A(Fourier(f))"
    # So we want to find g that gives a small value of the bound.
    # The bound for a specific g is: A(g)*A(F(g)) where A is some norm.
    #
    # Let me reconsider. In the uncertainty principle context:
    # A(f) might be ||f||_1, and the inequality is ||f||_1 * ||f̂||_1 >= C * ||f||_2^2
    # The best C is the infimum over all f of ||f||_1 * ||f̂||_1 / ||f||_2^2
    #
    # For our test function g, we compute R(g) = ||g||_1 * ||ĝ||_1 / ||g||_2^2
    # and the "upper bound" we're trying to reduce is this ratio.
    # Lower ratio = better bound = smaller C4.
    
    # Compute ||g||_1
    dx = x[1] - x[0]
    norm1 = np.sum(np.abs(g)) * dx
    
    # Compute ||g||_2
    norm2_sq = np.sum(g**2) * dx
    
    # Compute Fourier transform of g numerically
    # g(x) = P(x) * exp(-x^2/2)
    # The Fourier transform of exp(-x^2/2) is exp(-xi^2/2) (with appropriate convention)
    # Using convention: f̂(ξ) = ∫ f(x) exp(-2πixξ) dx
    # For f(x) = exp(-x^2/2), f̂(ξ) = sqrt(2π) * exp(-2π^2 ξ^2)
    #
    # Actually, let's use the convention where F(f)(ξ) = ∫ f(x) e^{-ixξ} dx
    # Then for f(x) = exp(-x^2/2), F(f)(ξ) = sqrt(2π) * exp(-ξ^2/2)
    #
    # For polynomial times Gaussian, the Fourier transform is also polynomial times Gaussian.
    # Let me compute it numerically to be safe.
    
    # Use FFT-based approach or direct integration
    # Direct integration on a fine grid for the Fourier transform
    xi = np.linspace(-15, 15, 20000)
    
    # F(g)(ξ) = ∫ g(x) e^{-ixξ} dx
    # = ∫ P(x) e^{-x^2/2} e^{-ixξ} dx
    # = ∫ P(x) e^{-(x^2/2 + ixξ)} dx
    # = ∫ P(x) e^{-(x+ξ)^2/2 + ξ^2/2} dx
    # = e^{ξ^2/2} ∫ P(x) e^{-(x+ξ)^2/2} dx
    # Let u = x + ξ: = e^{ξ^2/2} ∫ P(u-ξ) e^{-u^2/2} du
    #
    # This is getting complex. Let me just do numerical integration.
    
    # For efficiency, compute Fourier transform via the fact that
    # F(P(x)e^{-x^2/2})(ξ) can be computed analytically for polynomials
    #
    # F(x^n e^{-x^2/2})(ξ) = i^n d^n/dξ^n [sqrt(2π) e^{-ξ^2/2}]
    #
    # H_n(x) e^{-x^2/2} has a simple Fourier transform:
    # F(H_n(x) e^{-x^2/2})(ξ) = i^n sqrt(2π) H_n(ξ) e^{-ξ^2/2}
    # (with the convention F(f)(ξ) = ∫ f(x) e^{-ixξ} dx)
    
    # So F(g)(ξ) = [a0*H0 + a4*H4 + a8*H8 + a12*H12 - a12] * sqrt(2π) * e^{-ξ^2/2}
    # Wait, that's not quite right because of the i^n factors.
    #
    # F(H_n(x) e^{-x^2/2})(ξ) = i^n sqrt(2π) H_n(ξ) e^{-ξ^2/2}
    #
    # So F(g)(ξ) = sqrt(2π) e^{-ξ^2/2} [a0*H0(ξ) + a4*i^4*H4(ξ) + a8*i^8*H8(ξ) + a12*i^12*H12(ξ) - a12]
    # i^4 = 1, i^8 = 1, i^12 = i^4*i^4*i^4 = 1
    # So F(g)(ξ) = sqrt(2π) e^{-ξ^2/2} [a0*H0(ξ) + a4*H4(ξ) + a8*H8(ξ) + a12*H12(ξ) - a12]
    #
    # Interesting! The Fourier transform has the same form with the same polynomial!
    
    sqrt_2pi = np.sqrt(2 * np.pi)
    g_hat = sqrt_2pi * P * np.exp(-xi**2 / 2)  # Note: P evaluated at xi, but P is same form
    
    # Actually P is a function of its argument, so P(xi) = a0*H0(xi) + a4*H4(xi) + a8*H8(xi) + a12*H12(xi) - a12
    # Since we already computed P on x grid, and xi grid is same as x grid, P values are the same
    # (the grid is symmetric and same points)
    
    norm1_hat = np.sum(np.abs(g_hat)) * dx
    
    if norm2_sq <= 0:
        return 1e10
    
    ratio = (norm1 * norm1_hat) / norm2_sq
    return ratio

def _objective(params):
    a0, a4, a8 = params
    return _compute_bound(a0, a4, a8)

def construct():
    """Return three coefficients (a0, a4, a8) minimizing the C4 bound."""
    # Start from prior art
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_bound(*best_params)
    
    # Try some initial perturbations
    np.random.seed(42)
    candidates = [best_params.copy()]
    
    # Random perturbations around prior art
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for _ in range(5):
            pert = best_params + scale * np.random.randn(3)
            candidates.append(pert)
    
    # Also try some structured variations
    candidates.append(np.array([-1.13, 0.04, 0.0003]))
    candidates.append(np.array([-1.0, 0.05, 0.0005]))
    candidates.append(np.array([-1.2, 0.03, 0.0002]))
    candidates.append(np.array([-1.1, 0.045, 0.00035]))
    
    # Evaluate all candidates
    for c in candidates:
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        val = _compute_bound(*c)
        if val < best_val:
            best_val = val
            best_params = c.copy()
    
    # Local optimization from best point
    if time.time() - _START_TIME < _TIME_LIMIT - 30:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() - _START_TIME < _TIME_LIMIT - 15:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
