"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math


def _make_bound_evaluator(a0, a4, a8):
    """
    Returns a function that, given a (1D) array of x, computes
        A(f)  = (1 / ∫ f^2) ∫ |f|
        A(F)  = (1 / ∫ F^2) ∫ |F|
    where f(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) * exp(-x^2/2)
    with a12 chosen so that f(0) = 0.
    The product A(f)*A(F) is the upper bound on C4.
    """
    # Precompute factorials and powers for Hk(x) = (-1)^k exp(x^2/2) d^k/dx^k exp(-x^2/2)
    # Physicist's Hermite polynomials: H0=1, H4=16x^4-48x^2+12, H8=256x^8-1280x^6+2880x^4-1920x^2+168
    # Actually, let's use the standard physicist's Hermite polynomials.
    # H_n(x) defined by exp(-x^2) = sum_{n=0}^inf H_n(x) t^n / n! is not right.
    # Standard: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
    # H0 = 1
    # H1 = 2x
    # H2 = 4x^2 - 2
    # H3 = 8x^3 - 12x
    # H4 = 16x^4 - 48x^2 + 12
    # H5 = 32x^5 - 160x^3 + 120x
    # H6 = 64x^6 - 480x^4 + 720x^2 - 120
    # H7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
    # H8 = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
    # H9 = 512x^9 - 10240x^7 + 60480x^5 - 120960x^3 + 60480x
    # H10 = 1024x^10 - 28160x^8 + 230400x^6 - 604800x^4 + 403200x^2 - 30240
    # H11 = 2048x^11 - 73728x^9 + 783360x^7 - 3144960x^5 + 4536000x^3 - 1814400x
    # H12 = 4096x^12 - 184320x^10 + 2603520x^8 - 15244800x^6 + 38016000x^4 - 35315200x^2 + 7683840

    def H0(x):
        return np.ones_like(x)

    def H4(x):
        return 16*x**4 - 48*x**2 + 12

    def H8(x):
        return 256*x**8 - 3584*x**6 + 13440*x**4 - 13440*x**2 + 1680

    def H12(x):
        return (4096*x**12 - 184320*x**10 + 2603520*x**8 - 15244800*x**6 + 
                38016000*x**4 - 35315200*x**2 + 7683840)

    def f(x):
        # a12 chosen so that f(0) = 0
        # f(0) = (a0*1 + a4*12 + a8*1680 + a12*7683840) * 1 = 0
        # => a12 = -(a0 + 12*a4 + 1680*a8) / 7683840
        a12 = -(a0 + 12*a4 + 1680*a8) / 7683840.0
        g = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
        return g * np.exp(-x**2 / 2.0)

    def F(x):
        # Fourier transform of f. For f(x) = sum c_k H_k(x) exp(-x^2/2),
        # using the unitary Fourier transform convention: F(ξ) = (1/√(2π)) ∫ f(x) e^{-iξx} dx
        # There's a known relation: if f(x) = H_n(x) exp(-x^2/2), then its FT is (-i)^n H_n(x) exp(-x^2/2) / √(2π) * √(2π) ... 
        # Actually, let me use the convention where FT of exp(-x^2/2) is exp(-ξ^2/2).
        # Under unitary FT: F(ξ) = (1/√(2π)) ∫ f(x) e^{-iξx} dx
        # FT of exp(-x^2/2) is exp(-ξ^2/2)
        # FT of H_n(x) exp(-x^2/2) = (-i)^n H_n(ξ) exp(-ξ^2/2)
        # So F(ξ) = [a0*(-i)^0 H0(ξ) + a4*(-i)^4 H4(ξ) + a8*(-i)^8 H8(ξ) + a12*(-i)^12 H12(ξ)] exp(-ξ^2/2)
        # Since (-i)^4 = (-i)^8 = (-i)^12 = 1 (all even powers of -i are 1)
        # Wait: (-i)^2 = -1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1. Yes.
        # So F(ξ) = [a0 H0(ξ) + a4 H4(ξ) + a8 H8(ξ) + a12 H12(ξ)] exp(-ξ^2/2) = f(ξ)
        # That's interesting - the FT is the same function? No, that's not right.
        # Let me re-check. The FT of x^n exp(-x^2/2) involves derivatives.
        # Actually, the property is: if g(x) = H_n(x) exp(-x^2/2), then F(g)(ξ) = (-i)^n H_n(ξ) exp(-ξ^2/2).
        # For n=0: F(exp(-x^2/2)) = exp(-ξ^2/2). (-i)^0 = 1. OK.
        # For n=1: F(2x exp(-x^2/2)) = (-i) H_1(ξ) exp(-ξ^2/2) = (-i)(2ξ) exp(-ξ^2/2). 
        #   Indeed, FT of x exp(-x^2/2) = i d/dξ exp(-ξ^2/2) / √(2π) * √(2π) ... let me just use numerical FT to be safe.
        
        # To avoid confusion with conventions, let me compute the FT numerically using a large enough domain.
        # But for speed, I'll use the analytical relation. The key insight:
        # Under the unitary FT, the Hermite functions ψ_n(x) = (2^n n! √π)^{-1/2} H_n(x) exp(-x^2/2) are eigenfunctions
        # with eigenvalue (-i)^n. So F[ψ_n](ξ) = (-i)^n ψ_n(ξ).
        # Therefore, if f(x) = Σ c_n ψ_n(x), then F(f)(ξ) = Σ c_n (-i)^n ψ_n(ξ).
        # For n=0: eigenvalue 1
        # For n=4: (-i)^4 = 1
        # For n=8: (-i)^8 = 1
        # For n=12: (-i)^12 = 1
        # So ALL even-degree Hermite functions in this set have eigenvalue 1!
        # Thus F(f)(ξ) = f(ξ) for this particular combination.
        # Wait, that means A(f) = A(F) and the product is A(f)^2.
        # Let me verify: (-i)^4 = (-i)^2 * (-i)^2 = (-1)(-1) = 1. Yes.
        # (-i)^8 = ((-i)^4)^2 = 1. (-i)^12 = ((-i)^4)^3 = 1.
        # So indeed F(f) = f for any linear combination of H_0, H_4, H_8, H_12 times exp(-x^2/2).
        # This simplifies things enormously: A(f)*A(F) = A(f)^2 where A(f) = ∫|f|/∫f^2 (up to normalization).
        # Actually A(f) = (1/||f||_2) ||f||_1 = ∫|f| / sqrt(∫f^2).
        # And A(F) = ∫|F| / sqrt(∫F^2). Since ||F||_2 = ||f||_2 by Parseval, and F=f in this case,
        # A(F) = A(f). So the product is A(f)^2.
        pass

    # Let's compute A(f)^2 = (∫|f| / sqrt(∫f^2))^2 = (∫|f|)^2 / ∫f^2
    
    # We'll use numerical integration on a sufficiently large grid
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    fvals = f(x)
    int_f2 = np.trapz(fvals**2, x)
    int_abs_f = np.trapz(np.abs(fvals), x)
    if int_f2 < 1e-30:
        return np.inf
    A = int_abs_f / np.sqrt(int_f2)
    return A * A  # This is A(f)*A(F) since A(F)=A(f)


def _objective(params):
    a0, a4, a8 = params
    try:
        val = _make_bound_evaluator(a0, a4, a8)
        if not np.isfinite(val):
            return 1e10
        return val
    except:
        return 1e10


def construct():
    start_time = time.time()
    deadline = start_time + 175  # Safety margin
    
    # Initial best: the seed values
    seed = np.array([-113/100, 1/25, 1/3240])
    best_params = seed.copy()
    best_val = _make_bound_evaluator(*seed)
    
    # Differential evolution for global optimization
    # Bounds: coefficients shouldn't be too large
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    if time.time() < deadline:
        try:
            result = differential_evolution(
                _objective, bounds, maxiter=50, popsize=20, 
                seed=42, tol=1e-6, polish=True
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Local refinement using L-BFGS-B
    if time.time() < deadline - 10:
        try:
            local_result = minimize(
                _objective, best_params, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if local_result.fun < best_val:
                best_val = local_result.fun
                best_params = local_result.x
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            polish_result = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_params = polish_result.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
