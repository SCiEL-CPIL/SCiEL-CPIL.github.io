"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _compute_bound(a0, a4, a8):
    """Compute the upper bound on C4 for given Hermite coefficients.
    
    The test function is f(x) = P(x) * exp(-x^2/2) where
    P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    with a12 determined by P(0) = 0.
    
    The upper bound on A(f)*A(F(f)) involves C4 which depends on
    the L2 norm of f and its Fourier transform, plus boundary terms.
    """
    # Physicists' Hermite polynomials: H_n(x)
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 32256x^10 + 84672x^8 - 108864x^6 + 64512x^4 - 14336x^2 + 10368
    
    # P(0) = a0*1 + a4*12 + a8*168 + a12*10368 = 0
    # => a12 = -(a0 + 12*a4 + 168*a8) / 10368
    a12 = -(a0 + 12.0*a4 + 168.0*a8) / 10368.0
    
    # The key quantity for the uncertainty bound is related to
    # the ratio of certain integrals. For f = P*exp(-x^2/2):
    # 
    # ||f||_2^2 = integral P(x)^2 * exp(-x^2) dx
    # ||f'||_2^2 = integral (P'(x) - x*P(x))^2 * exp(-x^2) dx
    # 
    # The Fourier transform of exp(-x^2/2) is exp(-xi^2/2), and
    # the Fourier transform of P(x)*exp(-x^2/2) can be computed.
    #
    # For the uncertainty inequality bound, we need:
    # C4 = (integral x^2 |f(x)|^2 dx) * (integral xi^2 |F(f)(xi)|^2 dxi)
    #     / (integral |f(x)|^2 dx)^2
    #
    # But actually the bound on A(f)*A(F(f)) uses a different formulation.
    # Let me use the approach from the literature:
    #
    # For f in S, the bound is:
    # A(f) * A(F) <= sqrt( (integral x^2 |f|^2) * (integral |f'|^2) ) / (integral |f|^2)
    #
    # Actually, let me compute the relevant quantities using Gaussian integrals.
    # 
    # For P(x) = sum_k c_k * H_k(x) (physicists' convention):
    # integral H_m(x) H_n(x) exp(-x^2) dx = 2^n n! sqrt(pi) delta_{mn}
    #
    # So ||f||_2^2 = sum_k c_k^2 * 2^k * k! * sqrt(pi)
    
    # Coefficients in Hermite basis
    coeffs = np.array([a0, a4, a8, a12])
    degrees = np.array([0, 4, 8, 12])
    
    # ||f||_2^2
    norm_sq = np.sum(coeffs**2 * 2.0**degrees * np.array([1, 24, 40320, 479001600]) * np.sqrt(np.pi))
    
    # For the momentum term, we need integral |f'|^2 or integral xi^2 |F|^2
    # Using the property: for f = P*exp(-x^2/2), 
    # F(f)(xi) = P(-i*d/dxi) * exp(-xi^2/2) ... this gets complicated.
    #
    # Alternative: use the fact that for the Hermite-Gaussian basis,
    # the uncertainty product can be computed from the second moments.
    #
    # Let me use a direct numerical approach with quadrature.
    
    # Use Gauss-Hermite quadrature for efficiency
    # integral g(x) exp(-x^2) dx ≈ sum w_i g(x_i)
    n_pts = 200
    x, w = np.polynomial.hermite.hermgauss(n_pts)
    w = w * np.sqrt(np.pi)
    
    # Build P(x)
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 32256x^10 + 84672x^8 - 108864x^6 + 64512x^4 - 14336x^2 + 10368
    
    x2 = x**2
    H0 = np.ones_like(x)
    H4 = 16*x2**2 - 48*x2 + 12
    H8 = 256*x2**4 - 1344*x2**3 + 2688*x2**2 - 1344*x2 + 168
    H12 = 4096*x2**6 - 32256*x2**5 + 84672*x2**4 - 108864*x2**3 + 64512*x2**2 - 14336*x2 + 10368
    
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    f = P * np.exp(-x2/2)
    f_sq = f**2
    
    # ||f||_2^2
    norm2 = np.sum(w * f_sq)
    
    # Second moment: integral x^2 |f|^2
    mom2 = np.sum(w * x2 * f_sq)
    
    # For the Fourier side, use Parseval: integral xi^2 |F(f)|^2 dxi = integral |f'|^2 dx
    # f'(x) = P'(x)*exp(-x^2/2) - x*P(x)*exp(-x^2/2) = (P'(x) - x*P(x))*exp(-x^2/2)
    dP = (64*x2**3 - 96*x) * a4 + (2048*x2**3 - 8064*x2 + 10752*x - 1344) * a8 \
         + (49152*x2**5 - 322560*x2**3 + 677376*x2 - 14336) * a12
    # Wait, let me recompute derivatives properly
    # d/dx H4 = 64x^3 - 96x
    # d/dx H8 = 2048x^7 - 8064x^5 + 10752x^3 - 1344x
    # d/dx H12 = 49152x^11 - 322560x^9 + 677376x^7 - 653184x^5 + 258048x^3 - 14336x
    
    dP = (64*x**3 - 96*x) * a4 \
         + (2048*x**7 - 8064*x**5 + 10752*x**3 - 1344*x) * a8 \
         + (49152*x**11 - 322560*x**9 + 677376*x**7 - 653184*x**5 + 258048*x**3 - 14336*x) * a12
    
    fp = (dP - x*P) * np.exp(-x2/2)
    fp_sq = fp**2
    momentum2 = np.sum(w * fp_sq)
    
    # The uncertainty bound: A(f)*A(F) <= sqrt(mom2 * momentum2) / norm2
    # But we need to be careful about what A(f) means.
    # 
    # From the task context, C4 is the constant in the bound.
    # The bound is typically: A(f) * A(F) <= C
    # where C = sqrt( integral x^2|f|^2 * integral |f'|^2 ) / integral|f|^2
    #
    # Actually, re-reading: "score is the upper bound on C4"
    # and "reduce the upper bound on A(f)*A(Fourier(f))"
    #
    # I think C4 is the value of the bound itself.
    
    if norm2 < 1e-30:
        return 1e10
    
    bound = np.sqrt(mom2 * momentum2) / norm2
    return bound


def _objective(coeffs):
    """Objective function for optimization."""
    a0, a4, a8 = coeffs
    return _compute_bound(a0, a4, a8)


def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Must finish within 180s, use 180s budget
    
    # Initial guess from prior state of the art
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_bound(*best_coeffs)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_coeffs
    
    # Try differential evolution for global exploration
    try:
        res_de = differential_evolution(
            _objective,
            bounds=[(-50, 50), (-50, 50), (-50, 50)],
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=True,
            popsize=15,
            init='sobol'
        )
        elapsed = time.time() - start_time
        if res_de.fun < best_val:
            best_val = res_de.fun
            best_coeffs = res_de.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return best_coeffs
    
    # Local refinement with L-BFGS-B
    try:
        res_local = minimize(
            _objective,
            best_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if res_local.fun < best_val:
            best_val = res_local.fun
            best_coeffs = res_local.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_coeffs
    
    # Final high-precision polish with Nelder-Mead
    try:
        res_nm = minimize(
            _objective,
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
        )
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_coeffs = res_nm.x
    except Exception:
        pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
