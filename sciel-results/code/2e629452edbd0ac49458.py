"""Minimize the upper bound C4 for the uncertainty inequality using Hermite-Gaussian test functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial, eval_hermite

_start_time = time.time()
_TIME_LIMIT = 190.0  # safety margin before 210s kill


def _hermite_even_vals(n_list, x):
    """Evaluate physicist's Hermite polynomials at x for given even orders."""
    return np.array([eval_hermite(n, x) for n in n_list])


def _compute_bound_numeric(a):
    """
    Compute the uncertainty product bound using numerical quadrature.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    """
    a0, a4, a8 = a
    # Enforce g(0) = 0
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    ns = [0, 4, 8, 12]
    coeffs = np.array([a0, a4, a8, a12])
    
    # Numerical integration over a sufficiently large domain
    x = np.linspace(-15, 15, 10000)
    dx = x[1] - x[0]
    
    g = np.zeros_like(x)
    for ci, ni in zip(coeffs, ns):
        g += ci * eval_hermite(ni, x)
    g *= np.exp(-x**2 / 2.0)
    
    norm_sq = np.trapz(g**2, x)
    if norm_sq < 1e-30:
        return 1e10
    
    # x^2 moment
    x2_moment = np.trapz(x**2 * g**2, x)
    
    # Fourier transform of g: F(k) = integral g(x) exp(-2πikx) dx
    # For Gaussian-modulated Hermite, we can use the fact that FT of H_n(x)exp(-x²/2)
    # is related to i^n H_n(k) exp(-k²/2) up to constants
    # Let's compute numerically
    k = np.linspace(-15, 15, 10000)
    dk = k[1] - k[0]
    
    # Compute G(k) = FT[g](k) using the relation for Hermite functions
    # FT of H_n(x) e^{-x²/2} = i^n H_n(k) e^{-k²/2} * (2π)^{-1/2} * something
    # Actually, let's just do it numerically
    G = np.zeros_like(k, dtype=complex)
    for ki in k:
        G[ki] = np.trapz(g * np.exp(-2j * np.pi * ki * x), x)
    
    # This is too slow for 10000 points. Let's use the analytic relation.
    # FT of H_n(x) e^{-x²/2} dx = i^n * (2π)^{1/2} * H_n(2πk) e^{-(2πk)²/2} / sqrt(2π)
    # Actually the standard result: if ψ_n(x) = (2^n n! sqrt(π))^{-1/2} H_n(x) e^{-x²/2}
    # then FT[ψ_n](k) = i^n ψ_n(k) (up to sign convention)
    
    # Let me use a different approach. The bound on C4 involves:
    # A(f) * A(F) where A involves second moments.
    # For our test function, the bound is:
    # bound = (∫ x² |g|² dx) * (∫ k² |G(k)|² dk) / (∫ |g|² dx)²
    
    # Using Parseval and the Hermite properties:
    # ∫ |G(k)|² dk = ∫ |g(x)|² dx (with proper normalization)
    # ∫ k² |G(k)|² dk can be computed from the coefficients
    
    # For H_n(x) e^{-x²/2}, the Fourier transform (unitary) is i^n H_n(k) e^{-k²/2}
    # So if g(x) = Σ c_n H_n(x) e^{-x²/2}, then 
    # G(k) = Σ c_n i^n H_n(k) e^{-k²/2} (up to a phase/convention)
    
    # ∫ k² |G(k)|² dk = Σ_{m,n} c_m c̄_n i^m (-i)^n ∫ k² H_m(k) H_n(k) e^{-k²} dk
    # = Σ_{m,n} c_m c_n (-1)^{(m-n)/2} ∫ k² H_m(k) H_n(k) e^{-k²} dk (for even m,n)
    # By orthogonality and x² H_n relation, this equals the same as ∫ x² |g|² dx
    # when m=n terms dominate, but cross terms matter.
    
    # Let's just compute it analytically using the same moment_x2 function
    sqrt_pi = np.sqrt(np.pi)
    
    def norm_sq_n(n):
        return sqrt_pi * (2.0 ** n) * factorial(n, exact=False)
    
    def moment_x2(m, n):
        """∫ x² H_m(x) H_n(x) e^{-x²} dx"""
        result = 0.0
        if m + 2 == n:
            result += 0.5 * sqrt_pi * (2.0**n) * factorial(n, exact=False)
        if m == n:
            result += m * sqrt_pi * (2.0**m) * factorial(m, exact=False)
        if m >= 2 and m - 2 == n:
            result += (m / 2.0) * sqrt_pi * (2.0**n) * factorial(n, exact=False)
        return result
    
    norm_sq_ana = 0.0
    for ci, ni in zip(coeffs, ns):
        norm_sq_ana += ci**2 * norm_sq_n(ni)
    
    x2_moment_ana = 0.0
    for i, (ci, ni) in enumerate(zip(coeffs, ns)):
        for j, (cj, nj) in enumerate(zip(coeffs, ns)):
            x2_moment_ana += ci * cj * moment_x2(ni, nj)
    
    # The bound: (∫ x²|g|²)(∫ k²|G|²) / (∫|g|²)²
    # By the Hermite-Fourier symmetry, ∫ k²|G|² = ∫ x²|g|² (same expression)
    # So bound = x2_moment_ana² / norm_sq_ana
    
    if norm_sq_ana < 1e-30:
        return 1e10
    
    bound = x2_moment_ana**2 / norm_sq_ana
    return bound


def _objective(a):
    return _compute_bound_numeric(a)


def construct():
    """
    Returns three coefficients [a0, a4, a8] for the Hermite combination.
    a12 is determined by g(0)=0.
    Minimizes the upper bound on C4.
    """
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        return np.array([-1.13, 0.04, 0.0003086])
    
    # Initial best guess
    best_a = np.array([-1.13, 0.04, 0.0003086])
    best_val = _compute_bound_numeric(best_a)
    
    # Define bounds for differential evolution
    bounds = [(-10, 10), (-10, 10), (-1, 1)]
    
    # Use differential evolution for global search
    if time.time() - _start_time < _TIME_LIMIT - 60:
        try:
            de_result = differential_evolution(
                _objective, bounds, 
                maxiter=50, popsize=20, 
                seed=42, tol=1e-10, 
                polish=True
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_a = de_result.x
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() - _start_time < _TIME_LIMIT - 10:
        try:
            nm_result = minimize(
                _objective, best_a, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if nm_result.fun < best_val:
                best_val = nm_result.fun
                best_a = nm_result.x
        except Exception:
            pass
    
    # Additional local searches from different starting points
    if time.time() - _start_time < _TIME_LIMIT - 5:
        rng = np.random.default_rng(123)
        for _ in range(20):
            if time.time() - _start_time > _TIME_LIMIT - 3:
                break
            x0 = rng.uniform(-5, 5, size=3)
            try:
                res = minimize(_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res.fun < best_val:
                    best_val = res.fun
                    best_a = res.x
            except Exception:
                continue
    
    return best_a
# EVOLVE-BLOCK-END
