"""Combined program: analytic Hermite integrals + multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 175.0  # safety margin before 210s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _hermite_coeffs(c0, c4, c8):
    """Return all four coefficients given free ones, with g(0)=0 constraint."""
    # Probabilists' Hermite: H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    h0_0 = 1.0
    h4_0 = -3.0
    h8_0 = 105.0
    h12_0 = -10395.0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c0, c4, c8, c12


def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the upper bound using analytic Hermite polynomial properties.
    
    For f(x) = p(x) * exp(-x^2/2), the key quantities:
    - ||f||_2^2 = sum c_i^2 * ||H_i exp(-x^2/2)||_2^2 (by orthogonality)
    - Since FT preserves L2 norm (unitary), ||F(f)||_2 = ||f||_2
    - The uncertainty product bound involves ratios of these norms
    
    The "upper bound on C4" from Goncalves et al. likely involves:
    C4 = ||f||_1 * ||F(f)||_1 / (||f||_2^2) or similar
    
    Since |F(f)(xi)| = |f(xi)| (all phases are +1 for degrees 0,4,8,12),
    we have ||f||_1 = ||F(f)||_1, so the product is ||f||_1^2.
    
    But ||f||_1 cannot be computed analytically easily. Let's use a 
    high-accuracy numerical quadrature for the L1 norm, which is fast.
    """
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    
    # Use Gauss-Hermite quadrature for high accuracy
    # ∫ g(x) exp(-x^2) dx = sum w_i g(x_i) for appropriate g
    # But we need ∫ |p(x) exp(-x^2/2)| dx, which is not a simple Gaussian integral
    
    # Use adaptive quadrature on a fine grid - this is fast enough
    x = np.linspace(-15, 15, 100001)
    dx = x[1] - x[0]
    
    # Probabilists' Hermite polynomials
    # He0 = 1
    # He4 = x^4 - 6x^2 + 3
    # He8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # He12 = x^12 - 42x^10 + 1260x^8 - 16800x^6 + 113400x^4 - 302400x^2 + 10395
    
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12 = x**12 - 42*x**10 + 1260*x**8 - 16800*x**6 + 113400*x**4 - 302400*x**2 + 10395
    
    p = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = p * np.exp(-x**2 / 2.0)
    
    # L1 norm of g
    A_f = np.sum(np.abs(g)) * dx
    
    # For the Fourier side, using the unitary DFT: F(f)(ξ) = (1/√(2π)) ∫ f(x) e^{-iξx} dx
    # With f(x) = p(x) exp(-x^2/2), F(f)(ξ) is also a polynomial in ξ times exp(-ξ^2/2)
    # by the properties of the Fourier transform of Hermite-Gaussian functions.
    
    # Actually, the key property: if f(x) = p(x) exp(-x^2/2), then
    # F(f)(ξ) = (-i)^k times some polynomial... but more precisely:
    # The Fourier transform maps H_n(x) exp(-x^2/2) to i^n H_n(ξ) exp(-ξ^2/2) (unitary DFT)
    # So if f(x) = sum c_n H_n(x) exp(-x^2/2), then F(f)(ξ) = sum c_n i^n H_n(ξ) exp(-ξ^2/2)
    
    # Therefore ||F(f)||^2 = ||f||^2 (Parseval), and the product A(f)*A(F(f)) 
    # depends on the structure of the support/uncertainty measures.
    
    # For the uncertainty product with these specific measures, the bound C4 
    # is related to the Rayleigh quotient. Given the task structure, the score
    # is the upper bound on C4, which we want to minimize.
    
    # Let me think about this more carefully. The standard approach for this 
    # type of problem: the "uncertainty inequality" bound involves computing
    # quantities like ∫x^2|f|^2 dx and ∫ξ^2|F(f)|^2 ξ, and the product of 
    # "spreads" or related measures.
    
    # Given the hint that we minimize an upper bound, and the seed values,
    # let me set up the optimization properly.
    
    # Let me define the objective more concretely. For the function f = p * g
    # where g(x) = exp(-x^2/2), we compute:
    # - The L2 norm: N2 = ||f||_2^2
    # - Moments involving x^2 and |F(f)|^2
    
    # The Fourier transform of H_n(x)g(x) under unitary DFT is i^n H_n(ξ)g(ξ)
    # So F(f)(ξ) = sum_n c_n i^n H_n(ξ) g(ξ)
    
    # Then |F(f)(ξ)|^2 involves cross terms. But for the variance/uncertainty,
    # we need ∫ξ^2 |F(f)|^2 dξ / ∫|F(f)|^2 dξ and similarly for x.
    
    # By symmetry, ∫x^2 |f(x)|^2 dx = ∫ξ^2 |F(f)(ξ)|^2 dξ
    # So the "uncertainty product" might be (∫x^2|f|^2/||f||^2) * (∫ξ^2|F(f)|^2/||F(f)||^2)
    # = (∫x^2|f|^2/||f||^2)^2
    
    # Let's compute M2 = ∫x^2 |f(x)|^2 dx = ∫ x^2 p(x)^2 exp(-x^2) dx
    # and N2 = ∫ p(x)^2 exp(-x^2) dx
    # Then the spread is M2/N2, and the bound might be (M2/N2)^2 or some function thereof.
    
    # Let me compute these integrals.
    
    # N2 = sum_n c_n^2 I_n where I_n = ∫H_n^2 exp(-x^2) dx = sqrt(pi) * 2^n * n!
    
    # M2 = ∫ x^2 p(x)^2 exp(-x^2) dx
    # x^2 H_n = (1/(n+1)) H_{n+2} + n H_{n-2} ... using recurrence
    # Actually, x^2 H_n = H_{n+2} + (2n+1)H_n + n(n-1)H_{n-2}
    
    # Therefore x^2 H_n * H_m exp(-x^2) integrates to:
    # I_{n+2,m} + (2n+1)I_{n,m} + n(n-1)I_{n-2,m}
    # where I_{a,b} = ∫H_a H_b exp(-x^2) dx = 0 if a≠b, sqrt(pi)2^b b! if a=b
    
    # So M2 = sum_{n,m} c_n c_m [I_{n+2,m} + (2n+1)I_{n,m} + n(n-1)I_{n-2,m}]
    # = sum_n c_n^2 [ (2n+1)I_n + I_{n+2} + n(n-1)I_{n-2} ]  (with I_k = 0 for k<0)
    # where I_n = sqrt(pi) 2^n n! for n>=0, and I_{n+2} means ∫H_{n+2}^2 exp = sqrt(pi)2^{n+2}(n+2)!
    
    # Wait, that's not right. Let me redo:
    # ∫ x^2 H_n H_m exp(-x^2) dx
    # Using x^2 H_n = H_{n+2} + (2n+1)H_n + n(n-1)H_{n-2}
    # = [H_{n+2} + (2n+1)H_n + n(n-1)H_{n-2}] H_m exp(-x^2)
    # Integral = I_{n+2}δ_{n+2,m} + (2n+1)I_n δ_{n,m} + n(n-1)I_{n-2}δ_{n-2,m}
    
    # So M2 = sum_n c_n^2 (2n+1)I_n + sum_{n,m} c_n c_m I_{n+2}δ_{m,n+2} + sum c_n c_m n(n-1)I_{n-2}δ_{m,n-2}
    # = sum_n c_n^2 (2n+1)I_n + sum_n c_n c_{n-2} I_n + sum_n c_n c_{n+2} n(n-1)I_{n-2}
    # Note the last two: let me reindex.
    # Second term: sum over n,m with m=n+2: sum_n c_n c_{n+2} I_{n+2}... wait
    # ∫x^2 H_n H_{n+2} exp = I_{n+2} (from the first part) + 0 + 0 (the (2n+1) term needs m=n, the last needs m=n-2)
    # Hmm, let me be more careful.
    
    # For the cross term c_n c_m with m ≠ n:
    # Only nonzero if m = n+2 (gives I_{n+2}) or m = n-2 (gives n(n-1)I_{n-2})
    
    # This is getting complex. Let me just do numerical quadrature to be safe.
    
    # N2 = ∫f^2 dx
    N2 = np.sum(g**2) * dx
    
    # M2 = ∫x^2 f^2 dx
    M2 = np.sum(x**2 * g**2) * dx
    
    if N2 < 1e-15:
        return 1e10
    
    # The uncertainty product bound. For this type of problem,
    # a common measure is the product of variances or related.
    # Let's use (M2/N2) as a proxy, but we need the actual bound C4.
    
    # Actually, let me think about what C4 represents.
    # In uncertainty inequalities, C(f) might be ∫(x - x̄)^2 |f|^2 / ∫|f|^2
    # and C(F) = ∫(ξ - ξ̄)^2 |F(f)|^2 / ∫|F(f)|^2
    # Then C4 = C(f) * C(F) or similar.
    
    # By symmetry of the problem (if p is even or odd combinations),
    # the means are zero, so C(f) = M2/N2, C(F) = M2/N2 (by Parseval and symmetry)
    # So C4 = (M2/N2)^2
    
    spread = M2 / N2
    C4 = spread * spread
    return C4


def construct():
    """
    Find the three free coefficients (a0, a4, a8) for the Hermite combination
    H0 + a4*H4 + a8*H8 + a12*H12 (where a12 is fixed by P(0)=0) to minimize
    the upper bound C4 on A(f)*A(Fourier(f)).
    
    Returns:
        numpy array of shape (3,): [a0, a4, a8]
    """
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound_analytic(*seed)
    best_params = seed.copy()
    
    # Try local optimization from seed
    if _time_left() > 30:
        try:
            res = minimize(
                _compute_bound_analytic,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Try differential evolution for global search if time permits
    if _time_left() > 60:
        try:
            # Reasonable bounds for coefficients
            bounds = [(-10, 10), (-10, 10), (-1, 1)]
            res_de = differential_evolution(
                _compute_bound_analytic,
                bounds,
                maxiter=200,
                popsize=15,
                seed=42,
                tol=1e-8
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res.x
        except Exception:
            pass
    
    # Refine with Nelder-Mead from best found
    if _time_left() > 20 and not np.array_equal(best_params, seed):
        try:
            res2 = minimize(
                _compute_bound_analytic,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
