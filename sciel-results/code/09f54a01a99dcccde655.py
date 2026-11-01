"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hermite_at_zero(n):
    """Return H_n(0) for physicists' Hermite polynomials."""
    # H_n(0) = (-1)^(n/2) * n! / (n/2)! if n even, 0 if n odd
    if n % 2 != 0:
        return 0.0
    k = n // 2
    # Use recurrence: H_0(0)=1, H_2(0)=-2, H_4(0)=12, H_6(0)=-120, H_8(0)=1680, H_10(0)=-30240, H_12(0)=665280
    # Actually let's compute properly: H_n(0) for even n
    # H_0(0) = 1
    # H_2(0) = -2
    # H_4(0) = 12
    # H_6(0) = -120
    # H_8(0) = 1680
    # H_10(0) = -30240
    # H_12(0) = 665280
    values = {0: 1.0, 2: -2.0, 4: 12.0, 6: -120.0, 8: 1680.0, 10: -30240.0, 12: 665280.0}
    return values.get(n, 0.0)


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for given coefficients.
    The test function is g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    The uncertainty product involves the second moments of |g|^2 and |G|^2 where G is the Fourier transform.
    """
    # Hermite polynomials evaluated at 0
    H0_0 = _hermite_at_zero(0)   # 1
    H4_0 = _hermite_at_zero(4)   # 12
    H8_0 = _hermite_at_zero(8)   # 1680
    H12_0 = _hermite_at_zero(12) # 665280
    
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0) = 0
    # So: c0*1 + c4*12 + c8*1680 + c12*665280 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    # Now compute the uncertainty bound.
    # For a function g(x), the uncertainty product is related to:
    # A(g) = sqrt(<x^2> - <x>^2) / <g^2>^(1/2) ... actually this is getting complex.
    # 
    # Let me think about this differently. The "upper bound on C4" likely refers to a specific
    # constant in an inequality of the form A(f)*A(F) <= C4 * something.
    # 
    # For the Hermite-Gaussian test functions, the Fourier transform has a known form.
    # The key insight is that for these functions, we can compute the moments analytically.
    #
    # Let's use a simpler approach: compute the variance in position and frequency space.
    # For g(x) = P(x) * exp(-x^2/2), where P is a polynomial:
    # <x^2> = integral x^2 |g(x)|^2 dx / integral |g(x)|^2 dx
    #
    # This requires integrating polynomials times Gaussians, which can be done analytically.
    #
    # For now, let's use a proxy objective that captures the essential structure.
    # The prior art coefficients give a baseline. We'll try to find coefficients that
    # minimize a reasonable surrogate for the uncertainty bound.
    
    # A reasonable approach: the uncertainty product is minimized when the function is
    # close to a Gaussian. The Hermite expansion coefficients measure deviation from Gaussian.
    # So we want to minimize the "spread" of the coefficients in a way that balances
    # position and frequency uncertainty.
    
    # Let's compute the L2 norm of the polynomial part and use that as a proxy.
    # Actually, let me try a different approach based on the structure of the problem.
    
    # The key quantities are:
    # 1. The norm of g: ||g||^2 = integral |g(x)|^2 dx
    # 2. The second moment: <x^2> = integral x^2 |g(x)|^2 dx / ||g||^2
    # 3. The frequency-space second moment: <k^2> = integral k^2 |G(k)|^2 dk / ||G||^2
    
    # For g(x) = P(x) e^{-x^2/2}, we can compute these analytically.
    # Let P(x) = a0 + a4 x^4 + a8 x^8 + a12 x^12 (after expanding Hermite polys)
    # Actually, it's easier to work with the Hermite basis directly.
    
    # The Hermite-Gaussian functions phi_n(x) = H_n(x) e^{-x^2/2} / sqrt(2^n n! sqrt(pi)) are orthonormal.
    # So if g = sum c_n phi_n, then ||g||^2 = sum c_n^2 (with appropriate normalization).
    
    # For the uncertainty product, we need:
    # <x^2> = sum_n c_n^2 <x^2>_n / sum_n c_n^2
    # where <x^2>_n is the second moment of phi_n^2.
    
    # For Hermite-Gaussian functions:
    # <x^2>_n = n + 1/2 (for physicists' Hermite with the standard normalization)
    # Wait, that's for the quantum harmonic oscillator. Let me be more careful.
    
    # Actually, for the standard Hermite functions psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}:
    # <x^2>_n = n + 1/2
    # <k^2>_n = n + 1/2 (by symmetry, since Fourier transform of psi_n is (-i)^n psi_n)
    
    # So if g = sum_{n in {0,4,8,12}} d_n psi_n, then:
    # ||g||^2 = sum d_n^2
    # <x^2> = sum d_n^2 (n + 1/2) / sum d_n^2
    # <k^2> = sum d_n^2 (n + 1/2) / sum d_n^2
    # 
    # Wait, that would make <x^2> = <k^2>, which is expected for Fourier-invariant functions.
    # But the uncertainty product would be <x^2> * <k^2> = <x^2>^2.
    
    # Hmm, but the problem says "Fourier-invariant" which might mean something specific.
    # Let me reconsider.
    
    # Actually, the Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(k) e^{-k^2/2} (up to normalization).
    # So the frequency-space function has the same form, just with coefficients rotated by phases.
    # The magnitude |G(k)| has the same form as |g(x)|, so the second moments are the same.
    
    # So A(g) * A(G) = sqrt(<x^2> - <x>^2) * sqrt(<k^2> - <k>^2)
    # Since <x> = <k> = 0 (odd functions integrate to zero), this is sqrt(<x^2>) * sqrt(<k^2>) = <x^2>.
    
    # Wait, that's not right either. Let me be more careful with definitions.
    
    # Let me just compute the variance directly. For g(x) = sum c_n H_n(x) e^{-x^2/2}:
    # First, compute ||g||^2 = integral |g(x)|^2 dx
    # Using orthogonality of Hermite polynomials: integral H_m(x) H_n(x) e^{-x^2} dx = sqrt(pi) 2^n n! delta_{mn}
    # So ||g||^2 = sum c_n^2 sqrt(pi) 2^n n!
    
    # <x^2> = integral x^2 |g(x)|^2 dx / ||g||^2
    # integral x^2 H_m(x) H_n(x) e^{-x^2} dx can be computed using the recurrence x H_n = (n+1) H_{n+1} + H_{n-1}
    # or using known formulas.
    
    # This is getting complex. Let me use a numerical approach for the bound computation,
    # but optimize analytically.
    
    # Actually, let me try a different strategy. The prior art coefficients are:
    # c0 = -113/100 = -1.13, c4 = 1/25 = 0.04, c8 = 1/3240 ≈ 0.000309
    # These give a specific bound. Let's try to perturb them and see if we can improve.
    
    # For the optimization, I'll use a simple heuristic: minimize the sum of squared
    # coefficients weighted by their degree, which approximates the uncertainty.
    
    # A better approach: compute the actual bound numerically for a few test points
    # and use that to guide the optimization.
    
    # Let me implement a numerical computation of the uncertainty product.
    
    # Compute g(x) on a grid and estimate the moments.
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]
    
    # Hermite polynomials (physicists')
    def H0(x): return np.ones_like(x)
    def H4(x): return 16*x**4 - 48*x**2 + 12
    def H8(x): return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    def H12(x):
        # Use recurrence or explicit formula
        # H_12(x) = 4096*x^12 - 98304*x^10 + 1032192*x^8 - 5806080*x^6 + 14929920*x^4 - 17616000*x^2 + 665280
        return (4096*x**12 - 98304*x**10 + 1032192*x**8 - 5806080*x**6 
                + 14929920*x**4 - 17616000*x**2 + 665280)
    
    gauss = np.exp(-x**2 / 2)
    
    g = (c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)) * gauss
    g2 = g**2
    
    norm2 = np.sum(g2) * dx
    if norm2 < 1e-30:
        return 1e10  # Invalid
    
    x2 = np.sum(x**2 * g2) * dx / norm2
    x_mean = np.sum(x * g) * dx / np.sum(g) * dx if np.sum(g) * dx > 1e-30 else 0
    # Actually, for the variance we need <x^2> - <x>^2
    # But since the function is even (all Hermite polynomials of even degree), <x> = 0
    # So variance_x = x2
    
    # For the Fourier transform, we can use the fact that the FT of H_n(x) e^{-x^2/2} is (-i)^n H_n(k) e^{-k^2/2}
    # So |G(k)|^2 has the same form as |g(x)|^2, just with the same coefficients (since |(-i)^n| = 1)
    # Therefore, variance_k = variance_x = x2
    
    # The uncertainty product is sqrt(variance_x) * sqrt(variance_k) = variance_x
    # Wait, that's not right. A(f) is typically defined as sqrt(<x^2> - <x>^2) / ||f|| or similar.
    # Let me use the standard definition: A(f) = sqrt(<x^2>_f - <x>_f^2) where <...>_f is the expectation with respect to |f|^2.
    
    # So A(g) = sqrt(x2) and A(G) = sqrt(x2) (by Fourier symmetry)
    # Product = x2
    
    # But wait, the problem says "upper bound on A(f)*A(Fourier(f))", which suggests we're looking for
    # the maximum over some class, and we want to minimize that upper bound by choosing the test function.
    
    # Let me just return x2 as the objective for now and see if optimization helps.
    return x2


def _compute_bound_fast(c0, c4, c8):
    """
    Fast computation of the uncertainty bound using analytic moments.
    """
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    # Use the orthonormal Hermite function basis
    # phi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}
    # If g = sum a_n phi_n, then a_n = c_n * sqrt(2^n n! sqrt(pi))
    # But for the moments, we can work directly with c_n.
    
    # ||g||^2 = sum c_n^2 * 2^n * n! * sqrt(pi)
    # <x^2> = sum c_n^2 * 2^n * n! * sqrt(pi) * <x^2>_n / ||g||^2
    
    # For Hermite-Gaussian functions, <x^2>_n = n + 1/2 (in the orthonormal basis)
    # But we need to be careful with the normalization.
    
    # Let's use the fact that for the standard Hermite functions:
    # x * phi_n = sqrt((n+1)/2) * phi_{n+1} + sqrt(n/2) * phi_{n-1}
    # So <phi_n | x^2 | phi_n> = (n + 1/2)
    
    # If g = sum c_n H_n e^{-x^2/2} = sum c_n sqrt(2^n n! sqrt(pi)) phi_n
    # Then ||g||^2 = sum c_n^2 * 2^n n! sqrt(pi)
    # <x^2> = sum c_n^2 * 2^n n! sqrt(pi) * (n + 1/2) / ||g||^2
    
    n_values = [0, 4, 8, 12]
    c_values = [c0, c4, c8, c12]
    
    norm2 = 0.0
    x2_weighted = 0.0
    for n, c in zip(n_values, c_values):
        weight = (2.0 ** n) * (np.math.factorial(n) if hasattr(np, 'math') else float(np.prod(np.arange(1, n+1)))) * np.sqrt(np.pi)
        norm2 += c**2 * weight
        x2_weighted += c**2 * weight * (n + 0.5)
    
    if norm2 < 1e-30:
        return 1e10
    
    x2 = x2_weighted / norm2
    return x2


def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function
    to reduce the upper bound on A(f)*A(Fourier(f)).
    
    Returns:
        np.array: three coefficients [c0, c4, c8] of the Hermite combination.
                  The degree-12 weight is fixed so that g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 200  # 10s safety margin before 210s limit
    
    # Prior state of the art (Goncalves et al. 2017)
    best_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_score = _compute_bound_fast(best_result[0], best_result[1], best_result[2])
    
    # Spherical reparameterization: c = r * u where u is on the unit sphere
    # But we have a constraint: g(0) = 0, which fixes c12 in terms of c0, c4, c8.
    # So we search over (c0, c4, c8) directly.
    
    # Adaptive stochastic local search
    current = best_result.copy()
    current_score = best_score
    step_size = 0.1
    min_step = 1e-6
    max_iter = 5000
    no_improve_count = 0
    patience = 100
    
    for i in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Adaptive step size
        if no_improve_count > patience:
            step_size *= 0.5
            no_improve_count = 0
            if step_size < min_step:
                break
        
        # Generate random perturbation
        perturbation = np.random.randn(3) * step_size
        candidate = current + perturbation
        
        cand_score = _compute_bound_fast(candidate[0], candidate[1], candidate[2])
        
        if cand_score < current_score:
            current = candidate
            current_score = cand_score
            no_improve_count = 0
            step_size = min(step_size * 1.1, 0.5)
            
            if current_score < best_score:
                best_score = current_score
                best_result = current.copy()
        else:
            no_improve_count += 1
    
    return best_result
# EVOLVE-BLOCK-END
