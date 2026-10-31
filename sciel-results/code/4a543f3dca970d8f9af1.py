"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, brentq
import time


def _hermite_coeffs(n):
    """Return coefficients for the probabilist's Hermite polynomial He_n(x)
    such that He_n(x) = sum_{k=0}^{n//2} (-1)^k * n! / (k! * 2^k * (n-2k)!) * x^{n-2k}.
    We return a list of (power, coeff) pairs for powers n, n-2, ..., 0."""
    coeffs = []
    for k in range(n // 2 + 1):
        # coeff = (-1)^k * n! / (k! * 2^k * (n-2k)!)
        import math
        c = ((-1) ** k) * math.factorial(n) / (math.factorial(k) * (2 ** k) * math.factorial(n - 2 * k))
        coeffs.append((n - 2 * k, c))
    return coeffs


def _g_val(x, a, b, c):
    """Evaluate g(x) = (a*H0 + b*H4 + c*H8 + d*H12) * exp(-x^2/2)
    where d is chosen so that g(0) = 0."""
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 1260x^8 - 12600x^6 + 55440x^4 - 100800x^2 + 10395
    
    # At x=0:
    # H0(0) = 1
    # H4(0) = 3
    # H8(0) = 105
    # H12(0) = 10395
    d = -(a * 1 + b * 3 + c * 105) / 10395
    
    x2 = x * x
    h0 = 1.0
    h4 = x2 * x2 - 6 * x2 + 3
    h8 = x2**4 - 28 * x2**3 + 210 * x2**2 - 420 * x2 + 105
    h12 = x2**6 - 84 * x2**5 + 1260 * x2**4 - 12600 * x2**3 + 55440 * x2**2 - 100800 * x2 + 10395
    
    poly = a * h0 + b * h4 + c * h8 + d * h12
    return poly * np.exp(-x2 / 2)


def _g_sq_integral(a, b, c, n_points=10000):
    """Compute ||g||_2^2 = integral of g(x)^2 dx over R.
    Use Gauss-Hermite quadrature for accuracy."""
    # Gauss-Hermite quadrature: integral f(x) e^{-x^2} dx ≈ sum w_i f(x_i)
    # We have g(x) = P(x) e^{-x^2/2}, so g(x)^2 = P(x)^2 e^{-x^2}
    # So integral g(x)^2 dx = integral P(x)^2 e^{-x^2} dx
    # Gauss-Hermite quadrature with weight e^{-x^2} is perfect for this.
    
    x_gq, w_gq = np.polynomial.hermite.hermgauss(n_points)
    
    x2 = x_gq * x_gq
    h0 = np.ones_like(x_gq)
    h4 = x2**2 - 6 * x2 + 3
    h8 = x2**4 - 28 * x2**3 + 210 * x2**2 - 420 * x2 + 105
    h12 = x2**6 - 84 * x2**5 + 1260 * x2**4 - 12600 * x2**3 + 55440 * x2**2 - 100800 * x2 + 10395
    
    # d from g(0)=0
    d = -(a * 1 + b * 3 + c * 105) / 10395
    
    P = a * h0 + b * h4 + c * h8 + d * h12
    integrand = P * P  # times e^{-x^2} which is the quadrature weight
    return np.sum(w_gq * integrand)


def _fourier_g_sq_integral(a, b, c, n_points=10000):
    """Compute ||F(g)||_2^2 = integral |G(k)|^2 dk over R.
    By Plancherel, this equals ||g||_2^2 if we use the unitary Fourier transform.
    But wait - we need to be careful about conventions.
    
    Actually, for the uncertainty principle bound, we need:
    A(f) = (integral x^2 |f(x)|^2 dx)^(1/2) / (integral |f(x)|^2 dx)^(1/2)
    A(F(f)) = (integral k^2 |F(k)|^2 dk)^(1/2) / (integral |F(k)|^2 dk)^(1/2)
    
    The product A(f)*A(F(f)) is what we're bounding.
    
    For our test function g(x) = P(x) e^{-x^2/2}, the Fourier transform is also
    a polynomial times Gaussian (since Fourier transform of x^n e^{-x^2/2} is
    related to Hermite polynomials).
    
    Actually, let's use the fact that for the Fourier transform with convention
    F(k) = (1/sqrt(2π)) ∫ f(x) e^{-ikx} dx, we have ||F||_2 = ||f||_2.
    
    But we need the second moments. Let me compute them directly.
    
    For g(x) = P(x) e^{-x^2/2}:
    ||g||_2^2 = ∫ P(x)^2 e^{-x^2} dx
    
    For the Fourier transform, since F is unitary, ||F(g)||_2^2 = ||g||_2^2.
    The second moment in frequency: ∫ k^2 |F(k)|^2 dk = ∫ |g'(x)|^2 dx
    (by integration by parts and the fact that F(ik f) = i F(f)' or similar)
    
    Actually: if F(f)(k) = (1/√(2π)) ∫ f(x) e^{-ikx} dx, then
    ∫ k^2 |F(f)(k)|^2 dk = ∫ |f'(x)|^2 dx
    
    So A(f) = sqrt(∫ x^2 |f|^2 dx / ∫ |f|^2 dx)
    A(F(f)) = sqrt(∫ |f'|^2 dx / ∫ |f|^2 dx)
    
    Product = sqrt(∫ x^2 |f|^2 dx * ∫ |f'|^2 dx / (∫ |f|^2 dx)^2)
    
    So I need to compute:
    1. I0 = ∫ |g|^2 dx = ∫ P^2 e^{-x^2} dx
    2. I2 = ∫ x^2 |g|^2 dx = ∫ x^2 P^2 e^{-x^2} dx
    3. J0 = ∫ |g'|^2 dx
    
    Let me compute g'(x) and then J0."""
    # For now, let's use numerical quadrature for all three quantities
    
    x_gq, w_gq = np.polynomial.hermite.hermgauss(n_points)
    x2 = x_gq * x_gq
    
    h0 = np.ones_like(x_gq)
    h4 = x2**2 - 6 * x2 + 3
    h8 = x2**4 - 28 * x2**3 + 210 * x2**2 - 420 * x2 + 105
    h12 = x2**6 - 84 * x2**5 + 1260 * x2**4 - 12600 * x2**3 + 55440 * x2**2 - 100800 * x2 + 10395
    
    d = -(a * 1 + b * 3 + c * 105) / 10395
    P = a * h0 + b * h4 + c * h8 + d * h12
    
    I0 = np.sum(w_gq * P * P)
    I2 = np.sum(w_gq * x2 * P * P)
    
    # g(x) = P(x) e^{-x^2/2}
    # g'(x) = P'(x) e^{-x^2/2} - x P(x) e^{-x^2/2} = (P'(x) - x P(x)) e^{-x^2/2}
    # |g'|^2 = (P' - xP)^2 e^{-x^2}
    
    # Compute P'(x):
    # P(x) = a*1 + b*(x^4-6x^2+3) + c*(x^8-28x^6+210x^4-420x^2+105) + d*(x^12-84x^10+1260x^8-12600x^6+55440x^4-100800x^2+10395)
    # P'(x) = b*(4x^3-12x) + c*(8x^7-168x^5+840x^3-840x) + d*(12x^11-840x^9+10080x^7-75600x^5+221760x^3-201600x)
    
    x3 = x_gq**3
    x5 = x_gq**5
    x7 = x_gq**7
    x9 = x_gq**9
    x11 = x_gq**11
    
    P_prime = (b * (4*x3 - 12*x_gq) + 
               c * (8*x7 - 168*x5 + 840*x3 - 840*x_gq) + 
               d * (12*x11 - 840*x9 + 10080*x7 - 75600*x5 + 221760*x3 - 201600*x_gq))
    
    g_prime_poly = P_prime - x_gq * P
    J0 = np.sum(w_gq * g_prime_poly * g_prime_poly)
    
    return I0, I2, J0


def _objective(params):
    a, b, c = params
    try:
        I0, I2, J0 = _fourier_g_sq_integral(a, b, c, n_points=200)
        if I0 <= 0 or J0 <= 0:
            return 1e10
        # A(f) = sqrt(I2/I0), A(F(f)) = sqrt(J0/I0)
        # Product = sqrt(I2 * J0 / I0^2) = sqrt(I2 * J0) / I0
        product = np.sqrt(I2 * J0) / I0
        return product
    except Exception:
        return 1e10


def construct():
    start_time = time.time()
    deadline = 180.0  # seconds
    
    # Seed value
    seed_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_result.copy()
    best_score = _objective(best_params)
    
    # Try multiple starting points with local optimization
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    start_points = [best_params.copy()]
    
    # Add perturbed versions of the seed
    for i in range(10):
        perturbation = rng.randn(3) * 0.1
        start_points.append(seed_result + perturbation)
    
    # Add some random points
    for i in range(10):
        start_points.append(rng.randn(3) * 0.5)
    
    # Add structured points
    for a_val in [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1]:
        for b_val in [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]:
            start_points.append(np.array([a_val, b_val, 0.01]))
    
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(_objective, sp, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Try to refine with different methods
    if time.time() < deadline - 20:
        try:
            result = minimize(_objective, best_params, method='Powell', 
                            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final check
    final_score = _objective(best_params)
    if final_score < best_score:
        best_score = final_score
    
    # Make sure we're better than seed
    seed_score = _objective(seed_result)
    if best_score > seed_score:
        best_params = seed_result.copy()
    
    return best_params
# EVOLVE-BLOCK-END
