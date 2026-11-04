"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite_moments(c0, c4, c8, c12):
    """
    Compute the uncertainty product A(f)*A(F) analytically for
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    
    Using the fact that for Hermite-Gaussian functions, we can compute
    <x^n> and <x^n> in Fourier space analytically.
    
    The normalized function: g/||g||
    A(f) = sqrt(<x^2> - <x>^2)  (standard deviation in x-space)
    A(F) = sqrt(<xi^2> - <xi>^2) (standard deviation in Fourier space)
    
    For the specific form with even Hermite polynomials, <x> = <xi> = 0.
    So A(f)*A(F) = sqrt(<x^2>_x * <x^2>_xi)
    
    By the uncertainty principle structure, for these functions:
    <x^2>_x and <x^2>_xi can be computed from the coefficients.
    """
    # Hermite polynomial values and their moments
    # H0(x) = 1, H4(x) = x^4 - 6x^2 + 3, H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 1260x^8 - 12600x^6 + 55440x^4 - 75600x^2 + 10395
    
    # We need: ||g||^2 = ∫ g^2 dx, ∫ x^2 g^2 dx, and the Fourier-space equivalent
    # For g = P(x) e^{-x^2/2}, ∫ g^2 dx = ∫ P(x)^2 e^{-x^2} dx
    # ∫ x^2 g^2 dx = ∫ x^2 P(x)^2 e^{-x^2} dx
    
    # Using: ∫ x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    # And P(x)^2 is a polynomial, so we can expand and integrate term by term.
    
    # Build P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Let's work with the polynomial coefficients (power basis)
    
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 84x^10 + 1260x^8 - 12600x^6 + 55440x^4 - 75600x^2 + 10395
    
    # P(x) in power basis: coeffs for x^0, x^2, x^4, ..., x^12
    # Only even powers appear
    
    # p[0] = coeff of x^0, p[1] = coeff of x^2, ..., p[6] = coeff of x^12
    p = np.zeros(7)
    p[0] = c0 * 1 + c4 * 3 + c8 * 105 + c12 * 10395
    p[1] = c4 * (-6) + c8 * (-420) + c12 * (-75600)
    p[2] = c4 * 1 + c8 * 210 + c12 * 55440
    p[3] = c8 * (-28) + c12 * (-12600)
    p[4] = c8 * 1 + c12 * 1260
    p[5] = c12 * (-84)
    p[6] = c12 * 1
    
    # P(x)^2: need to square this polynomial
    # (sum_{i=0}^{6} p[i] x^{2i})^2 = sum_{i,j} p[i]*p[j] x^{2(i+j)}
    # So Q(x) = P(x)^2 = sum_{k=0}^{12} q[k] x^{2k} where q[k] = sum_{i+j=k} p[i]*p[j]
    
    q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            if i + j <= 12:
                q[i + j] += p[i] * p[j]
    
    # ∫ x^{2n} e^{-x^2} dx = (2n)! / (2^n * n!) * sqrt(pi)
    # = (2n-1)!! * sqrt(pi) / 2^n
    
    def even_moment(n):
        """∫ x^{2n} e^{-x^2} dx = (2n)! / (2^n n!) * sqrt(pi)"""
        if n == 0:
            return np.sqrt(np.pi)
        result = 1.0
        for k in range(1, n + 1):
            result *= (2 * k - 1) / k
        return result * np.sqrt(np.pi)
    
    # ||g||^2 = ∫ P(x)^2 e^{-x^2} dx = sum_{k=0}^{12} q[k] * even_moment(k)
    norm_sq = 0.0
    for k in range(13):
        norm_sq += q[k] * even_moment(k)
    
    # ∫ x^2 P(x)^2 e^{-x^2} dx = sum_{k=0}^{12} q[k] * ∫ x^{2k+2} e^{-x^2} dx
    # = sum_{k=0}^{12} q[k] * even_moment(k+1)
    x2_moment = 0.0
    for k in range(13):
        x2_moment += q[k] * even_moment(k + 1)
    
    if norm_sq <= 0:
        return np.inf
    
    # <x^2> = x2_moment / norm_sq
    x2 = x2_moment / norm_sq
    
    # For the Fourier space: by the structure of the Hermite-Gaussian,
    # the Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(xi) e^{-xi^2/2} (up to constants)
    # So the Fourier-space variance has the same form with the same coefficients!
    # Actually, more carefully: F[H_n(x) e^{-x^2/2}](xi) = (-i)^n sqrt(2*pi) H_n(xi) e^{-xi^2/2}
    # So the Fourier transform is (up to a constant) the same form with the same coefficients.
    # Therefore <xi^2> = <x^2> and A(f)*A(F) = x2 (since both variances equal x2).
    
    # Wait, let me reconsider. The Fourier transform convention matters.
    # If F(f)(xi) = ∫ f(x) e^{-2πixi} dx, then for f(x) = H_n(x) e^{-x^2/2},
    # F(f)(xi) involves H_n with rescaled argument.
    
    # Actually, the standard result: if f(x) = e^{-x^2/2}, F(f)(xi) = sqrt(2π) e^{-π^2 xi^2}
    # This is NOT the same form. So I need to be more careful.
    
    # Let me use the convention where the uncertainty product is well-defined.
    # For f(x) = P(x) e^{-x^2/2}, the Fourier transform is:
    # F(f)(xi) = ∫ P(x) e^{-x^2/2} e^{-2πixi} dx
    # 
    # This is more complex. Let me use a different approach.
    # 
    # Actually, for the Hermite functions ψ_n(x) = H_n(x) e^{-x^2/2} / (2^n n! sqrt(pi))^{1/2},
    # they are eigenfunctions of the Fourier transform with eigenvalue (-i)^n.
    # So if g(x) = Σ c_n ψ_n(x), then F(g)(xi) = Σ c_n (-i)^n ψ_n(xi).
    # 
    # Since all our n are even (0,4,8,12), (-i)^n = 1 for all of them.
    # So F(g)(xi) = g(xi) (up to the normalization of the Fourier transform).
    # 
    # This means the x-space and xi-space variances are IDENTICAL.
    # Therefore A(f)*A(F) = <x^2> (the variance, since <x>=0).
    
    return x2


def _objective(coeffs):
    """Compute A(f)*A(F) for the given coefficients."""
    c0, c4, c8 = coeffs
    # a12 chosen so that g(0) = 0
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    return _hermite_moments(c0, c4, c8, c12)


def construct():
    """
    Find the three coefficients (a0, a4, a8) that minimize A(f)*A(F).
    a12 is determined by the constraint g(0) = 0.
    """
    start_time = time.time()
    deadline = 210.0  # evaluator kills at 210s
    
    # Initial guess from the seed
    best_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = _objective(best_coeffs)
    
    # Try multiple starting points with L-BFGS-B
    # The objective is a ratio of polynomials, so it should be smooth
    
    # Generate diverse starting points
    np.random.seed(42)
    start_points = [best_coeffs.copy()]
    
    # Add some random perturbations
    for _ in range(20):
        sp = best_coeffs + np.random.randn(3) * 0.5
        start_points.append(sp)
    
    # Add some structured starting points
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for c0 in [-1, 0, 1]:
            for c4 in [-1, 0, 1]:
                for c8 in [-1, 0, 1]:
                    sp = np.array([c0, c4, c8]) * scale
                    start_points.append(sp)
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 5.0  # 5s safety margin
    
    if remaining <= 0:
        return best_coeffs
    
    # Run local optimization from each starting point
    for sp in start_points:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        try:
            result = minimize(
                _objective,
                sp,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.success or result.fun < best_val:
                if result.fun < best_val:
                    best_val = result.fun
                    best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerances
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        try:
            result = minimize(
                _objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
