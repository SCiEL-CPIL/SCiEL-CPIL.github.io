"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# We need to understand the uncertainty inequality context.
# The task mentions:
# - Fourier-invariant Hermite-polynomial times Gaussian test function
# - Three free coefficients multiply H0, H4, H8
# - H12 coefficient derived to impose P(0)=0 (i.e., g(0) = 0)
# - Score is the upper bound on C4 = A(f)*A(Fourier(f))
# - Lower is better
# - Seed program returns [-113/100, 1/25, 1/3240] with score 1.0

# The Hermite polynomials (physicists' convention) are:
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 50400x^4 - 75600x^2 + 10395

# The test function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
# where a12 is chosen so that g(0) = 0.

# At x=0:
# H0(0) = 1
# H4(0) = 3
# H8(0) = 105
# H12(0) = 10395

# So g(0) = (a0 + 3*a4 + 105*a8 + 10395*a12) * exp(0) = 0
# => a12 = -(a0 + 3*a4 + 105*a8) / 10395

# The uncertainty product involves:
# A(f) = (∫ x^2 |f(x)|^2 dx) / (∫ |f(x)|^2 dx)  -- variance-like quantity
# A(F) = (∫ k^2 |F(k)|^2 dk) / (∫ |F(k)|^2 dk)

# For g(x) = P(x) * exp(-x^2/2), where P is a polynomial:
# |g(x)|^2 = P(x)^2 * exp(-x^2)

# The Fourier transform of x^n * exp(-x^2/2) can be computed.
# Actually, let's think about this more carefully.

# For a Gaussian times polynomial, we can compute the integrals analytically.
# Let's use the fact that ∫ x^{2m} exp(-x^2) dx = (2m-1)!! * sqrt(pi) / 2^m

# Actually, let me set up the problem more carefully.

# g(x) = P(x) * exp(-x^2/2)
# |g(x)|^2 = P(x)^2 * exp(-x^2)

# Let P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12

# We need:
# I0 = ∫ P(x)^2 * exp(-x^2) dx  (normalization)
# I2 = ∫ x^2 * P(x)^2 * exp(-x^2) dx  (for position variance)
# I4 = ∫ x^4 * P(x)^2 * exp(-x^2) dx  (might be needed)

# For the Fourier side, we need the Fourier transform of g(x).
# The Fourier transform of exp(-x^2/2) is sqrt(2*pi) * exp(-k^2/2) (depending on convention)
# Let's use the convention: F(k) = ∫ g(x) exp(-i*k*x) dx

# The Fourier transform of x^n * exp(-x^2/2) is related to derivatives of the Gaussian transform.
# Actually, it's easier to note that Hermite polynomials have nice Fourier transform properties.

# For physicists' Hermite polynomials: F[H_n(x) * exp(-x^2/2)] = i^n * sqrt(2*pi) * H_n(k) * exp(-k^2/2)
# Wait, let me verify. The standard result is:
# ∫ H_n(x) exp(-x^2/2) exp(-i*k*x) dx = i^n * sqrt(2*pi) * H_n(k) * exp(-k^2/2)

# So if g(x) = P(x) * exp(-x^2/2) where P(x) = sum c_n H_n(x),
# then G(k) = sum c_n * i^n * sqrt(2*pi) * H_n(k) * exp(-k^2/2)

# So |G(k)|^2 = 2*pi * |sum c_n * i^n * H_n(k)|^2 * exp(-k^2)

# Since H_n is real, |i^n * H_n(k)|^2 = H_n(k)^2, but cross terms matter.
# Actually |sum c_n i^n H_n(k)|^2 = (sum c_n i^n H_n(k)) * (sum c_m (-i)^m H_m(k))
# = sum_{n,m} c_n c_m * i^{n-m} * H_n(k) H_m(k)

# The integral ∫ H_n(k) H_m(k) exp(-k^2) dk = sqrt(pi) * 2^n * n! * delta_{nm}

# So ∫ |G(k)|^2 dk = 2*pi * sqrt(pi) * sum_n c_n^2 * 2^n * n!
# And ∫ k^2 |G(k)|^2 dk = 2*pi * sum_{n,m} c_n c_m i^{n-m} ∫ k^2 H_n(k) H_m(k) exp(-k^2) dk

# This is getting complex. Let me use a numerical approach for the integrals.

# Actually, for the uncertainty product, we can compute:
# A(f) = I2/I0 - (I1/I0)^2  where I1 = ∫ x*P(x)^2*exp(-x^2) dx
# But since P(x)^2 * exp(-x^2) is even (P has only even powers if all H_n are even), I1 = 0.
# So A(f) = I2/I0

# Similarly for the Fourier side.

# Let me just compute numerically for given coefficients.

def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x)"""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2*x
    elif n == 2:
        return 4*x**2 - 2
    elif n == 3:
        return 8*x**3 - 12*x
    elif n == 4:
        return 16*x**4 - 48*x**2 + 12
    elif n == 5:
        return 32*x**5 - 160*x**3 + 120*x
    elif n == 6:
        return 64*x**6 - 480*x**4 + 720*x**2 - 120
    elif n == 7:
        return 128*x**7 - 1344*x**5 + 3360*x**3 - 1680*x
    elif n == 8:
        return 256*x**8 - 3584*x**6 + 13440*x**4 - 13440*x**2 + 1680
    elif n == 9:
        return 512*x**9 - 9216*x**7 + 48384*x**5 - 80640*x**3 + 30240*x
    elif n == 10:
        return 1024*x**10 - 23040*x**8 + 161280*x**6 - 403200*x**4 + 302400*x**2 - 30240
    elif n == 11:
        return 2048*x**11 - 55296*x**9 + 516096*x**7 - 1848000*x**5 + 2540160*x**3 - 665280*x
    elif n == 12:
        return 4096*x**12 - 129024*x**10 + 1548288*x**8 - 7983360*x**6 + 17418240*x**4 - 13305600*x**2 + 1330560
    else:
        raise ValueError(f"n={n} not implemented")


def _build_g(coeffs):
    """
    Build the polynomial P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    where a12 is determined by the constraint P(0) = 0.
    coeffs = [a0, a4, a8]
    """
    a0, a4, a8 = coeffs
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
    return a0, a4, a8, a12


def _compute_bound(coeffs):
    """
    Compute the uncertainty product A(f)*A(F) for the given coefficients.
    """
    a0, a4, a8, a12 = _build_g(coeffs)
    
    # Use numerical integration
    # g(x) = P(x) * exp(-x^2/2)
    # |g(x)|^2 = P(x)^2 * exp(-x^2)
    
    x = np.linspace(-10, 10, 10000)
    dx = x[1] - x[0]
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    
    g_sq = P**2 * np.exp(-x**2)
    
    I0 = np.trapz(g_sq, x)
    I2 = np.trapz(x**2 * g_sq, x)
    
    if I0 <= 0:
        return np.inf
    
    A_f = I2 / I0
    
    # Fourier transform: G(k) = sum c_n * i^n * sqrt(2*pi) * H_n(k) * exp(-k^2/2)
    # |G(k)|^2 = 2*pi * |sum c_n * i^n * H_n(k)|^2 * exp(-k^2)
    
    # For the Fourier side, we need to be careful with the complex coefficients.
    # c0 = a0, c4 = a4, c8 = a8, c12 = a12
    # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = 1
    # So all the i^n factors are 1 for n = 0, 4, 8, 12!
    
    # Therefore G(k) = sqrt(2*pi) * (a0*H0(k) + a4*H4(k) + a8*H8(k) + a12*H12(k)) * exp(-k^2/2)
    # |G(k)|^2 = 2*pi * P(k)^2 * exp(-k^2)
    
    # So the Fourier side integral is the same form as the position side!
    k = x  # symmetric
    G_sq = 2 * np.pi * P**2 * np.exp(-k**2)
    
    J0 = np.trapz(G_sq, k)
    J2 = np.trapz(k**2 * G_sq, k)
    
    if J0 <= 0:
        return np.inf
    
    A_F = J2 / J0
    
    return A_f * A_F


def construct():
    """
    Return three coefficients [a0, a4, a8] that minimize the uncertainty product bound.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # safety margin
    
    # Start from the known good point
    initial = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_bound(initial)
    best_coeffs = initial.copy()
    
    print(f"Initial bound: {best_val}")
    
    # Try local optimization from the initial point
    try:
        result = minimize(_compute_bound, initial, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if time.time() < deadline:
            val = _compute_bound(result.x)
            if val < best_val:
                best_val = val
                best_coeffs = result.x.copy()
                print(f"After Nelder-Mead: {best_val}")
    except Exception as e:
        print(f"Nelder-Mead failed: {e}")
    
    # Try a grid search in a neighborhood
    if time.time() < deadline - 60:
        # Explore around the initial point
        for scale in [0.5, 1.0, 2.0, 5.0]:
            if time.time() > deadline - 30:
                break
            # Try perturbations
            n_pert = 200
            for i in range(n_pert):
                if time.time() > deadline - 30:
                    break
                pert = np.random.randn(3) * scale
                candidate = initial + pert
                val = _compute_bound(candidate)
                if val < best_val:
                    best_val = val
                    best_coeffs = candidate.copy()
    
    # Final polish
    if time.time() < deadline - 10:
        try:
            result = minimize(_compute_bound, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            val = _compute_bound(result.x)
            if val < best_val:
                best_val = val
                best_coeffs = result.x.copy()
                print(f"After final polish: {best_val}")
        except Exception as e:
            print(f"Final polish failed: {e}")
    
    print(f"Best bound: {best_val}")
    print(f"Best coeffs: {best_coeffs}")
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
