"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite

# Use physicists' Hermite polynomials: H_n(x) where H_n is defined by
# H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H_0 = 1
# H_1 = 2x
# H_2 = 4x^2 - 2
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H_12 = 4096x^12 - 98304x^10 + 884736x^8 - 3932160x^6 + 7654080x^4 - 5242880x^2 + 462270

# Precompute H_n(0) for the constraint g(0) = 0
# H_0(0) = 1
# H_4(0) = 12
# H_8(0) = 1680
# H_12(0) = 462270

# The test function is g(x) = e^{-x^2/2} * P(x)
# where P(x) = a0*H_0(x) + a4*H_4(x) + a8*H_8(x) + a12*H_12(x)
# Constraint: g(0) = P(0) = 0
# So: a0*H_0(0) + a4*H_4(0) + a8*H_8(0) + a12*H_12(0) = 0
# a12 = -(a0*1 + a4*12 + a8*1680) / 462270

# The uncertainty bound C4 involves A(f) * A(Fourier(f))
# For a Fourier-invariant setup with Hermite-Gaussian test functions,
# we need to compute the relevant integrals.

# A(f) typically involves ||f||_2 or related norms, and the uncertainty
# product involves moments. Let me think about what the "upper bound on C4"
# means in this context.

# The task says "reduce the upper bound on A(f)*A(Fourier(f))" and
# "score is the upper bound on C4"
# The seed returns coefficients and gets score 1.0.
# The evaluator measures "raw value (lower is better)".

# Let me think about this more carefully. The uncertainty principle says:
# A(f) * A(F) >= C, where A(f) is the uncertainty in x-space and A(F) in
# frequency space.

# For a function g(x) = e^{-x^2/2} P(x) where P is a polynomial:
# - ||g||_2^2 = integral |g|^2 dx
# - The x-variance: sigma_x^2 = (integral x^2 |g|^2 dx) / ||g||_2^2
# - The Fourier transform of e^{-x^2/2} P(x) can be computed analytically
#   since Fourier transform of Hermite functions is known.

# Actually, let me reconsider. The Hermite functions are psi_n(x) = H_n(x) e^{-x^2/2}
# (up to normalization). Their Fourier transforms are (up to sign) psi_n itself.
# So if g = a0*psi_0 + a4*psi_4 + a8*psi_8 + a12*psi_12, then
# F(g) is a linear combination of psi_0, psi_4, psi_8, psi_12 with the same
# coefficients (possibly with signs).

# So A(f) * A(F) = sigma_x * sigma_xi, and by Fourier invariance of the
# Hermite-Gaussian system, sigma_x = sigma_xi (the variance in x and in
# frequency space are the same for such combinations).

# So we need to minimize sigma_x^2 = (integral x^2 |g|^2 dx) / (integral |g|^2 dx)

# But wait - we also need g(0) = 0, which is the constraint.

# Let me compute the necessary integrals.
# g(x) = e^{-x^2/2} * P(x)
# |g|^2 = e^{-x^2} * P(x)^2

# integral |g|^2 dx = integral e^{-x^2} P(x)^2 dx
# integral x^2 |g|^2 dx = integral x^2 e^{-x^2} P(x)^2 dx

# These can be computed numerically or analytically using Hermite polynomial
# orthogonality.

# Actually, let's use the fact that the Hermite functions are orthogonal:
# integral psi_m(x) psi_n(x) dx = sqrt(pi) * 2^n * n! * delta_{mn}
# (for physicists' Hermite functions, with the standard normalization)

# Wait, let me be more careful. The physicists' Hermite functions are:
# psi_n(x) = (1/sqrt(2^n n! sqrt(pi))) H_n(x) e^{-x^2/2}
# These are orthonormal: integral psi_m psi_n dx = delta_{mn}

# So if g = sum c_n psi_n, then ||g||_2^2 = sum |c_n|^2
# And sigma_x^2 = sum c_n^2 <x^2>_{n} / sum c_n^2

# where <x^2>_n = integral x^2 psi_n^2 dx

# For Hermite functions, <x^2>_n = n + 1/2

# Wait, that's for the quantum harmonic oscillator where H_n are physicists'
# Hermite polynomials. Let me verify:
# psi_n(x) = (1/sqrt(2^n n! sqrt(pi))) H_n(x) e^{-x^2/2}
# <x^2>_n = n + 1/2

# So sigma_x^2 = sum c_n^2 (n + 1/2) / sum c_n^2

# But we need to express this in terms of the coefficients a0, a4, a8, a12
# where g = a0*H_0*e^{-x^2/2} + a4*H_4*e^{-x^2/2} + ...

# Let c_n = a_n / sqrt(2^n n! sqrt(pi))

# So:
# c_0 = a_0 / sqrt(sqrt(pi)) = a_0 / pi^{1/4}
# c_4 = a_4 / sqrt(2^4 * 4! * sqrt(pi)) = a_4 / sqrt(16 * 24 * sqrt(pi))
# c_8 = a_8 / sqrt(2^8 * 8! * sqrt(pi)) = a_8 / sqrt(256 * 40320 * sqrt(pi))
# c_12 = a_12 / sqrt(2^12 * 12! * sqrt(pi)) = a_12 / sqrt(4096 * 479001600 * sqrt(pi))

# And sigma_x^2 = (c_0^2 * 0.5 + c_4^2 * 4.5 + c_8^2 * 8.5 + c_12^2 * 12.5) / (c_0^2 + c_4^2 + c_8^2 + c_12^2)

# This is a smooth function of a0, a4, a8 (with a12 determined by the constraint).

# Let me code this up.

def _compute_sigma_x_squared(a0, a4, a8):
    """Compute sigma_x^2 for the given coefficients."""
    # Compute a12 from the constraint g(0) = 0
    # a0*H_0(0) + a4*H_4(0) + a8*H_8(0) + a12*H_12(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 462270.0
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Compute normalization factors
    # c_n = a_n / sqrt(2^n * n! * sqrt(pi))
    sqrt_pi = np.sqrt(np.pi)
    
    norm_0 = np.sqrt(2**0 * 1 * sqrt_pi)  # 0! = 1
    norm_4 = np.sqrt(2**4 * 24 * sqrt_pi)  # 4! = 24
    norm_8 = np.sqrt(2**8 * 40320 * sqrt_pi)  # 8! = 40320
    norm_12 = np.sqrt(2**12 * 479001600 * sqrt_pi)  # 12! = 479001600
    
    c0 = a0 / norm_0
    c4 = a4 / norm_4
    c8 = a8 / norm_8
    c12 = a12 / norm_12
    
    # sigma_x^2 = sum c_n^2 * (n + 0.5) / sum c_n^2
    num = c0**2 * 0.5 + c4**2 * 4.5 + c8**2 * 8.5 + c12**2 * 12.5
    den = c0**2 + c4**2 + c8**2 + c12**2
    
    if den < 1e-30:
        return np.inf
    
    return num / den

def _objective(params):
    """Objective function to minimize: sigma_x^2."""
    a0, a4, a8 = params
    return _compute_sigma_x_squared(a0, a4, a8)

def construct():
    """Find the three coefficients that minimize the uncertainty bound."""
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s limit
    
    best_value = np.inf
    best_params = np.array([-113/100, 1/25, 1/3240])
    
    # Evaluate the seed
    seed_val = _compute_sigma_x_squared(best_params[0], best_params[1], best_params[2])
    if seed_val < best_value:
        best_value = seed_val
    
    # Try a grid search first to get a good starting point
    # The coefficients should be of moderate size
    # Let's try a coarse grid
    a0_vals = np.linspace(-5, 5, 30)
    a4_vals = np.linspace(-5, 5, 30)
    a8_vals = np.linspace(-5, 5, 30)
    
    for a0 in a0_vals:
        if time.time() > deadline:
            break
        for a4 in a4_vals:
            if time.time() > deadline:
                break
            for a8 in a8_vals:
                val = _compute_sigma_x_squared(a0, a4, a8)
                if val < best_value:
                    best_value = val
                    best_params = np.array([a0, a4, a8])
    
    # Refine with local optimization from the best point found
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-15}
            )
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline:
                break
            try:
                init = rng.uniform(-10, 10, size=3)
                result = minimize(
                    _objective,
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-15}
                )
                if result.fun < best_value:
                    best_value = result.fun
                    best_params = result.x
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-17}
            )
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
