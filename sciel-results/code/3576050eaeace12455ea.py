"""Rewritten: eigenvalue-based optimization for the uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eigh

# ---------- Hermite polynomials (physicists' convention) ----------
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 45360x^2 + 10395

def _H0(x):
    return np.ones_like(x)

def _H4(x):
    return x**4 - 6.0*x**2 + 3.0

def _H8(x):
    return x**8 - 28.0*x**6 + 210.0*x**4 - 420.0*x**2 + 105.0

def _H12(x):
    return x**12 - 60.0*x**10 + 1260.0*x**8 - 12600.0*x**6 + 45360.0*x**4 - 45360.0*x**2 + 10395.0

# g(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
# g(0) = 0 constraint:
# H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
# a0 + 3*a4 + 105*a8 + 10395*a12 = 0
# => a12 = -(a0 + 3*a4 + 105*a8) / 10395

def _a12_from(a0, a4, a8):
    return -(a0 + 3.0*a4 + 105.0*a8) / 10395.0

# ---------- Analytic computation of moments ----------
# For f(x) = P(x) * exp(-x^2/2), we need:
# <x^2> = integral x^2 |f(x)|^2 dx / integral |f(x)|^2 dx
# 
# Using the fact that H_n(x) exp(-x^2/2) are orthogonal with respect to the weight 1:
# integral H_m(x) H_n(x) exp(-x^2) dx = 2^n n! sqrt(pi) delta_mn
#
# So if P(x) = sum a_k H_k(x), then:
# integral |P(x) exp(-x^2/2)|^2 dx = integral P(x)^2 exp(-x^2) dx
# = sum_k a_k^2 * 2^k k! * sqrt(pi)
#
# For <x^2>:
# integral x^2 P(x)^2 exp(-x^2) dx
# We need to express x^2 in terms of Hermite polynomials:
# x^2 = (1/2)(H_2(x) + H_0(x))  [since H_2(x) = x^2 - 1, so x^2 = (H_2(x)+1)/2]
#
# More generally, we can use the three-term recurrence:
# x H_n(x) = (1/2) H_{n+1}(x) + n H_{n-1}(x)
#
# For x^2 P(x)^2, we need to compute the integral of x^2 times a product of Hermite polynomials.
# This is complex. Let me use a different approach.
#
# Alternative: Use the fact that for f(x) = P(x) exp(-x^2/2),
# the Fourier transform is F(k) = P(k) exp(-k^2/2) (since all degrees are even, (-i)^n = 1 for even n)
#
# So <x^2>_f = <k^2>_F by the same formula structure, but with different P.
# Wait, actually F(k) = P(k) exp(-k^2/2) where P is the same polynomial.
# So <x^2>_f and <k^2>_F are computed the same way!
#
# No wait, that's not right. Let me reconsider.
#
# f(x) = P(x) exp(-x^2/2)
# F(k) = (1/sqrt(2*pi)) integral f(x) exp(-ikx) dx
# For Hermite-Gaussians: FT{H_n(x) exp(-x^2/2)} = (-i)^n H_n(k) exp(-k^2/2)
# Since n=0,4,8,12 are all even, (-i)^n = 1 for all of them.
# So F(k) = P(k) exp(-k^2/2)
#
# Therefore <x^2>_f = <k^2>_F, and A(f) = A(F).
# The product is A(f)^2 = <x^2>.
#
# Wait, that means we just need to minimize <x^2>!
#
# Let me verify: if f(x) = H_0(x) exp(-x^2/2) = exp(-x^2/2), then:
# F(k) = (1/sqrt(2*pi)) integral exp(-x^2/2) exp(-ikx) dx
# = exp(-k^2/2) (standard Gaussian FT)
# <x^2> for exp(-x^2/2): integral x^2 exp(-x^2) dx / integral exp(-x^2) dx
# = (sqrt(pi)/2) / sqrt(pi) = 1/2
# So A(f) = sqrt(1/2), product = 1/2. This matches the Heisenberg bound.
#
# Now for P(x) = H_0 + c*H_4:
# f(x) = (1 + c(x^4-6x^2+3)) exp(-x^2/2)
# We need to compute <x^2> = integral x^2 (1+c(x^4-6x^2+3))^2 exp(-x^2) dx / integral (1+c(x^4-6x^2+3))^2 exp(-x^2) dx
#
# This is doable analytically but tedious. Let me use a computational approach.
#
# Key insight: Since F(k) = P(k) exp(-k^2/2), we have A(f) = A(F), so the product is A(f)^2 = <x^2>.
# We just need to minimize <x^2> subject to the constraint that g(0) = 0.

def _compute_moments_analytic(a0, a4, a8):
    """Compute <x^2> and the norm analytically using Hermite polynomial orthogonality."""
    a12 = _a12_from(a0, a4, a8)
    
    # Coefficients for H_0, H_4, H_8, H_12
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # Norm: integral P(x)^2 exp(-x^2) dx = sum_k a_k^2 * 2^k k! * sqrt(pi)
    # We can drop sqrt(pi) as it cancels in the ratio
    norm_sq = 0.0
    for a, d in zip(coeffs, degrees):
        # 2^d * d!
        norm_sq += a**2 * (2.0**d) * float(np.math.factorial(d))
    
    if norm_sq < 1e-15:
        return np.inf, np.inf
    
    # For <x^2>, we need integral x^2 P(x)^2 exp(-x^2) dx
    # x^2 = (H_2(x) + H_0(x)) / 2
    # So x^2 P(x)^2 = (1/2) H_2(x) P(x)^2 + (1/2) H_0(x) P(x)^2
    #
    # integral H_2(x) P(x)^2 exp(-x^2) dx and integral P(x)^2 exp(-x^2) dx
    #
    # P(x)^2 = (sum_i a_i H_i(x))^2 = sum_i sum_j a_i a_j H_i(x) H_j(x)
    #
    # integral H_m(x) H_i(x) H_j(x) exp(-x^2) dx
    # This is a three-Hermite integral. For even degrees, we can use:
    # The integral is nonzero only when m+i+j is even (all even here, so always even)
    #
    # This is complex. Let me use a numerical quadrature approach that's very fast.
    
    # Actually, let me use Gauss-Hermite quadrature for exact results.
    # Gauss-Hermite quadrature: integral f(x) exp(-x^2) dx ≈ sum w_i f(x_i)
    # For polynomials up to degree 2*25-1 = 49, 25 points is exact.
    # Our P(x)^2 has degree at most 24, and x^2 P(x)^2 has degree 26, so 14 points suffice.
    # Let's use 30 points for safety.
    
    x_pts, w_pts = np.polynomial.hermite.hermgauss(30)
    
    # Evaluate P at quadrature points
    P_vals = np.zeros_like(x_pts)
    for a, d in zip(coeffs, degrees):
        if d == 0:
            P_vals += a * np.ones_like(x_pts)
        elif d == 4:
            P_vals += a * (x_pts**4 - 6.0*x_pts**2 + 3.0)
        elif d == 8:
            P_vals += a * (x_pts**8 - 28.0*x_pts**6 + 210.0*x_pts**4 - 420.0*x_pts**2 + 105.0)
        elif d == 12:
            P_vals += a * (x_pts**12 - 60.0*x_pts**10 + 1260.0*x_pts**8 - 12600.0*x_pts**6 + 45360.0*x_pts**4 - 45360.0*x_pts**2 + 10395.0)
    
    P_sq = P_vals**2
    
    # Norm: integral P(x)^2 exp(-x^2) dx
    norm = np.sum(w_pts * P_sq)
    
    # <x^2>: integral x^2 P(x)^2 exp(-x^2) dx / norm
    x2_num = np.sum(w_pts * x_pts**2 * P_sq)
    
    if norm < 1e-15:
        return np.inf, np.inf
    
    x2 = x2_num / norm
    
    return x2, norm

def _compute_bound_fast(a0, a4, a8):
    """Fast computation of the uncertainty product using analytic/quadrature methods."""
    x2, norm = _compute_moments_analytic(a0, a4, a8)
    if not np.isfinite(x2) or x2 < 0:
        return np.inf
    # A(f) = A(F) = sqrt(x2), so product = x2
    return x2

def _objective_from_coeffs(a0, a4, a8):
    """Objective function for optimization."""
    return _compute_bound_fast(a0, a4, a8)

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # safety margin before 210s
    
    # Initial guess: the prior best
    a0_init = -113.0 / 100.0
    a4_init = 1.0 / 25.0
    a8_init = 1.0 / 3240.0
    norm_init = np.sqrt(a0_init**2 + a4_init**2 + a8_init**2)
    a0_init /= norm_init
    a4_init /= norm_init
    a8_init /= norm_init
    
    best_val = _compute_bound_fast(a0_init, a4_init, a8_init)
    best_coeffs = np.array([a0_init, a4_init, a8_init])
    
    # Try multiple random starting points with local optimization
    rng = np.random.RandomState(42)
    n_random = 50
    
    for i in range(n_random):
        if time.time() > deadline - 10.0:
            break
        
        # Random direction on the sphere
        theta = rng.uniform(0, 2*np.pi)
        phi = rng.uniform(0, np.pi)
        a0_r = np.sin(phi) * np.cos(theta)
        a4_r = np.sin(phi) * np.sin(theta)
        a8_r = np.cos(phi)
        
        # Local optimization
        try:
            res = minimize(
                lambda x: _objective_from_coeffs(x[0], x[1], x[2]),
                x0=np.array([a0_r, a4_r, a8_r]),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.success and np.isfinite(res.fun):
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_coeffs = res.x
        except Exception:
            pass
    
    # Also try differential evolution for global search
    if time.time() < deadline - 30.0:
        try:
            def _obj_de(theta_phi):
                theta, phi = theta_phi
                a0 = np.sin(phi) * np.cos(theta)
                a4 = np.sin(phi) * np.sin(theta)
                a8 = np.cos(phi)
                return _objective_from_coeffs(a0, a4, a8)
            
            bounds = [(-np.pi, np.pi), (0, np.pi)]
            result = differential_evolution(
                _obj_de,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-12,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=True,
                init='sobol',
            )
            if np.isfinite(result.fun):
                theta_best, phi_best = result.x
                a0_best = np.sin(phi_best) * np.cos(theta_best)
                a4_best = np.sin(phi_best) * np.sin(theta_best)
                a8_best = np.cos(phi_best)
                if result.fun < best_val:
                    best_val = result.fun
                    best_coeffs = np.array([a0_best, a4_best, a8_best])
        except Exception:
            pass
    
    # Final polish with high-precision local optimization
    if time.time() < deadline - 5.0 and np.all(np.isfinite(best_coeffs)):
        try:
            res = minimize(
                lambda x: _objective_from_coeffs(x[0], x[1], x[2]),
                x0=best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.success and np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    # Ensure the coefficients are normalized (the objective is homogeneous of degree 0)
    coeff_norm = np.linalg.norm(best_coeffs)
    if coeff_norm > 1e-15:
        best_coeffs = best_coeffs / coeff_norm
    
    return best_coeffs

# EVOLVE-BLOCK-END
