"""Improved program: multi-start Nelder-Mead + high-precision polish on the 2-DOF root landscape."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction
import math

# Precompute Hermite polynomial values at 0 (exact integers)
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H12(x) = 4096x^12 - 73728x^10 + 516096x^8 - 1548288x^6 + 2123328x^4 - 1290240x^2 + 2032128
H0_0 = 1
H4_0 = 12
H8_0 = 1680
H12_0 = 2032128

def _compute_value(c0, c1, c2):
    """Compute the objective value: r^2 / (2*pi) where r is the largest sign-changing real root of q."""
    # c3 is determined by g(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Build g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12(x) = 4096x^12 - 73728x^10 + 516096x^8 - 1548288x^6 + 2123328x^4 - 1290240x^2 + 2032128
    
    # g(x) = c0*1 + c1*(16x^4 - 48x^2 + 12) + c2*(256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680) 
    #        + c3*(4096x^12 - 73728x^10 + 516096x^8 - 1548288x^6 + 2123328x^4 - 1290240x^2 + 2032128)
    
    # Collect coefficients for powers x^0, x^2, x^4, x^6, x^8, x^10, x^12
    # Only even powers appear
    
    # x^0: c0 + c1*12 + c2*1680 + c3*2032128
    # x^2: c1*(-48) + c2*(-13440) + c3*(-1290240)
    # x^4: c1*16 + c2*11520 + c3*2123328
    # x^6: c2*(-3072) + c3*(-1548288)
    # x^8: c2*256 + c3*516096
    # x^10: c3*(-73728)
    # x^12: c3*4096
    
    a0 = c0 + c1 * 12.0 + c2 * 1680.0 + c3 * 2032128.0
    a2 = c1 * (-48.0) + c2 * (-13440.0) + c3 * (-1290240.0)
    a4 = c1 * 16.0 + c2 * 11520.0 + c3 * 2123328.0
    a6 = c2 * (-3072.0) + c3 * (-1548288.0)
    a8 = c2 * 256.0 + c3 * 516096.0
    a10 = c3 * (-73728.0)
    a12 = c3 * 4096.0
    
    # Check if g tends to negative limit at infinity: leading coeff is a12 = c3*4096
    # If a12 < 0, then g(x) -> -infinity as x -> infinity, so we replace g by -g
    if a12 < 0:
        a0 = -a0
        a2 = -a2
        a4 = -a4
        a6 = -a6
        a8 = -a8
        a10 = -a10
        a12 = -a12
    
    # q(x) = g(x) / x^2, which must divide exactly
    # q(x) = a0/x^2 + a2 + a4*x^2 + a6*x^4 + a8*x^6 + a10*x^8 + a12*x^10
    # Wait, that's not right. g(x) is a polynomial in x. For g(x)/x^2 to divide exactly,
    # g(0) must be 0 (which it is by construction of c3) and the x^1 term must be 0 (it is, only even powers).
    # So g(x) = a0 + a2*x^2 + a4*x^4 + ... + a12*x^12
    # Since g(0) = a0 = 0 (by construction), we have:
    # Actually let's verify: a0 should be 0 because c3 was chosen so g(0)=0.
    # g(0) = c0*1 + c1*12 + c2*1680 + c3*2032128 = 0
    # So a0 = 0.
    
    # q(x) = g(x)/x^2 = a2 + a4*x^2 + a6*x^4 + a8*x^6 + a10*x^8 + a12*x^10
    # Let z = x^2, then Q(z) = a2 + a4*z + a6*z^2 + a8*z^3 + a10*z^4 + a12*z^5
    # We need the largest real root r of q(x) at which q changes sign.
    # Since q(x) = Q(x^2), the roots in x are +/- sqrt(z_i) for each positive root z_i of Q.
    # q changes sign at x = r iff Q changes sign at z = r^2.
    # Since Q is a polynomial in z, it changes sign at a simple real root.
    # We want the largest real root r of q, i.e., the largest positive z root of Q, then r = sqrt(z).
    
    # Build Q(z) coefficients: [a2, a4, a6, a8, a10, a12] for z^0, z^1, ..., z^5
    # Find all real roots of Q(z) = 0
    # Use numpy roots
    coeffs = np.array([a12, a10, a8, a6, a4, a2])  # highest to lowest
    if np.abs(coeffs[0]) < 1e-30:
        # Leading coefficient is zero, reduce degree
        # Find first non-zero
        idx = 0
        while idx < len(coeffs) - 1 and np.abs(coeffs[idx]) < 1e-30:
            idx += 1
        coeffs = coeffs[idx:]
    
    roots_z = np.roots(coeffs)
    
    # Filter real positive roots
    real_pos_roots = []
    for rz in roots_z:
        if np.isreal(rz):
            rz_real = np.real(rz)
            if rz_real > 1e-15:  # positive
                real_pos_roots.append(rz_real)
    
    if not real_pos_roots:
        # No valid positive real root; return a large value
        return 1e10
    
    # For each positive root z_i, check if Q changes sign at that root
    # Q changes sign at a simple root. For a polynomial, sign change occurs at odd-multiplicity roots.
    # Numerically, we check by evaluating Q at z_i - eps and z_i + eps.
    best_r = 0.0
    for z_i in real_pos_roots:
        # Check sign change
        eps = 1e-8 * max(1.0, np.abs(z_i))
        q_minus = a2 + a4*(z_i-eps) + a6*(z_i-eps)**2 + a8*(z_i-eps)**3 + a10*(z_i-eps)**4 + a12*(z_i-eps)**5
        q_plus = a2 + a4*(z_i+eps) + a6*(z_i+eps)**2 + a8*(z_i+eps)**3 + a10*(z_i+eps)**4 + a12*(z_i+eps)**5
        if q_minus * q_plus < 0:  # sign change
            r_i = np.sqrt(z_i)
            if r_i > best_r:
                best_r = r_i
    
    if best_r == 0.0:
        # No sign-changing root found; use the largest positive root as fallback
        best_r = np.sqrt(max(real_pos_roots))
    
    value = best_r**2 / (2.0 * np.pi)
    return value


def _objective(params):
    """Objective function for optimization."""
    c0, c1, c2 = params
    try:
        return _compute_value(c0, c1, c2)
    except:
        return 1e10


def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180s safety margin
    
    # Seed point
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_params = seed.copy()
    best_val = _compute_value(*seed)
    
    # Multi-start Nelder-Mead
    # Generate diverse starting points
    np.random.seed(42)
    
    # Create a set of initial points around the seed
    starts = [seed.copy()]
    
    # Add perturbations
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(5):
            perturbation = np.random.randn(3) * scale
            starts.append(seed + perturbation)
    
    # Also try some structured points
    for c0 in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        for c1 in [-0.5, 0.0, 0.5, 1.0]:
            starts.append(np.array([c0, c1, 0.0]))
    
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except:
            pass
    
    # Polish with tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-14,
                    'fatol': 1e-15,
                    'adaptive': True
                }
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except:
            pass
    
    # Final safety check
    if time.time() > deadline - 5.0:
        return np.array(best_params)
    
    # One more polish attempt if time permits
    if time.time() < deadline - 5.0:
        try:
            # Try BFGS for a local polish
            result = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
