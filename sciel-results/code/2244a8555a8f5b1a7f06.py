"""Rewritten program: uses analytic L2, root-finding for L1/Linf, and global optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import sys
import math

# ---------------------------------------------------------------------------
# Mathematical setup
# ---------------------------------------------------------------------------

def _hermite_coeffs(n):
    """
    Return the coefficients of the physicists' Hermite polynomial H_n(x)
    in descending order of powers (for np.roots).
    H_0 = 1
    H_1 = 2x
    H_2 = 4x^2 - 2
    H_3 = 8x^3 - 12x
    ...
    Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    """
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    
    H_nm1 = np.array([1.0])
    H_n = np.array([2.0, 0.0])
    
    for k in range(1, n):
        # H_{k+1} = 2x H_k - 2k H_{k-1}
        # Pad to same length
        len_n = len(H_n)
        len_nm1 = len(H_nm1)
        max_len = max(len_n, len_nm1) + 1
        
        H_new = np.zeros(max_len)
        
        # 2x H_k
        # H_k is [c_0, c_1, ...] where c_0 is coeff of x^{len_n-1}
        # Shifting left by 1 and multiplying by 2
        H_new[:-1] += 2.0 * H_n
        # -2k H_{k-1}
        # Align H_nm1 to the right
        offset = max_len - len_nm1
        H_new[offset:] -= 2.0 * k * H_nm1
        
        H_nm1 = H_n
        H_n = H_new
        
    return H_n

# Precompute coefficient vectors for H0, H4, H8, H12
# These are in descending power order for np.roots
_H0_c = _hermite_coeffs(0)
_H4_c = _hermite_coeffs(4)
_H8_c = _hermite_coeffs(8)
_H12_c = _hermite_coeffs(12)

def _poly_coeffs(a0, a4, a8, a12):
    """Combine Hermite polynomials into a single coefficient vector."""
    # Ensure all have same length
    max_len = max(len(_H0_c), len(_H4_c), len(_H8_c), len(_H12_c))
    p = np.zeros(max_len)
    
    def add_poly(coeffs, weight):
        offset = max_len - len(coeffs)
        p[offset:] += weight * coeffs

    add_poly(_H0_c, a0)
    add_poly(_H4_c, a4)
    add_poly(_H8_c, a8)
    add_poly(_H12_c, a12)
    
    return p

def _eval_poly(p, x):
    """Evaluate polynomial with coefficients p (descending order) at x."""
    return np.polynomial.polynomial.polyval(x, p[::-1])

def _objective(a):
    """
    Compute the upper bound C4 = A(f) * A(F(f)) = (||g||_1 / ||g||_2) * (||g||_inf / ||g||_2)
    = ||g||_1 * ||g||_inf / ||g||_2^2
    """
    a0, a4, a8 = a
    # Calculate a12 such that g(0) = 0
    # g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * 1
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    H0_0 = 1.0
    H4_0 = -3.0
    H8_0 = 105.0
    H12_0 = -10395.0
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # 1. Analytic L2 norm squared
    # ||g||_2^2 = integral (P(x) e^{-x^2/2})^2 dx = integral P(x)^2 e^{-x^2} dx
    # Orthogonality: integral H_n(x) H_m(x) e^{-x^2} dx = sqrt(pi) 2^n n! delta_nm
    # P(x) = sum c_k H_k(x)
    # ||g||_2^2 = sqrt(pi) * sum c_k^2 2^k k!
    
    norm2_sq = 0.0
    norm2_sq += a0**2 * (2**0 * math.factorial(0))
    norm2_sq += a4**2 * (2**4 * math.factorial(4))
    norm2_sq += a8**2 * (2**8 * math.factorial(8))
    norm2_sq += a12**2 * (2**12 * math.factorial(12))
    norm2_sq *= math.sqrt(math.pi)
    
    if norm2_sq < 1e-12:
        return 1e100
        
    # 2. Numerical L1 norm
    # Find roots of P(x) to split integral
    p_coeffs = _poly_coeffs(a0, a4, a8, a12)
    
    # Get real roots
    roots = np.roots(p_coeffs)
    real_roots = np.sort(roots[np.isreal(roots)].real)
    
    # Define integrand
    def integrand(x):
        val = _eval_poly(p_coeffs, x) * np.exp(-0.5 * x * x)
        return val
        
    # Integrate |P(x) e^{-x^2/2}|
    # We integrate on intervals between roots
    # Since e^{-x^2/2} is positive, the sign is determined by P(x)
    # We can just integrate the absolute value using quad on the whole line?
    # Quad might struggle with oscillations. Splitting is better.
    
    # Add boundaries
    # The function decays rapidly. Limit to [-15, 15] is safe.
    bounds = [-15.0] + list(real_roots) + [15.0]
    bounds = [b for b in bounds if -15.0 <= b <= 15.0]
    bounds.sort()
    
    # Ensure unique bounds
    unique_bounds = []
    for b in bounds:
        if not unique_bounds or abs(b - unique_bounds[-1]) > 1e-9:
            unique_bounds.append(b)
            
    l1_val = 0.0
    for i in range(len(unique_bounds) - 1):
        a_int = unique_bounds[i]
        b_int = unique_bounds[i + 1]
        # Integrate |integrand|
        # To avoid absolute value inside quad (which can be slow), we can determine the sign
        # at the midpoint
        mid = 0.5 * (a_int + b_int)
        sign = 1.0 if _eval_poly(p_coeffs, mid) > 0 else -1.0
        res, err = quad(integrand, a_int, b_int, limit=100)
        l1_val += abs(res)
        
    # 3. Numerical Linf norm
    # Find critical points of g(x) = P(x) e^{-x^2/2}
    # g'(x) = (P'(x) - x P(x)) e^{-x^2/2}
    # Let Q(x) = P'(x) - x P(x)
    # P(x) = sum c_i x^i (ascending)
    # p_coeffs is descending. Let's convert to ascending for differentiation.
    p_asc = p_coeffs[::-1]
    
    # Derivative P'(x)
    # If P(x) = c_0 + c_1 x + ..., P'(x) = c_1 + 2 c_2 x + ...
    dp_coeffs_asc = np.array([i * p_asc[i] for i in range(1, len(p_asc))]) if len(p_asc) > 1 else np.array([0.0])
    
    # x P(x)
    # If P(x) = c_0 + c_1 x + ..., x P(x) = c_0 x + c_1 x^2 + ...
    # Coeffs: [0, c_0, c_1, ...]
    xp_coeffs_asc = np.zeros(len(p_asc) + 1)
    xp_coeffs_asc[1:] = p_asc
    
    # Q(x) = P'(x) - x P(x)
    # Align lengths
    max_len = max(len(dp_coeffs_asc), len(xp_coeffs_asc))
    dp_pad = np.zeros(max_len)
    xp_pad = np.zeros(max_len)
    dp_pad[:len(dp_coeffs_asc)] = dp_coeffs_asc
    xp_pad[:len(xp_coeffs_asc)] = xp_coeffs_asc
    
    q_coeffs_asc = dp_pad - xp_pad
    
    # Find roots of Q(x)
    # Convert to descending for np.roots
    q_coeffs_desc = q_coeffs_asc[::-1]
    # Remove leading zeros
    while len(q_coeffs_desc) > 1 and abs(q_coeffs_desc[0]) < 1e-10:
        q_coeffs_desc = q_coeffs_desc[1:]
        
    q_roots = np.roots(q_coeffs_desc)
    real_q_roots = np.sort(q_roots[np.isreal(q_roots)].real)
    
    # Evaluate |g(x)| at these points and also at 0 (just in case) and boundaries
    candidates = list(real_q_roots) + [0.0]
    # Also check a few points in the tails just in case roots are missed
    candidates += np.linspace(-10, 10, 200).tolist()
    
    max_val = 0.0
    for x in candidates:
        val = abs(_eval_poly(p_coeffs, x) * np.exp(-0.5 * x * x))
        if val > max_val:
            max_val = val
            
    if max_val < 1e-100:
        return 1e100
        
    # Calculate bound
    bound = (l1_val * max_val) / norm2_sq
    return bound

def construct():
    """
    Return the three free Hermite coefficients (degrees 0, 4, 8).
    """
    # Bounds for the coefficients
    # Based on the reference values: a0 ~ -1.13, a4 ~ 0.04, a8 ~ 0.0003
    bounds = [
        (-5.0, 5.0),
        (-0.5, 0.5),
        (-0.01, 0.01)
    ]
    
    best_x = None
    best_val = float('inf')
    
    # First, try differential evolution for global exploration
    try:
        res = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-8,
            popsize=10,
            polish=False
        )
        best_x = res.x
        best_val = res.fun
    except Exception as e:
        print(f"Optimization failed: {e}", file=sys.stderr)
        best_x = np.array([-1.13, 0.04, 0.0003])
        best_val = _objective(best_x)
        
    # Polish with local optimization
    # Use multiple local starts around the best found
    local_starts = [best_x]
    # Add some perturbations
    np.random.seed(42)
    for _ in range(5):
        perturbation = np.array([
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.01, 0.01),
            np.random.uniform(-0.001, 0.001)
        ])
        local_starts.append(best_x + perturbation)
        
    for start in local_starts:
        try:
            res_local = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res_local.fun < best_val:
                best_val = res_local.fun
                best_x = res_local.x
        except Exception:
            pass
            
    return best_x
# EVOLVE-BLOCK-END
