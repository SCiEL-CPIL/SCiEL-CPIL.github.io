"""Bootstrap search program for the Uncertainty Inequality task."""
from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar, brentq

# Deadline: evaluator kills at 210s. Use 195s budget to be safe.
_BUDGET_SECONDS = 195.0

# Hermite Polynomials H_n(x) coefficients (Physicists' Hermite)
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 24192x^4 - 40320x^2 + 1680
# H_12(x) = 4096x^12 - 147456x^10 + 1684608x^8 - 9067200x^6 + 20901120x^4 - 20321280x^2 + 665280

# We work with q(y) = g(x)/x^2 where y=x^2.
# g(x) = c0 H0 + c1 H4 + c2 H8 + c3 H12
# g(0) = 0 => c3 = -(c0 H0(0) + c1 H4(0) + c2 H8(0)) / H12(0)
# H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280

# Let's define the polynomials in y for the basis functions.
# H0/x^2 is singular, but g(0)=0 handles the constant term.
# Actually, let's express q(y) directly in terms of c0, c1, c2.

# H4(x) = 16x^4 - 48x^2 + 12 => H4/x^2 = 16y - 48 + 12/y (singular at 0)
# Wait, g(x) is a polynomial. g(0)=0.
# g(x) = c0(1) + c1(16x^4 - 48x^2 + 12) + c2(256x^8 - 4480x^6 + 24192x^4 - 40320x^2 + 1680) 
#        + c3(4096x^12 - 147456x^10 + 1684608x^8 - 9067200x^6 + 20901120x^4 - 20321280x^2 + 665280)
#
# g(x) = A12 x^12 + A10 x^10 + A8 x^8 + A6 x^6 + A4 x^4 + A2 x^2 + A0
# A0 = c0 + 12c1 + 1680c2 + 665280c3 = 0
#
# q(x) = g(x)/x^2 = A12 x^10 + A10 x^8 + A8 x^6 + A6 x^4 + A4 x^2 + A2
# Let y = x^2.
# Q(y) = A12 y^5 + A10 y^4 + A8 y^3 + A6 y^2 + A4 y + A2

# Coefficients A_k in terms of c0, c1, c2, c3:
# A12 = 4096 c3
# A10 = -147456 c3
# A8  = 1684608 c3 + 256 c2
# A6  = -9067200 c3 - 4480 c2
# A4  = 20901120 c3 + 24192 c2 + 16 c1
# A2  = -20321280 c3 - 40320 c2 - 48 c1
# A0  = 665280 c3 + 1680 c2 + 12 c1 + c0 = 0 => c3 = -(c0 + 12c1 + 1680c2) / 665280

# Substitute c3 into A_k to get Q(y) in terms of c0, c1, c2.
# This defines a linear map from c = [c0, c1, c2] to coeffs of Q(y).

def _get_Q_coeffs(c0, c1, c2):
    """Returns coefficients [A2, A4, A6, A8, A10, A12] of Q(y) in ascending order."""
    c3 = -(c0 + 12.0*c1 + 1680.0*c2) / 665280.0
    
    A12 = 4096.0 * c3
    A10 = -147456.0 * c3
    A8  = 1684608.0 * c3 + 256.0 * c2
    A6  = -9067200.0 * c3 - 4480.0 * c2
    A4  = 20901120.0 * c3 + 24192.0 * c2 + 16.0 * c1
    A2  = -20321280.0 * c3 - 40320.0 * c2 - 48.0 * c1
    
    # Check leading coefficient sign.
    # If A12 < 0, g -> -inf, so we replace g by -g.
    # This flips all A_k.
    if A12 < 0:
        A12 = -A12
        A10 = -A10
        A8  = -A8
        A6  = -A6
        A4  = -A4
        A2  = -A2
        
    return np.array([A2, A4, A6, A8, A10, A12])

def _evaluate_from_coeffs(coeffs):
    """
    Given coeffs of Q(y) [A2..A12], find the largest real positive root 
    where Q changes sign.
    Returns value = r^2 / (2*pi) = y_root / (2*pi).
    """
    if coeffs[5] == 0:
        # Degenerate case, lower degree
        # Trim trailing zeros
        c = coeffs.copy()
        while len(c) > 1 and abs(c[-1]) < 1e-12:
            c = c[:-1]
        coeffs = c
    
    # Find roots of Q(y) = 0
    # np.roots expects descending order
    roots = np.roots(coeffs[::-1])
    
    max_sign_change_y = 0.0
    
    for r in roots:
        if abs(r.imag) < 1e-9:
            y_real = r.real
            if y_real > 1e-9:
                # Check sign change
                eps = 1e-7 * max(1.0, abs(y_real))
                val_minus = np.polyval(coeffs[::-1], y_real - eps)
                val_plus = np.polyval(coeffs[::-1], y_real + eps)
                if val_minus * val_plus < 0:
                    if y_real > max_sign_change_y:
                        max_sign_change_y = y_real
                        
    if max_sign_change_y == 0.0:
        return float('inf')
        
    return max_sign_change_y / (2.0 * np.pi)

def _evaluate(c0, c1, c2):
    try:
        coeffs = _get_Q_coeffs(c0, c1, c2)
        return _evaluate_from_coeffs(coeffs)
    except Exception:
        return float('inf')

def _seed_value():
    return _evaluate(-113.0/100.0, 1.0/25.0, 1.0/3240.0)

def construct():
    start_time = time.time()
    
    def _time_left():
        return _BUDGET_SECONDS - (time.time() - start_time)
        
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_value = _evaluate(best_c[0], best_c[1], best_c[2])
    
    # Strategy: Root-based parameterization
    # We assume the optimal Q(y) has a double root at y=R (tangency).
    # Q(R) = 0 and Q'(R) = 0.
    # Q(y) = c0 Q0(y) + c1 Q1(y) + c2 Q2(y)
    # Let's define the basis polynomials Q_i(y) corresponding to c_i=1, others=0.
    
    # Basis for c0=1, c1=0, c2=0
    c0_val = 1.0; c1_val = 0.0; c2_val = 0.0
    Q0_coeffs = _get_Q_coeffs(c0_val, c1_val, c2_val)
    
    # Basis for c0=0, c1=1, c2=0
    c0_val = 0.0; c1_val = 1.0; c2_val = 0.0
    Q1_coeffs = _get_Q_coeffs(c0_val, c1_val, c2_val)
    
    # Basis for c0=0, c1=0, c2=1
    c0_val = 0.0; c1_val = 0.0; c2_val = 1.0
    Q2_coeffs = _get_Q_coeffs(c0_val, c1_val, c2_val)
    
    # Q(y) = c0*Q0 + c1*Q1 + c2*Q2
    # Coeffs are [A2, A4, A6, A8, A10, A12]
    
    # We want to minimize R.
    # For a given R, we require:
    # 1. Q(R) = 0
    # 2. Q'(R) = 0
    # This is a homogeneous system for c0, c1, c2? No, Q is linear in c.
    # So we have 2 linear equations in 3 unknowns (c0, c1, c2).
    # The solution space is a line (1D).
    # We can pick a normalization, e.g., c0^2+c1^2+c2^2=1, or just find the null space of the 2x3 matrix.
    # Then we check if this specific Q(y) has R as the LARGEST sign-changing root.
    # If it does, value = R / (2*pi).
    # We can then scan R over a range to find the minimum valid R.
    
    # Matrix M(R) such that M(R) @ c = 0
    # Row 1: Coeffs of Q_i evaluated at R
    # Row 2: Coeffs of Q'_i evaluated at R
    
    def _find_null_space(c_matrix):
        # c_matrix is 2x3
        # Find null space
        U, S, Vh = np.linalg.svd(c_matrix)
        # Last row of Vh is the null vector
        v = Vh[-1, :]
        return v

    def _check_validity(R, c_vec):
        # Evaluate Q(y) for this c_vec
        coeffs = c_vec[0]*Q0_coeffs + c_vec[1]*Q1_coeffs + c_vec[2]*Q2_coeffs
        # Check if R is the largest sign-changing root
        # We know Q(R)=0 and Q'(R)=0 (by construction, approximately)
        # We need to ensure no larger root exists with sign change.
        # And that R itself is a sign-change root.
        # A double root is NOT a sign-change root!
        # Q(y) = (y-R)^2 * ... does not change sign at R.
        # So the "double root" assumption is WRONG for "sign change".
        
        # Sign change requires an ODD multiplicity root.
        # So Q(R)=0, Q'(R) != 0.
        # This is 1 constraint.
        # We have 3 degrees of freedom.
        # We can set Q(R)=0.
        # Then we have a 2D family of polynomials vanishing at R.
        # We want to minimize R such that R is the largest sign-changing root.
        
        # Let's refine:
        # For a fixed R, find c such that Q(R)=0.
        # This is a 2D subspace of c.
        # Within this subspace, we want to choose c such that:
        # 1. R is a sign-changing root (Q'(R) != 0).
        # 2. No other sign-changing root > R exists.
        # 3. R is as small as possible.
        
        # This is complex. Let's try a different angle.
        # The seed value is 1.0.
        # Let's just do a smarter local search on c, but using the root structure to guide steps.
        
        return True # Placeholder

    # Since the double root idea failed for sign-change, let's stick to a robust global search 
    # but with a better initialization and a specific focus on the "boundary" of the feasible region.
    
    # The feasible region is defined by:
    # 1. Leading coeff > 0 (handled by flip)
    # 2. Largest sign-changing root is the one we measure.
    
    # Let's use a multi-start Nelder-Mead with a penalty for non-smoothness.
    
    rng = np.random.default_rng(42)
    
    # Start points:
    # 1. Seed
    starts = [best_c.copy()]
    
    # 2. Random points in a box around seed
    for _ in range(50):
        c_rand = best_c + rng.normal(0, 0.5, 3)
        starts.append(c_rand)
        
    # 3. Points on the "boundary" where a root is at 0?
    # If R is small, value is small.
    # Can R be arbitrarily small?
    # Q(0) = A2. If A2=0, then y=0 is a root.
    # But we need y>0.
    
    # Let's run Nelder-Mead on each start
    for c_start in starts:
        if _time_left() < 5:
            break
            
        try:
            res = minimize(
                lambda x: _evaluate(x[0], x[1], x[2]),
                c_start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass

    # Final polish
    if _time_left() > 10:
        try:
            res = minimize(
                lambda x: _evaluate(x[0], x[1], x[2]),
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass

    return np.array(best_c, dtype=np.float64)

# EVOLVE-BLOCK-END
