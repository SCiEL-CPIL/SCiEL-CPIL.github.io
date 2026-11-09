"""Refined uncertainty inequality: Laguerre double roots (m = 12, n = 1)."""
import time
import math
import numpy as np
import sympy as sp

# EVOLVE-BLOCK-START

def _bessel_j0_zeros(n):
    """Return the first n positive zeros of J_0 using Newton's method on the ratio form."""
    # Use the standard asymptotic starting points and Newton iteration
    # J_0(x) = 0. We use the recurrence-based Newton step.
    # J_0'(x) = -J_1(x).
    # Newton: x_{k+1} = x_k - J_0(x_k) / (-J_1(x_k)) = x_k + J_0(x_k)/J_1(x_k)
    zeros = []
    for k in range(1, n + 1):
        # Initial guess: asymptotic
        x = (k - 0.25) * math.pi
        for _ in range(50):
            # Compute J_0 and J_1 using mpmath-like precision via sympy? 
            # Actually, for speed, use scipy if available, else a simple series.
            # Let's use scipy.special for speed and accuracy.
            try:
                from scipy.special import j0, j1
                j0x = j0(x)
                j1x = j1(x)
            except ImportError:
                # Fallback: use sympy (slow)
                j0x = float(sp.besselfunctions.besselfunctions.besselj(0, x).evalf())
                j1x = float(sp.besselfunctions.besselfunctions.besselj(1, x).evalf())
            if abs(j1x) < 1e-15:
                break
            dx = -j0x / j1x
            x += dx
            if abs(dx) < 1e-15:
                break
        zeros.append(x)
    return zeros


def _evaluate(z_list):
    """
    Given a list of 12 positive distinct floats, compute the score.
    Returns (score, valid) where valid is bool.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin

    z_sympy = [sp.Rational(z) for z in z_list]
    z_sympy = sorted(z_sympy)

    # Check distinctness and positivity
    if any(z <= 0 for z in z_sympy):
        return None, False
    if len(set(z_sympy)) != 12:
        return None, False

    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)

    # Build the 26 Laguerre polynomials L_0, L_2, ..., L_50
    # L_k^(alpha)(x)
    # We need to solve for c_0..c_25
    # Equations:
    # g(0) = 0
    # g'(0) = 1
    # For each z_i: g(z_i) = 0, g'(z_i) = 0  (24 equations)
    # Total: 2 + 24 = 26 equations for 26 unknowns.

    # Precompute Laguerre polynomials and their derivatives at the points
    # L_k^(alpha)(x)
    # We'll build the matrix A (26x26) and vector b (26x1)

    # Indices: 0..25 correspond to degrees 0, 2, ..., 50
    # A[i][j] is the coefficient of c_j in equation i.

    # Let's compute L_k^(alpha)(x) and d/dx L_k^(alpha)(x) symbolically or numerically?
    # The task says "solved exactly (sympy LUsolve)". So we must use sympy.
    # This will be slow. Let's optimize by precomputing the Laguerre values.

    # L_k^(alpha)(x) = sum_{m=0}^k binom(k+alpha, k-m) * (-x)^m / m!
    # We can compute these exactly using sympy.

    # To speed up, let's compute the values at x=0 and x=z_i for all k=0,2,...,50
    # and also the derivatives.

    # L_k^(alpha)(0) = binom(k+alpha, k)
    # d/dx L_k^(alpha)(x) at x=0: 
    # L_k^(alpha)(x) = sum_{m=0}^k binom(k+alpha, k-m) (-1)^m x^m / m!
    # d/dx L_k^(alpha)(x) = sum_{m=1}^k binom(k+alpha, k-m) (-1)^m m x^{m-1} / m!
    # at x=0, only m=1 term survives: binom(k+alpha, k-1) (-1)^1 * 1 / 1! = -binom(k+alpha, k-1)

    # For general x, we need to evaluate the sum. This is expensive.
    # Let's use sympy's genlaguerre function which might be optimized.

    # Actually, let's just build the matrix using sympy's genlaguerre.
    # L_k^(alpha)(x) = sp.genlaguerre(k, alpha, x)

    # We need 26 polynomials of degrees 0, 2, ..., 50.
    # Let's create them.
    lags = [sp.genlaguerre(2*k, alpha, x) for k in range(26)]
    lag_derivs = [sp.diff(lag, x) for lag in lags]

    # Build the system
    # Equation 0: g(0) = 0 => sum c_k * L_k(0) = 0
    # Equation 1: g'(0) = 1 => sum c_k * L_k'(0) = 1
    # Equations 2..25: for each z_i, g(z_i)=0, g'(z_i)=0

    A = sp.zeros(26, 26)
    b = sp.zeros(26, 1)

    # Fill in row 0: g(0) = 0
    for k in range(26):
        A[0, k] = lags[k].subs(x, 0)
    b[0, 0] = 0

    # Fill in row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = lag_derivs[k].subs(x, 0)
    b[1, 0] = 1

    # Fill in rows 2..25: for each z_i
    for i, z in enumerate(z_sympy):
        row_g = 2 + 2*i
        row_gp = 3 + 2*i
        for k in range(26):
            A[row_g, k] = lags[k].subs(x, z)
            A[row_gp, k] = lag_derivs[k].subs(x, z)
        b[row_g, 0] = 0
        b[row_gp, 0] = 0

    # Solve the system
    try:
        c = sp.LUsolve(A, b)
    except Exception:
        return None, False

    # Check if the system is singular (LUsolve would have raised an exception, but let's be safe)
    if c.has(sp.Zoo) or c.has(sp.NaN):
        return None, False

    # Construct g(x) = sum c_k L_2k(x)
    g = sum(c[k] * lags[k] for k in range(26))
    g = sp.simplify(g)

    # Construct the denominator: x * prod_i (x - z_i)^2
    denom = x
    for z in z_sympy:
        denom = denom * (x - z)**2
    denom = sp.expand(denom)

    # Perform exact polynomial division: q(x) = g(x) / denom
    # We expect g(x) to be divisible by denom.
    try:
        q, r = sp.div(g, denom, x)
    except Exception:
        return None, False

    # If the remainder is not zero, something went wrong.
    if r != 0:
        return None, False

    # Find real roots of q
    # q is a polynomial of degree 50 - 25 = 25.
    # Use sympy.real_roots
    try:
        roots = sp.real_roots(q)
    except Exception:
        return None, False

    # For each real root r, check if q has strictly opposite signs at r - 1e-198 and r + 1e-198
    # Note: 1e-198 is a float, but we're working with exact rationals.
    # The task says "sympy.real_roots" and "strictly opposite signs at r - 1e-198 and r + 1e-198".
    # Since r is a sympy object (likely an algebraic number or rational), we need to evaluate q at these points.
    # But 1e-198 is a float. Let's convert it to a Rational?
    # The task says "converted exactly to a rational, sympy.Rational(float(z_i))".
    # For the sign check, it says "r - 1e-198". Let's use sp.Rational(1, 10**198) as an approximation.
    # Actually, 1e-198 is not exactly representable. Let's use sp.Float(1e-198) or just 1e-198 as a float.
    # The task says "All arithmetic is exact rational", but the sign check uses 1e-198.
    # Let's interpret 1e-198 as a float and evaluate q(r + delta) and q(r - delta) numerically.
    # But q is a sympy polynomial. We can convert it to a lambda function or use .evalf().

    # Let's convert q to a numerical polynomial for evaluation.
    # q is a sympy Poly. We can get its coefficients.
    q_poly = sp.Poly(q, x)
    coeffs = q_poly.all_coeffs()  # list of sympy Rationals
    # Convert to floats for evaluation
    coeffs_float = [float(c) for c in coeffs]

    # Build a numpy poly1d for fast evaluation
    q_np = np.poly1d(coeffs_float)

    counted_roots = []
    for r in roots:
        # r is a sympy object. Convert to float.
        r_float = float(r)
        if not np.isfinite(r_float):
            continue
        # Check sign at r - 1e-198 and r + 1e-198
        delta = 1e-198
        # If r is very close to 0, r - delta might be negative. But q is a polynomial, so it's fine.
        q_minus = q_np(r_float - delta)
        q_plus = q_np(r_float + delta)
        # Check strictly opposite signs
        if q_minus * q_plus < 0:
            counted_roots.append(r_float)

    if not counted_roots:
        B = 0.0
    else:
        B = max(counted_roots)
        if B < 0:
            B = 0.0

    score = B / (2 * math.pi)
    return score, True


def construct():
    """
    Return 12 double-root locations z_1..z_12.
    """
    start_time = time.time()
    deadline = start_time + 620.0

    # Strategy: Use Bessel zeros of J_0.
    # The knowledge says: "Bessel zeros (k*pi)^2 align with the asymptotic zero distribution..."
    # Wait, the knowledge says "(k*pi)^2". Let's re-read.
    # "Bessel zeros (k*pi)^2 align with the asymptotic zero distribution of Laguerre polynomials..."
    # This suggests z_i = (j_{0,i})^2 where j_{0,i} is the i-th zero of J_0.
    # Let's compute the first 12 zeros of J_0 and square them.

    try:
        from scipy.special import j0, j1
    except ImportError:
        # Fallback to a simple search if scipy is not available
        pass

    # Compute zeros of J_0
    zeros_j0 = []
    for k in range(1, 13):
        x = (k - 0.25) * math.pi
        # Newton iteration
        for _ in range(50):
            try:
                from scipy.special import j0, j1
                j0x = j0(x)
                j1x = j1(x)
            except ImportError:
                # Use a rough approximation or break
                break
            if abs(j1x) < 1e-15:
                break
            dx = -j0x / j1x
            x += dx
            if abs(dx) < 1e-15:
                break
        zeros_j0.append(x)

    # Square them to get z_i
    z_list = [z**2 for z in zeros_j0]

    # Ensure they are positive and distinct
    # J_0 zeros are positive and distinct, so their squares are positive and distinct.
    # But let's double check.
    z_list = sorted(z_list)
    if len(set(z_list)) != 12:
        # Fallback to a simple sequence
        z_list = [4.0 * (i + 1) for i in range(12)]

    # Evaluate the score
    score, valid = _evaluate(z_list)

    if not valid:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]

    return z_list

# EVOLVE-BLOCK-END
