"""Search-based solver for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from sympy import (
    Rational, symbols, Poly, hermite_poly, nsolve,
    sympify, real_roots, sqrt, pi, S, Integer
)
from sympy.polys.polytools import Poly as SymPyPoly
import random
from fractions import Fraction

# Precompute the Hermite polynomial values at 0 and the polynomial structure
_x = symbols('x')
_H0 = hermite_poly(0, _x)
_H4 = hermite_poly(4, _x)
_H8 = hermite_poly(8, _x)
_H12 = hermite_poly(12, _x)

_H0_0 = _H0.subs(_x, 0)  # 1
_H4_0 = _H4.subs(_x, 0)  # 12
_H8_0 = _H8.subs(_x, 0)  # 1008
_H12_0 = _H12.subs(_x, 0)

# Precompute the coefficients of the Hermite polynomials as lists
# H_n(x) = sum_{k=0}^{n} h_{n,k} x^k
# We'll work with the z-polynomial where z = x^2
# g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
# Since all H_n for even n are even polynomials, g(x) is even
# g(x) = x^2 * q(x) where q is a polynomial in x^2 = z
# q(z) is a degree 5 polynomial in z

# Let's extract the coefficients of each H_{2k} in terms of z = x^2
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 53760x^2 + 1008
# H_12(x) = 4096x^12 - 122880x^10 + 1548288x^8 - 10752000x^6 + 37158400x^4 - 53084160x^2 + 103950

# In terms of z = x^2:
# H_0(z) = 1
# H_4(z) = 16z^2 - 48z + 12
# H_8(z) = 256z^4 - 4480z^3 + 26880z^2 - 53760z + 1008
# H_12(z) = 4096z^6 - 122880z^5 + 1548288z^4 - 10752000z^3 + 37158400z^2 - 53084160z + 103950

# But wait, g(x) = x^2 * q(x), so q(x) = g(x)/x^2
# Let's compute the coefficients of g in terms of x, then divide by x^2

# Actually, let's just build the polynomial directly and extract q's coefficients

# Precompute coefficient tables for the Hermite polynomials in z = x^2
# H_0(z) = 1
# H_4(z) = 16z^2 - 48z + 12
# H_8(z) = 256z^4 - 4480z^3 + 26880z^2 - 53760z + 1008
# H_12(z) = 4096z^6 - 122880z^5 + 1548288z^4 - 10752000z^3 + 37158400z^2 - 53084160z + 103950

_H0_z_coeffs = [1, 0, 0, 0, 0, 0]  # degree 0
_H4_z_coeffs = [12, -48, 16, 0, 0, 0]  # degree 2
_H8_z_coeffs = [1008, -53760, 26880, -4480, 256, 0]  # degree 4
_H12_z_coeffs = [103950, -53084160, 37158400, -10752000, 1548288, -122880, 4096]  # degree 6

# But wait - g(x) = x^2 * q(x), and q(x) is in x^2 = z
# g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
# Each H_{2k}(x) is a polynomial in x^2 = z
# So g(x) = G(z) where G is a polynomial in z of degree 6
# g(0) = G(0) = 0, so G(z) has no constant term
# G(z) = z * Q(z) where Q is degree 5 in z
# And q(x) = g(x)/x^2 = G(z)/z = Q(z)

# So Q(z) = c0*H0_z(z) + c1*H4_z(z) + c2*H8_z(z) + c3*H12_z(z) all divided by z
# i.e., Q(z) = (c0*H0_z(z) + c1*H4_z(z) + c2*H8_z(z) + c3*H12_z(z)) / z

# Since G(0) = 0, the constant term of G is zero, so we can divide by z
# G(z) = sum_{k=1}^{6} g_k z^k, Q(z) = sum_{k=0}^{5} g_{k+1} z^k

# G(z) coefficients:
# g_0 = c0*1 + c1*12 + c2*1008 + c3*103950 = 0 (by construction of c3)
# g_1 = c1*(-48) + c2*(-53760) + c3*(-53084160)
# g_2 = c0*0 + c1*16 + c2*26880 + c3*37158400
# g_3 = c2*(-4480) + c3*(-10752000)
# g_4 = c2*256 + c3*1548288
# g_5 = c3*(-122880)
# g_6 = c3*4096

def _build_q_coeffs(c0_r, c1_r, c2_r):
    """Build the coefficients of q(z) where z = x^2, q is degree 5 in z.
    Returns list [q0, q1, q2, q3, q4, q5] where q(z) = q0 + q1*z + ... + q5*z^5."""
    c3_r = -(c0_r * _H0_0 + c1_r * _H4_0 + c2_r * _H8_0) / _H12_0
    
    # q(z) = G(z)/z where G(z) = c0*H0_z(z) + c1*H4_z(z) + c2*H8_z(z) + c3*H12_z(z)
    # G has 7 coefficients (degree 6), q has 6 coefficients (degree 5)
    # q_j = G_{j+1} for j=0..5
    
    # G coefficients (index 0 to 6):
    # G_0 = c0*1 + c1*12 + c2*1008 + c3*103950 = 0
    # G_1 = c1*(-48) + c2*(-53760) + c3*(-53084160)
    # G_2 = c1*16 + c2*26880 + c3*37158400
    # G_3 = c2*(-4480) + c3*(-10752000)
    # G_4 = c2*256 + c3*1548288
    # G_5 = c3*(-122880)
    # G_6 = c3*4096
    
    q_coeffs = [
        c1_r * (-48) + c2_r * (-53760) + c3_r * (-53084160),
        c1_r * 16 + c2_r * 26880 + c3_r * 37158400,
        c2_r * (-4480) + c3_r * (-10752000),
        c2_r * 256 + c3_r * 1548288,
        c3_r * (-122880),
        c3_r * 4096
    ]
    
    return q_coeffs, c3_r


def compute_value_exact(c0, c1, c2, high_precision=False):
    """
    Given three floats, convert to nearby rationals, compute the value.
    Returns the value r^2/(2*pi) where r is the largest real root of q at which q changes sign.
    If no valid root, returns infinity.
    """
    try:
        # Convert floats to rationals using Fraction for exact float representation
        c0_r = Fraction(c0)
        c1_r = Fraction(c1)
        c2_r = Fraction(c2)
        
        # Build q coefficients
        q_coeffs, c3_r = _build_q_coeffs(c0_r, c1_r, c2_r)
        
        if all(c == 0 for c in q_coeffs):
            return float('inf')
        
        # q(z) = sum_{j=0}^{5} q_coeffs[j] * z^j, where z = x^2
        # We need the largest real root r of q(x) at which q changes sign
        # Since q is even, roots come in pairs +/-r
        # q(x) = q(z) where z = x^2
        # A root r of q(x) corresponds to z = r^2 being a root of q(z)
        # q changes sign at r iff q(z) changes sign at z = r^2 (for z > 0)
        
        # Use numpy for fast root finding
        coeffs_np = np.array([float(c) for c in reversed(q_coeffs)])  # highest degree first
        roots_np = np.roots(coeffs_np)
        
        # Filter for positive real roots
        positive_real_roots = []
        for root in roots_np:
            if abs(root.imag) < 1e-10 and root.real > 0:
                positive_real_roots.append(root.real)
        
        if not positive_real_roots:
            return float('inf')
        
        # Evaluate q(z) at a point
        def qz_val(z_val):
            # Horner's method
            result = q_coeffs[5]
            for j in range(4, -1, -1):
                result = result * z_val + q_coeffs[j]
            return float(result)
        
        valid_z_roots = []
        for z0 in positive_real_roots:
            # Check if q changes sign at z0
            # Use a small relative epsilon
            eps = max(1e-10, z0 * 1e-10)
            left_val = qz_val(z0 - eps)
            right_val = qz_val(z0 + eps)
            
            if left_val * right_val < 0:
                valid_z_roots.append(z0)
        
        if not valid_z_roots:
            return float('inf')
        
        # The largest z0 at which q changes sign
        z_max = max(valid_z_roots)
        
        # Value = r^2 / (2*pi) = z_max / (2*pi)
        value = z_max / (2 * float(np.pi))
        
        return value
    
    except Exception as e:
        return float('inf')


def construct():
    """
    Generate candidate coefficients and return the best one found.
    """
    start_time = time.time()
    deadline = start_time + 205.0  # Safety margin before 210s
    
    # Seed
    seed = [-113/100, 1/25, 1/3240]
    best_value = compute_value_exact(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    
    # If seed value is infinity, we need a different approach
    if best_value == float('inf'):
        # Try some other starting points
        test_points = [
            [0.0, 0.0, 0.0],
            [-0.5, 0.1, 0.01],
            [-1.0, 0.2, 0.02],
            [-0.1, 0.01, 0.001],
        ]
        for tp in test_points:
            v = compute_value_exact(tp[0], tp[1], tp[2])
            if v < best_value:
                best_value = v
                best_coeffs = tp.copy()
    
    # Random search with local refinement
    random.seed(42)
    
    # Define a function to evaluate a candidate
    def evaluate(c0, c1, c2):
        return compute_value_exact(c0, c1, c2)
    
    # Nelder-Mead style local optimization
    def nelder_mead(c0, c1, c2, max_iter=200, tol=1e-10):
        """Simple Nelder-Mead optimization for 3 parameters."""
        # Initial simplex
        step = 0.01
        simplex = [
            [c0, c1, c2],
            [c0 + step, c1, c2],
            [c0, c1 + step, c2],
            [c0, c1, c2 + step]
        ]
        
        # Evaluate initial simplex
        values = [evaluate(s[0], s[1], s[2]) for s in simplex]
        
        for _ in range(max_iter):
            # Sort by value
            order = sorted(range(4), key=lambda i: values[i])
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            # Check convergence
            if abs(values[0] - values[3]) < tol:
                break
            
            # Centroid of all but worst
            centroid = [
                sum(simplex[i][j] for i in range(3)) / 3.0
                for j in range(3)
            ]
            
            # Reflection
            xr = [2 * centroid[j] - simplex[3][j] for j in range(3)]
            fr = evaluate(xr[0], xr[1], xr[2])
            
            if fr < values[0]:
                # Expansion
                xe = [centroid[j] + 2 * (xr[j] - centroid[j]) for j in range(3)]
                fe = evaluate(xe[0], xe[1], xe[2])
                if fe < fr:
                    simplex[3] = xe
                    values[3] = fe
                else:
                    simplex[3] = xr
                    values[3] = fr
            elif fr < values[2]:
                # Accept reflection
                simplex[3] = xr
                values[3] = fr
            else:
                # Contraction
                xc = [centroid[j] + 0.5 * (simplex[3][j] - centroid[j]) for j in range(3)]
                fc = evaluate(xc[0], xc[1], xc[2])
                if fc < values[3]:
                    simplex[3] = xc
                    values[3] = fc
                else:
                    # Shrink
                    for i in range(1, 4):
                        simplex[i] = [(simplex[i][j] + simplex[0][j]) / 2 for j in range(3)]
                        values[i] = evaluate(simplex[i][0], simplex[i][1], simplex[i][2])
        
        # Return best point
        order = sorted(range(4), key=lambda i: values[i])
        best_idx = order[0]
        return simplex[best_idx], values[best_idx]
    
    # Multi-start Nelder-Mead
    n_starts = 0
    max_starts = 50
    
    while time.time() < deadline - 10.0 and n_starts < max_starts:
        # Start from current best with small perturbation
        scale = 0.05
        c0 = best_coeffs[0] + random.gauss(0, scale)
        c1 = best_coeffs[1] + random.gauss(0, scale * 0.1)
        c2 = best_coeffs[2] + random.gauss(0, scale * 0.01)
        
        result, val = nelder_mead(c0, c1, c2, max_iter=100)
        if val < best_value:
            best_value = val
            best_coeffs = result
            n_starts = 0  # Reset counter
        
        n_starts += 1
    
    # Broader random search with local refinement
    n_broad = 0
    max_broad = 200
    
    while time.time() < deadline - 10.0 and n_broad < max_broad:
        # Sample in a wider range
        c0 = random.uniform(-2.0, 0.5)
        c1 = random.uniform(-0.5, 0.5)
        c2 = random.uniform(-0.1, 0.1)
        
        result, val = nelder_mead(c0, c1, c2, max_iter=50)
        if val < best_value:
            best_value = val
            best_coeffs = result
            n_broad = 0  # Reset
        
        n_broad += 1
    
    # Final high-precision polish using coordinate descent with very small steps
    scale = 1e-6
    n_fine = 0
    max_fine = 1000
    
    while time.time() < deadline - 2.0 and n_fine < max_fine:
        coord = random.randint(0, 2)
        step = random.gauss(0, scale)
        new_coeffs = best_coeffs.copy()
        new_coeffs[coord] += step
        
        val = evaluate(new_coeffs[0], new_coeffs[1], new_coeffs[2])
        if val < best_value:
            best_value = val
            best_coeffs = new_coeffs
            scale *= 0.95
            n_fine = 0  # Reset
        else:
            scale *= 0.98
        n_fine += 1
    
    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
