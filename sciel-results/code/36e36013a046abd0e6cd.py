"""Structural change: use bisection-based sign-change detection on q(z) to find the
smallest positive sign-changing root, then optimize c0,c1,c2 to minimize that root.
The key insight from inspirations is that we want the SMALLEST positive sign-changing
root, not the largest. This is a fundamentally different objective landscape."""

import numpy as np
import time
from scipy.optimize import minimize
import sys

# Precompute Hermite polynomial coefficients (low to high power)
def _hermite_coeffs_low(n):
    from sympy import symbols, Poly, hermite_poly
    x = symbols('x')
    p = Poly(hermite_poly(n, x), x)
    coeffs = p.all_coeffs()  # high to low
    return np.array([float(c) for c in reversed(coeffs)])

_H0 = _hermite_coeffs_low(0)    # [1]
_H4 = _hermite_coeffs_low(4)    # [12, 0, -48, 0, 16]
_H8 = _hermite_coeffs_low(8)
_H12 = _hermite_coeffs_low(12)

_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]

def _q_coeffs(c0, c1, c2):
    """
    Compute coefficients of q(z) = g(x)/x^2 where z = x^2.
    g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
    c3 is fixed so g(0) = 0.
    q(z) has degree 5 (since g has degree 12 and we divide by x^2, then substitute z=x^2).
    Returns coefficients of q(z) as array [q0, q1, ..., q5] (low to high).
    """
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # Build g coefficients (low to high, degree 0 to 12)
    g = np.zeros(13)
    for i in range(len(_H0)):
        g[i] += c0 * _H0[i]
    for i in range(len(_H4)):
        g[i] += c1 * _H4[i]
    for i in range(len(_H8)):
        g[i] += c2 * _H8[i]
    for i in range(len(_H12)):
        g[i] += c3 * _H12[i]
    
    # q(z) = sum_{k=0}^{6} g[2k+2] * z^k  (since g/x^2 = sum g[2k] x^{2k-2}, let z=x^2)
    # Actually: g(x) = sum_{k=0}^{6} g[2k] x^{2k}. g(x)/x^2 = sum_{k=0}^{6} g[2k] x^{2k-2}
    # = sum_{j=-1}^{5} g[2(j+1)] x^{2j}. The j=-1 term is g[0] * x^{-2} which should be 0 since g(0)=0.
    # Wait, g(0) = g[0] = 0 by construction. So g[0] = 0.
    # g(x)/x^2 = sum_{k=1}^{6} g[2k] x^{2k-2} = sum_{k=1}^{6} g[2k] z^{k-1}
    # So q(z) = sum_{j=0}^{5} g[2(j+1)] z^j = [g[2], g[4], g[6], g[8], g[10], g[12]]
    
    q = np.array([g[2], g[4], g[6], g[8], g[10], g[12]])
    return q

def _smallest_positive_sign_changing_root(q_coeffs):
    """
    Find the smallest positive real root of q(z) at which q changes sign.
    Uses a combination of:
    1. Finding all real roots via numpy
    2. Verifying sign change via bisection
    Returns 0.0 if no such root exists.
    """
    # Remove leading zeros (highest degree)
    while len(q_coeffs) > 1 and abs(q_coeffs[-1]) < 1e-14:
        q_coeffs = q_coeffs[:-1]
    
    if len(q_coeffs) < 2:
        return 0.0
    
    # Find all roots
    q_high = q_coeffs[::-1]  # high to low for numpy
    roots = np.roots(q_high)
    
    # Filter for positive real roots
    candidates = []
    for r in roots:
        if abs(r.imag) > 1e-10 * max(1.0, abs(r.real)):
            continue
        zr = float(r.real)
        if zr <= 1e-12:
            continue
        candidates.append(zr)
    
    if not candidates:
        return 0.0
    
    # Sort candidates
    candidates.sort()
    
    # For each candidate, verify it's a sign-changing root using bisection
    def q_eval(z):
        # Evaluate q(z) using Horner's method
        result = 0.0
        for c in reversed(q_coeffs):
            result = result * z + c
        return result
    
    def is_sign_changing(zr, tol=1e-8):
        """Check if q changes sign near zr by evaluating at zr-tol and zr+tol."""
        eps = max(tol * zr, 1e-10)
        left = q_eval(zr - eps)
        right = q_eval(zr + eps)
        # Sign change if left and right have opposite signs
        return left * right < 0
    
    for zr in candidates:
        if is_sign_changing(zr):
            return zr
    
    # If no sign-changing root found among numpy roots, do a sweep
    # Scan [0, max_candidate * 1.5] for sign changes
    max_z = max(candidates) * 2.0 if candidates else 10.0
    n_sweep = 1000
    z_vals = np.linspace(1e-6, max_z, n_sweep)
    q_vals = np.array([q_eval(z) for z in z_vals])
    
    for i in range(n_sweep - 1):
        if q_vals[i] * q_vals[i+1] < 0:
            # Sign change between z_vals[i] and z_vals[i+1]
            # Bisection to find the root
            a, b = z_vals[i], z_vals[i+1]
            fa, fb = q_vals[i], q_vals[i+1]
            for _ in range(60):
                m = (a + b) / 2
                fm = q_eval(m)
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            return (a + b) / 2
    
    return 0.0

def _objective(c):
    """Minimize the smallest positive sign-changing root of q(z)."""
    try:
        c0, c1, c2 = c
        q = _q_coeffs(c0, c1, c2)
        r = _smallest_positive_sign_changing_root(q)
        return r
    except Exception:
        return 1e10

def _local_refine(x0, deadline):
    """Local refinement using Nelder-Mead."""
    best_x = x0.copy()
    best_val = _objective(best_x)
    
    # Nelder-Mead with tight tolerances
    try:
        res = minimize(
            _objective,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Second polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 180s limit
    
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_x = seed.copy()
    best_val = _objective(best_x)
    
    # Multi-start local optimization
    np.random.seed(12345)
    
    # Generate diverse starting points
    starts = [seed.copy()]
    
    # Random points in a reasonable box
    for _ in range(80):
        if time.time() > deadline - 40:
            break
        p = np.random.uniform(-5, 5, size=3)
        p[2] *= 0.1  # c2 tends to be small
        starts.append(p)
    
    # Perturbations of seed at various scales
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(10):
            if time.time() > deadline - 40:
                break
            p = seed + scale * np.random.randn(3)
            p[2] *= 0.5  # dampen c2 perturbation
            starts.append(p)
    
    # Grid-like exploration in a smaller box around seed
    for c0 in [-2, -1.5, -1.13, -1.0, -0.5, 0, 0.5, 1.0]:
        for c1 in [-0.5, -0.1, 0.0, 0.04, 0.1, 0.5]:
            if time.time() > deadline - 40:
                break
            for c2 in [-0.01, -0.001, 0.0, 0.0003, 0.001, 0.01]:
                p = np.array([c0, c1, c2])
                starts.append(p)
    
    # Run local refinement from each start
    for i, start in enumerate(starts):
        if time.time() > deadline - 10:
            break
        try:
            x_opt, val_opt = _local_refine(start, deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt
        except Exception:
            continue
    
    # Final high-precision polish
    if best_val < 1e10 and time.time() < deadline - 2:
        try:
            x_opt, val_opt = _local_refine(best_x, deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt
        except Exception:
            pass
    
    return best_x
