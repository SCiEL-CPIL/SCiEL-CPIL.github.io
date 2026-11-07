"""
Direct root-space optimization: parameterize by the roots of Q(z) directly.
Q(z) = a12 * prod_{i=0}^{4} (z - z_i)
Given the 5 roots, recover a12, a10, a8, a6, a4, a2.
Then solve the 2x2 linear system for c1, c2 (c0 is free, set to 1).
Objective: minimize the largest positive sign-changing root.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _Q_coeffs_from_roots(roots):
    """
    Given 5 roots z_0..z_4, compute the Q coefficients [a2, a4, a6, a8, a10, a12].
    Q(z) = a12 * prod (z - z_i)
    Expand to get all coefficients.
    """
    # Build polynomial from roots: numpy.poly gives highest-degree-first
    # poly(roots) = [1, -sum, sum_pairs, -sum_triples, sum_quads, -prod]
    coeffs_hdf = np.poly(roots)  # [1, c4, c3, c2, c1, c0] for z^4..z^0
    # But Q is degree 5, so we need a12 * z^5 + ...
    # Actually numpy.poly of 5 roots gives degree-5 polynomial: [1, ...]
    # coeffs_hdf = [1, p4, p3, p2, p1, p0] where Q(z)/a12 = z^5 + p4*z^4 + ... + p0
    # So a12 * z^5 + a12*p4 * z^4 + ... 
    # We want a12 to be the leading coefficient. But a12 is also the coefficient of z^5.
    # Actually the leading coefficient of Q is a12, so Q(z) = a12 * z^5 + a10*z^4 + a8*z^3 + a6*z^2 + a4*z + a2
    # And Q(z) = a12 * prod(z - z_i) = a12 * (z^5 + p4*z^4 + p3*z^3 + p2*z^2 + p1*z + p0)
    # So: a10 = a12*p4, a8 = a12*p3, a6 = a12*p2, a4 = a12*p1, a2 = a12*p0
    # We have freedom to choose a12. But actually a12 is determined by c3.
    # Wait: a12 = 4096 * c3. And c3 is determined by c0, c1, c2.
    # So a12 is not free. Let me reconsider.
    # 
    # Actually, the mapping from (c0,c1,c2) to Q coefficients is linear.
    # Q(z) = a2 + a4*z + a6*z^2 + a8*z^3 + a10*z^4 + a12*z^5
    # with:
    # a2  = -48*c1 - 13440*c2 - 1290240*c3
    # a4  =  16*c1 + 11520*c2 + 2123328*c3
    # a6  = -3072*c2 - 1548288*c3
    # a8  =  256*c2 + 516096*c3
    # a10 = -73728*c3
    # a12 =  4096*c3
    # c3 = -(c0 + 12*c1 + 1680*c2) / 2032128
    #
    # So a12 = 4096 * c3 = 4096 * (-(c0 + 12*c1 + 1680*c2)) / 2032128
    # a12 = -(4096/2032128) * (c0 + 12*c1 + 1680*c2)
    # 4096/2032128 = 1/496.125 = 16/1984.5... let me compute: 2032128/4096 = 496.125? 
    # 4096 * 496 = 2031616, 2032128 - 2031616 = 512, so 4096*496 + 512 = 4096*496 + 4096/8 = 4096*(496+0.125) = 4096*496.125. Yes.
    # So a12 = -(c0 + 12*c1 + 1680*c2) / 496.125
    #
    # This is getting complex. Let me instead just use the direct approach:
    # Given 5 roots, compute the monic polynomial. Then scale so that the leading
    # coefficient a12 is consistent with the constraint. But actually, we have 3 free
    # parameters (c0, c1, c2) and 6 Q coefficients with 3 constraints (the linear relations).
    # The space of valid Q polynomials is 3-dimensional.
    #
    # Simpler approach: just use the original (c0,c1,c2) parameterization but use a 
    # smarter optimizer. Let me use differential evolution + Nelder-Mead polish.
    #
    # Actually, let me try a different structural approach: 
    # The value is z_max / (2*pi) where z_max is the largest positive sign-changing root.
    # We want to minimize z_max. 
    #
    # Key insight: z_max is a piecewise-smooth function. The breakpoints occur when 
    # roots cross the real axis or when the identity of the "largest positive sign-changing
    # root" changes.
    #
    # Let me just do a very thorough multi-start with a faster objective.
    pass

def _compute_value_fast(c0, c1, c2):
    """Fast computation of the task value."""
    c3 = -(c0 + 12.0*c1 + 1680.0*c2) / 2032128.0
    
    a2  = -48.0*c1 - 13440.0*c2 - 1290240.0*c3
    a4  =  16.0*c1 + 11520.0*c2 + 2123328.0*c3
    a6  = -3072.0*c2 - 1548288.0*c3
    a8  =  256.0*c2 + 516096.0*c3
    a10 = -73728.0*c3
    a12 =  4096.0*c3
    
    if a12 < 0:
        a2, a4, a6, a8, a10, a12 = -a2, -a4, -a6, -a8, -a10, -a12
    
    coeffs = np.array([a12, a10, a8, a6, a4, a2])
    
    # Remove leading zeros
    idx = 0
    while idx < len(coeffs) - 1 and abs(coeffs[idx]) < 1e-14:
        idx += 1
    coeffs = coeffs[idx:]
    
    if len(coeffs) <= 1:
        return 1e10
    
    try:
        roots = np.roots(coeffs)
    except:
        return 1e10
    
    best_z = 0.0
    for rz in roots:
        if abs(rz.imag) < 1e-9:
            zr = rz.real
            if zr > 1e-10:
                eps = 1e-7 * max(1.0, abs(zr))
                q_minus = np.polyval(coeffs, zr - eps)
                q_plus = np.polyval(coeffs, zr + eps)
                if q_minus * q_plus < 0:
                    if zr > best_z:
                        best_z = zr
    
    if best_z == 0.0:
        return 1e10
    
    return best_z / (2.0 * np.pi)

def _objective(params):
    try:
        v = _compute_value_fast(*params)
        return v if np.isfinite(v) and v > 0 else 1e10
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 195.0
    
    seed = np.array([-1.13, 0.04, 1.0/3240.0])
    best_val = _compute_value_fast(*seed)
    best_params = seed.copy()
    
    np.random.seed(42)
    
    # Phase 1: Differential Evolution for global search
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.5, 0.5)]
    if time.time() < deadline - 60.0:
        try:
            de_res = differential_evolution(
                _objective, bounds,
                maxiter=200, tol=1e-12,
                popsize=30, seed=42,
                mutation=(0.3, 0.9), recombination=0.7,
                polish=True, init='latinhypercube'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except:
            pass
    
    # Phase 2: Multi-start Nelder-Mead from diverse points
    starts = [best_params.copy()]
    for _ in range(20):
        pert = np.random.randn(3) * np.array([0.5, 0.1, 0.05])
        starts.append(best_params + pert)
    
    # Add seed neighborhood
    for _ in range(10):
        pert = np.random.randn(3) * np.array([0.3, 0.05, 0.02])
        starts.append(seed + pert)
    
    for s in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(_objective, s, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12,
                                   'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    # Phase 3: Fine polish
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-15,
                                   'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    return list(best_params)
# EVOLVE-BLOCK-END
