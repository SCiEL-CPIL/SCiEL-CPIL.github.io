"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import sys

# Precompute Hermite polynomial coefficients
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 168.0
H12_0 = 18480.0

def compute_q_coeffs(c0, c1, c2):
    """Return coefficients of q(y) = g(x)/x^2 as polynomial in y=x^2.
    q(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    Returns [a0, a1, a2, a3, a4, a5] (ascending order)."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g(x) = c0*(1) + c1*(16x^4 - 48x^2 + 12) + c2*(256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168) + c3*(4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480)
    
    # Coefficients of x^(2k) for k=0..6:
    # k=0: c0*1 + c1*12 + c2*168 + c3*18480 = 0 (by construction)
    # k=1: c1*(-48) + c2*(-1344) + c3*(-921600)
    # k=2: c1*16 + c2*2688 + c3*1424640
    # k=3: c2*(-1344) + c3*(-1003520)
    # k=4: c2*256 + c3*332160
    # k=5: c3*(-46080)
    # k=6: c3*4096
    
    a0 = 0.0  # constant term (should be 0)
    a1 = c1 * (-48) + c2 * (-1344) + c3 * (-921600)
    a2 = c1 * 16 + c2 * 2688 + c3 * 1424640
    a3 = c2 * (-1344) + c3 * (-1003520)
    a4 = c2 * 256 + c3 * 332160
    a5 = c3 * (-46080)
    a6 = c3 * 4096
    
    return np.array([a0, a1, a2, a3, a4, a5, a6])

def find_largest_sign_change_root(q_coeffs):
    """Find the largest positive real root of q(y) at which q changes sign.
    q_coeffs is [a0, a1, ..., a6] ascending."""
    # Use numpy to find roots
    # np.roots expects descending order
    desc = q_coeffs[::-1]
    roots = np.roots(desc)
    
    # Filter real positive roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 1e-12:
            real_roots.append(r.real)
    
    if not real_roots:
        return None
    
    # Check sign change at each root
    # q(y) sign change means the root has odd multiplicity
    # We need to check if q changes sign at this root
    # Evaluate q at points just below and above each candidate root
    
    def q_eval(y):
        return np.polyval(desc, y)
    
    best_root = None
    for r in sorted(real_roots, reverse=True):
        # Check sign change
        eps = 1e-8 * max(1.0, abs(r))
        q_below = q_eval(r - eps)
        q_above = q_eval(r + eps)
        if q_below * q_above < 0:
            best_root = r
            break
    
    return best_root

def evaluate_value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) for given coefficients."""
    q_coeffs = compute_q_coeffs(c0, c1, c2)
    
    # Check if g tends to negative limit at infinity
    # Leading coefficient of g is c3 * 4096 (from H12)
    # g ~ c3 * 4096 * x^12 for large x
    # g/x^2 ~ c3 * 4096 * x^10
    # If c3 < 0, then g/x^2 -> -infinity, so g tends to negative limit
    # If c3 > 0, g tends to positive limit, so we flip g -> -g
    # If c3 == 0, need to check next term
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    if c3 > 0:
        # g tends to +infinity, so we replace g by -g
        # This flips all coefficients
        q_coeffs = -q_coeffs
        # Now leading coeff is -c3*4096 < 0, good
    # If c3 < 0, g tends to -infinity, keep as is
    # If c3 == 0, check a5 (coeff of y^5 in q, which is x^10 term)
    # Actually if c3==0, then a5=0, a6=0, so degree drops
    # Let's just proceed and find the root
    
    r = find_largest_sign_change_root(q_coeffs)
    
    if r is None:
        return float('inf')
    
    return r**2 / (2 * np.pi)

def objective(x):
    c0, c1, c2 = x
    try:
        val = evaluate_value(c0, c1, c2)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # safety margin
    
    seed = np.array([-113/100, 1/25, 1/3240])
    best_val = evaluate_value(seed[0], seed[1], seed[2])
    best_x = seed.copy()
    
    # Try local optimization from seed
    try:
        res = minimize(objective, seed, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        return best_x
    
    # Try differential evolution with reasonable bounds
    # Coefficients should be O(1) or smaller
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    try:
        de_result = differential_evolution(objective, bounds, 
                                           maxiter=200, popsize=20,
                                           tol=1e-12, seed=42,
                                           polish=True)
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_x = de_result.x
    except:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_x
    
    # Try more targeted search: we want to minimize the largest sign-change root
    # Let's try to make q(y) have no positive sign-change roots, or minimize the largest one
    # Try a grid-like search in a smaller region around the best found so far
    
    if best_x is not None and best_val < float('inf'):
        # Local refinement around best
        try:
            res2 = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_x
    
    # Try a few more random starting points
    rng = np.random.RandomState(123)
    for i in range(50):
        if time.time() > deadline - 15:
            break
        x_try = rng.uniform(-3, 3, size=3)
        try:
            val = objective(x_try)
            if val < best_val:
                best_val = val
                best_x = x_try.copy()
                # Refine
                res3 = minimize(objective, x_try, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_x = res3.x
        except:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
