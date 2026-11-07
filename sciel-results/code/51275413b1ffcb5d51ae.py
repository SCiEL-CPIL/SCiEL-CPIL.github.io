"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Precompute Hermite polynomial values and coefficients
# Physicists' Hermite polynomials:
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

# Values at x=0:
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 168.0
H12_0 = 18480.0

def compute_q_poly(c0, c1, c2):
    """Return coefficients of q(x) as polynomial in y=x^2: [a0, a1, a2, a3, a4, a5]
    where q = a0 + a1*y + a2*y^2 + a3*y^3 + a4*y^4 + a5*y^5"""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # After dividing by x^2, we get a polynomial in y=x^2 of degree 5
    
    # x^0 term is 0 by construction of c3
    # x^2 term: c1*(-48) + c2*(-1344) + c3*(-921600)
    # x^4 term: c1*16 + c2*2688 + c3*1424640
    # x^6 term: c2*(-1344) + c3*(-1003520)
    # x^8 term: c2*256 + c3*332160
    # x^10 term: c3*(-46080)
    # x^12 term: c3*4096
    
    # q(x) = g(x)/x^2, so:
    # q = (x^2 coeff) + (x^4 coeff)*x^2 + (x^6 coeff)*x^4 + (x^8 coeff)*x^6 + (x^10 coeff)*x^8 + (x^12 coeff)*x^10
    
    a0 = c1 * (-48) + c2 * (-1344) + c3 * (-921600)
    a1 = c1 * 16 + c2 * 2688 + c3 * 1424640
    a2 = c2 * (-1344) + c3 * (-1003520)
    a3 = c2 * 256 + c3 * 332160
    a4 = c3 * (-46080)
    a5 = c3 * 4096
    
    return np.array([a0, a1, a2, a3, a4, a5])

def find_largest_sign_change_root(coeffs):
    """Find the largest real root of the polynomial (in y) at which it changes sign.
    Returns the root r (in y-space), then value = r / (2*pi).
    If no such root exists, return -inf (meaning the value is very large/bad)."""
    # Find all real roots
    try:
        roots = np.roots(coeffs)
    except Exception:
        return -np.inf
    
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    # Check sign change at each root
    # We need to check if the polynomial changes sign at the root
    best_root = -np.inf
    for root in real_roots:
        if root < 0:
            continue  # y = x^2 >= 0
        # Check sign change: evaluate at root - eps and root + eps
        eps = 1e-10
        try:
            poly_minus = np.polyval(coeffs, root - eps)
            poly_plus = np.polyval(coeffs, root + eps)
        except Exception:
            continue
        # Sign change means poly * poly_plus < 0
        if poly_minus * poly_plus < 0:
            if root > best_root:
                best_root = root
    return best_root

def evaluate_value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) where r is the largest sign-changing root."""
    coeffs = compute_q_poly(c0, c1, c2)
    
    # Check if g tends to negative limit at infinity
    # As x -> infinity, g(x) ~ c3 * 4096 * x^12
    # If c3 < 0, g -> -infinity, so we'd replace g by -g
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    if c3 < 0:
        # Replace g by -g, which means negating all coefficients
        coeffs = -coeffs
        c3 = -c3
    
    if c3 <= 0:
        # Edge case, return large value (bad)
        return 1e10
    
    r = find_largest_sign_change_root(coeffs)
    if r < 0:
        # No valid sign-changing root found
        return 1e10
    
    return r / (2 * np.pi)

def objective(params):
    """Objective function for optimization: minimize the value."""
    c0, c1, c2 = params
    try:
        val = evaluate_value(c0, c1, c2)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    """Search for optimal coefficients c0, c1, c2."""
    start_time = time.time()
    deadline = start_time + 190  # 190 seconds total, leave margin before 210s kill
    
    # Seed solution
    seed = [-113.0/100.0, 1.0/25.0, 1.0/3240.0]
    best_params = seed[:]
    best_value = objective(best_params)
    
    # Strategy 1: Differential evolution with bounded search
    elapsed = time.time() - start_time
    if elapsed < deadline - 80:
        try:
            bounds = [(-5, 5), (-5, 5), (-0.1, 0.1)]
            result = differential_evolution(
                objective, bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if result.fun < best_value:
                best_params = result.x.tolist()
                best_value = result.fun
        except Exception:
            pass
    
    # Strategy 2: Local refinement with Nelder-Mead from best point
    elapsed = time.time() - start_time
    if elapsed < deadline - 40:
        try:
            nm_result = minimize(
                objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if nm_result.fun < best_value:
                best_params = nm_result.x.tolist()
                best_value = nm_result.fun
        except Exception:
            pass
    
    # Strategy 3: Perturb and refine from multiple starting points
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            rng = np.random.RandomState(123)
            for _ in range(20):
                if time.time() > deadline - 20:
                    break
                scale = 0.05 + 0.1 * np.random.rand()
                perturb = np.array(best_params) + rng.normal(0, scale, 3)
                local_result = minimize(
                    objective, perturb,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                if local_result.fun < best_value:
                    best_params = local_result.x.tolist()
                    best_value = local_result.fun
        except Exception:
            pass
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            final_result = minimize(
                objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if final_result.fun < best_value:
                best_params = final_result.x.tolist()
                best_value = final_result.fun
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
