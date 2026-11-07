"""Optimize coefficients for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from fractions import Fraction

# Correct physicists' Hermite polynomials coefficients
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
# H12(x) = 4096x^12 - 75264x^10 + 451584x^8 - 903168x^6 + 691200x^4 - 151200x^2 + 5040

# Values at x=0
H0_0 = 1
H4_0 = 12
H8_0 = 1680
H12_0 = 5040

def compute_q_coefficients(c0, c1, c2):
    """
    Compute coefficients of q(y) = b0 + b1*y + b2*y^2 + b3*y^3 + b4*y^4 + b5*y^5
    where y = x^2, and q(x) = g(x)/x^2.
    """
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) coefficients for x^(2k):
    # k=0: c0 + 12*c1 + 1680*c2 + 5040*c3 = 0
    # k=1: -48*c1 - 13440*c2 - 151200*c3
    # k=2: 16*c1 + 13440*c2 + 691200*c3
    # k=3: -3584*c2 - 903168*c3
    # k=4: 256*c2 + 451584*c3
    # k=5: -75264*c3
    # k=6: 4096*c3
    
    # q(y) = g(x)/x^2 where y=x^2, so q(y) has terms y^0 to y^5
    b0 = -48*c1 - 13440*c2 - 151200*c3
    b1 = 16*c1 + 13440*c2 + 691200*c3
    b2 = -3584*c2 - 903168*c3
    b3 = 256*c2 + 451584*c3
    b4 = -75264*c3
    b5 = 4096*c3
    
    return np.array([b0, b1, b2, b3, b4, b5])

def evaluate_value(c0, c1, c2):
    """
    Compute the value = r^2 / (2*pi) where r is the largest real root of q
    at which q changes sign. Lower is better.
    """
    q_coeffs = compute_q_coefficients(c0, c1, c2)
    
    # q(y) = b0 + b1*y + b2*y^2 + b3*y^3 + b4*y^4 + b5*y^5
    # Find roots of q(y) = 0 for y >= 0
    # Use np.roots with coefficients in descending order
    P_coeffs = q_coeffs[::-1]
    
    # Remove leading zeros
    while len(P_coeffs) > 1 and abs(P_coeffs[0]) < 1e-15:
        P_coeffs = P_coeffs[1:]
    
    if len(P_coeffs) <= 1:
        return 1e10
    
    try:
        roots_y = np.roots(P_coeffs)
    except Exception:
        return 1e10
    
    best_r = 0.0
    for ry in roots_y:
        if abs(ry.imag) > 1e-10:
            continue
        y = ry.real
        if y < -1e-10:
            continue
        y = max(0.0, y)
        x_val = np.sqrt(y)
        
        if y < 1e-10:
            continue
        
        # Check sign change: evaluate q at y-eps and y+eps
        eps = 1e-8
        y_minus = max(0.0, y - eps)
        y_plus = y + eps
        
        # Evaluate q(y) using Horner's method
        def q_eval(y_val):
            result = 0.0
            for coeff in reversed(q_coeffs):
                result = result * y_val + coeff
            return result
        
        p_minus = q_eval(y_minus)
        p_plus = q_eval(y_plus)
        
        if p_minus * p_plus < 0:
            if x_val > best_r:
                best_r = x_val
    
    if best_r == 0.0:
        return 1e10
    
    value = best_r**2 / (2 * np.pi)
    return value

def construct():
    """
    Find three coefficients [c0, c1, c2] that minimize the value.
    """
    start_time = time.time()
    deadline = start_time + 195.0
    
    # Seed
    seed = np.array([-113/100, 1/25, 1/3240])
    best_val = evaluate_value(*seed)
    best_coeffs = seed.copy()
    
    # Try multiple optimization strategies
    # Strategy 1: Nelder-Mead from seed
    try:
        result = minimize(
            lambda x: evaluate_value(*x),
            seed,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 30:
        return best_coeffs
    
    # Strategy 2: Differential evolution for global search
    try:
        bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
        result_de = differential_evolution(
            lambda x: evaluate_value(*x),
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42
        )
        if result_de.fun < best_val:
            best_val = result_de.fun
            best_coeffs = result_de.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 20:
        return best_coeffs
    
    # Strategy 3: Local refinement from best found
    try:
        result2 = minimize(
            lambda x: evaluate_value(*x),
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result2.fun < best_val:
            best_val = result2.fun
            best_coeffs = result2.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 10:
        return best_coeffs
    
    # Strategy 4: Try several random perturbations
    np.random.seed(123)
    for i in range(30):
        if time.time() > deadline - 5:
            break
        perturbation = np.random.randn(3) * 0.05
        init = best_coeffs + perturbation
        try:
            res = minimize(
                lambda x: evaluate_value(*x),
                init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
