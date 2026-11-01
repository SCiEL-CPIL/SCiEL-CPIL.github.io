"""Optimized uncertainty bound using correct Fourier model and analytical moments."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200

# Precompute moments I[2n] = int x^(2n) e^-x^2 dx
# I[2n] = (2n-1)!! / 2^n * sqrt(pi)
def _init_moments():
    moments = np.zeros(25) # up to x^24 maybe?
    moments[0] = np.sqrt(np.pi)
    for n in range(1, 13):
        # I[2n] = I[2n-2] * (2n-1) / (2n)
        moments[2*n] = moments[2*n-2] * (2*n - 1) / (2*n)
    return moments

MOMENTS = _init_moments()

# Precompute Q coefficients as function of (a0, a1, a2)
# P(x) = a0*H0 + a1*H4 + a2*H8 + a3*H12 where a3 = -(a0+3a1+105a2)/945
# Q(x) = P'(x) - x*P(x)
# This is linear in a0, a1, a2, so Q = a0*Q0 + a1*Q1 + a2*Q2
# where Qk = (Hk)' - x*Hk

def _compute_q_basis():
    """Compute Q0, Q1, Q2 as coefficient arrays"""
    # H0 = 1, H0' = 0, x*H0 = x
    # Q0 = 0 - x = -x => coeff: [0, -1, 0, ...]
    
    # H4 = x^4 - 6x^2 + 3, H4' = 4x^3 - 12x, x*H4 = x^5 - 6x^3 + 3x
    # Q1 = (4x^3 - 12x) - (x^5 - 6x^3 + 3x) = -x^5 + 10x^3 - 15x
    
    # H8 = x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
    # H8' = 8x^7 - 168x^5 + 1680x^3 - 3360x
    # x*H8 = x^9 - 28x^7 + 420x^5 - 1680x^3 + 105x
    # Q2 = (8x^7 - 168x^5 + 1680x^3 - 3360x) - (x^9 - 28x^7 + 420x^5 - 1680x^3 + 105x)
    #    = -x^9 + 36x^7 - 588x^5 + 3360x^3 - 3465x
    
    Q0 = np.zeros(14)
    Q0[1] = -1.0
    
    Q1 = np.zeros(14)
    Q1[1] = -15.0
    Q1[3] = 10.0
    Q1[5] = -1.0
    
    Q2 = np.zeros(14)
    Q2[1] = -3465.0
    Q2[3] = 3360.0
    Q2[5] = -588.0
    Q2[7] = 36.0
    Q2[9] = -1.0
    
    return Q0, Q1, Q2

Q0, Q1, Q2 = _compute_q_basis()

def _poly_coeffs(a0, a1, a2):
    """
    Returns coefficients c_j for P(x) = sum c_j x^j
    P(x) = a0 H0 + a1 H4 + a2 H8 + a3 H12
    """
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
    # H12 = x^12 - 132x^10 + 7920x^8 - 174720x^6 + 1663200x^4 - 4665600x^2 + 10395
    
    a3 = -(a0 + 3.0*a1 + 105.0*a2) / 945.0
    
    coeffs = np.zeros(13)
    
    # Add a0 * H0
    coeffs[0] += a0
    
    # Add a1 * H4
    coeffs[4] += a1
    coeffs[2] += -6.0 * a1
    coeffs[0] += 3.0 * a1
    
    # Add a2 * H8
    coeffs[8] += a2
    coeffs[6] += -28.0 * a2
    coeffs[4] += 420.0 * a2
    coeffs[2] += -1680.0 * a2
    coeffs[0] += 105.0 * a2
    
    # Add a3 * H12
    coeffs[12] += a3
    coeffs[10] += -132.0 * a3
    coeffs[8] += 7920.0 * a3
    coeffs[6] += -174720.0 * a3
    coeffs[4] += 1663200.0 * a3
    coeffs[2] += -4665600.0 * a3
    coeffs[0] += 10395.0 * a3
    
    return coeffs

def _integrate_sq(poly_c, weight_power=0):
    """
    Computes int (sum c_i x^i)^2 x^(weight_power) e^-x^2 dx
    """
    # Result is sum_{i,j} c_i c_j int x^(i+j+weight_power) e^-x^2 dx
    total = 0.0
    max_deg = len(poly_c) - 1
    for i in range(max_deg + 1):
        if poly_c[i] == 0: continue
        for j in range(max_deg + 1):
            if poly_c[j] == 0: continue
            power = i + j + weight_power
            if power % 2 != 0:
                continue
            if power >= len(MOMENTS):
                # Extend moments if needed? Max degree is 12+12=24.
                # MOMENTS size 25 covers 0..24.
                pass
            total += poly_c[i] * poly_c[j] * MOMENTS[power]
    return total

def _compute_bound(a0, a1, a2):
    """
    Compute the uncertainty bound for given coefficients.
    """
    coeffs = _poly_coeffs(a0, a1, a2)
    
    # m0 = int P^2 e^-x^2
    m0 = _integrate_sq(coeffs, 0)
    if m0 <= 0 or not np.isfinite(m0):
        return 1e18
        
    # m2 = int x^2 P^2 e^-x^2
    m2 = _integrate_sq(coeffs, 2)
    
    # Use precomputed Q basis
    q_coeffs = a0 * Q0 + a1 * Q1 + a2 * Q2
    
    # N2 = int Q^2 e^-x^2
    n2 = _integrate_sq(q_coeffs, 0)
    
    if n2 <= 0 or not np.isfinite(n2):
        return 1e18
        
    # A(f) = sqrt(m2/m0)
    # A(F) = sqrt(n2/m0)
    # Product = sqrt(m2*n2/m0^2)
    
    val = np.sqrt(m2 * n2) / m0
    return val

def _objective(params):
    a0, a1, a2 = params
    return _compute_bound(a0, a1, a2)

def construct():
    # Strategy:
    # 1. Multi-start with wider bounds and multiple DE runs
    # 2. L-BFGS-B polish with analytical gradients
    
    # Wider bounds to explore more of the space
    bounds = [(-2.0, 2.0), (-1.0, 1.0), (-0.5, 0.5)]
    
    best_val = 1e18
    best_params = np.array([1.0, 0.0, 0.0])
    
    # Run multiple DE with different seeds and bounds
    for seed in [42, 123, 456, 789, 1000]:
        try:
            if time.time() > _DEADLINE - 30:
                break
            result_de = differential_evolution(_objective, bounds, maxiter=200, popsize=25, seed=seed, polish=False, tol=1e-10)
            if result_de.fun < best_val:
                best_val = result_de.fun
                best_params = result_de.x
        except Exception:
            pass
    
    # Also try with tighter bounds around promising regions
    tight_bounds = [(-0.5, 0.5), (-0.2, 0.2), (-0.05, 0.05)]
    for seed in [42, 123]:
        try:
            if time.time() > _DEADLINE - 20:
                break
            result_de = differential_evolution(_objective, tight_bounds, maxiter=150, popsize=20, seed=seed, polish=False, tol=1e-12)
            if result_de.fun < best_val:
                best_val = result_de.fun
                best_params = result_de.x
        except Exception:
            pass
    
    # Polish with L-BFGS-B using finite difference gradients
    def obj_grad(params):
        val = _objective(params)
        h = 1e-7
        grad = np.zeros(3)
        for i in range(3):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            grad[i] = (_objective(p_plus) - _objective(p_minus)) / (2*h)
        return val, grad
        
    try:
        if time.time() < _DEADLINE - 10:
            res = minimize(obj_grad, best_params, method='L-BFGS-B', jac=True, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
    except Exception:
        pass
        
    # Final Nelder-Mead for fine-tuning
    try:
        if time.time() < _DEADLINE - 5:
            res = minimize(_objective, best_params, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
    except Exception:
        pass
        
    if not np.all(np.isfinite(best_params)):
        return np.array([1.0, 0.0, 0.0])
        
    return np.array(best_params)
# EVOLVE-BLOCK-END
