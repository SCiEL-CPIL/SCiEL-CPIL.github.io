import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _he_n_coeffs(n):
    """
    Returns coefficients for the probabilists' Hermite polynomial He_n(x).
    He_0 = 1
    He_1 = x
    He_2 = x^2 - 1
    He_3 = x^3 - 3x
    He_4 = x^4 - 6x^2 + 3
    ...
    We need up to n=12.
    Returns list of coefficients [c0, c1, ..., cn] such that sum(c_i * x^i).
    """
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 1.0]
    
    # Recurrence: He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)
    # We store coeffs in ascending order of power.
    prev = _he_n_coeffs(n - 1)
    prev2 = _he_n_coeffs(n - 2) if n >= 2 else [0.0]
    
    # x * He_{n-1}(x)
    term1 = [0.0] + prev
    # n * He_{n-2}(x)
    term2 = [n * c for c in prev2]
    
    # Align lengths
    max_len = max(len(term1), len(term2))
    term1 = term1 + [0.0] * (max_len - len(term1))
    term2 = term2 + [0.0] * (max_len - len(term2))
    
    res = [t1 - t2 for t1, t2 in zip(term1, term2)]
    return res

# Precompute coefficients for He_0, He_4, He_8, He_12
HE_0 = _he_n_coeffs(0)
HE_4 = _he_n_coeffs(4)
HE_8 = _he_n_coeffs(8)
HE_12 = _he_n_coeffs(12)

# Values at 0
HE_0_0 = HE_0[0]
HE_4_0 = HE_4[0]
HE_8_0 = HE_8[0]
HE_12_0 = HE_12[0]

def _poly_mul(p1, p2):
    """Multiply two polynomials given as coefficient lists (ascending order)."""
    res = [0.0] * (len(p1) + len(p2) - 1)
    for i, c1 in enumerate(p1):
        if c1 == 0: continue
        for j, c2 in enumerate(p2):
            if c2 == 0: continue
            res[i + j] += c1 * c2
    return res

def _poly_deriv(p):
    """Derivative of polynomial given as coefficient list (ascending order)."""
    if len(p) <= 1:
        return [0.0]
    return [i * c for i, c in enumerate(p[1:], 1)]

def _poly_add(p1, p2):
    """Add two polynomials."""
    max_len = max(len(p1), len(p2))
    p1 = p1 + [0.0] * (max_len - len(p1))
    p2 = p2 + [0.0] * (max_len - len(p2))
    return [a + b for a, b in zip(p1, p2)]

def _poly_sub(p1, p2):
    """Subtract two polynomials."""
    max_len = max(len(p1), len(p2))
    p1 = p1 + [0.0] * (max_len - len(p1))
    p2 = p2 + [0.0] * (max_len - len(p2))
    return [a - b for a, b in zip(p1, p2)]

def _poly_scale(p, s):
    """Scale polynomial by scalar s."""
    return [s * c for c in p]

def _int_gauss_moment(p, weight_exp=0.5):
    """
    Compute integral of p(x) * exp(-x^2 / (2*weight_exp)) dx?
    No, weight is exp(-x^2/2).
    So integral of p(x) exp(-x^2/2) dx.
    p(x) = sum c_k x^k.
    Integral is sum c_k * I_k.
    I_k = 0 if k odd.
    I_{2m} = sqrt(2*pi) * (2m-1)!! / 2^m.
    """
    if not p:
        return 0.0
    
    total = 0.0
    # Precompute moments up to max degree
    max_deg = len(p) - 1
    moments = [0.0] * (max_deg + 1)
    
    # I_0 = sqrt(2*pi)
    # I_2 = sqrt(2*pi) * 1/2
    # I_4 = sqrt(2*pi) * 3/4
    # I_{2m} = I_{2m-2} * (2m-1)/2
    
    sqrt_2pi = np.sqrt(2 * np.pi)
    moments[0] = sqrt_2pi
    if max_deg >= 2:
        moments[2] = moments[0] * 0.5
    for m in range(2, max_deg // 2 + 1):
        # 2m is the index
        idx = 2 * m
        if idx > max_deg: break
        moments[idx] = moments[idx - 2] * (2 * m - 1) / 2.0
        
    for k, c in enumerate(p):
        if k % 2 == 0 and c != 0:
            total += c * moments[k]
            
    return total

def _objective(c):
    c0, c4, c8 = c
    
    # Calculate c12
    # He_0(0)=1, He_4(0)=3, He_8(0)=105, He_12(0)=10395
    # c0*1 + c4*3 + c8*105 + c12*10395 = 0
    c12 = -(c0 * HE_0_0 + c4 * HE_4_0 + c8 * HE_8_0) / HE_12_0
    
    # Construct P(x) = c0 He0 + c4 He4 + c8 He8 + c12 He12
    # Scale and add
    P = _poly_add(_poly_add(_poly_add(
        _poly_scale(HE_0, c0),
        _poly_scale(HE_4, c4)
    ), _poly_scale(HE_8, c8)), _poly_scale(HE_12, c12))
    
    # I0 = int P^2 exp(-x^2/2)
    P_sq = _poly_mul(P, P)
    I0 = _int_gauss_moment(P_sq)
    
    if I0 <= 1e-12:
        return 1e10 # Avoid division by zero, large penalty
    
    # I2 = int x^2 P^2 exp(-x^2/2)
    # x^2 P^2 is P_sq shifted by 2
    x2_P_sq = [0.0, 0.0] + P_sq
    I2 = _int_gauss_moment(x2_P_sq)
    
    # I_deriv = int (P' - x P / 2)^2 exp(-x^2/2)
    P_prime = _poly_deriv(P)
    # x P / 2
    x_P_half = _poly_scale([0.0] + P, 0.5)
    # S = P' - x P / 2
    S = _poly_sub(P_prime, x_P_half)
    S_sq = _poly_mul(S, S)
    I_deriv = _int_gauss_moment(S_sq)
    
    # J = sqrt( (I2 / I0) * (I_deriv / I0) )
    # Minimize J^2 to avoid sqrt
    val = (I2 * I_deriv) / (I0 * I0)
    return np.sqrt(val)

def construct():
    start_time = time.time()
    deadline = start_time + 200.0 # 10s buffer
    
    # Initial guess from seed
    # Seed: [-1.13, 0.04, 0.000308]
    x0 = np.array([-1.13, 0.04, 0.000308])
    
    best_x = x0
    best_val = _objective(best_x)
    
    # Bounds for DE
    # Scale invariance means we can restrict range, but let's be safe.
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    # Differential Evolution
    try:
        res_de = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-8, 
            polish=False,
            init='sobol'
        )
        if res_de.fun < best_val:
            best_val = res_de.fun
            best_x = res_de.x
        
        if time.time() > deadline - 30:
            return best_x
    except Exception:
        pass
    
    # Local Refinement (Nelder-Mead)
    try:
        res_nm = minimize(
            _objective, 
            best_x, 
            method='Nelder-Mead', 
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000}
        )
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_x = res_nm.x
            
        if time.time() > deadline - 10:
            return best_x
    except Exception:
        pass
        
    # L-BFGS-B for final polish
    try:
        res_lbfgs = minimize(
            _objective, 
            best_x, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000}
        )
        if res_lbfgs.fun < best_val:
            best_val = res_lbfgs.fun
            best_x = res_lbfgs.x
    except Exception:
        pass
        
    return best_x

# EVOLVE-BLOCK-END
