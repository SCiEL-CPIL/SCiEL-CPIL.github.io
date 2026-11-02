import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

def _time_left():
    return _DEADLINE - time.time()

# Precompute Hermite polynomial coefficients [c_0, c_1, ..., c_n] for H_n(x) = sum c_k x^k
H0 = np.array([1.0])
H4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
H8 = np.array([10080.0, 0.0, -34560.0, 0.0, 23040.0, 0.0, -4608.0, 0.0, 256.0])
H12 = np.array([10321920.0, 0.0, -34406400.0, 0.0, 34406400.0, 0.0, -12902400.0, 0.0, 2090112.0, 0.0, -147456.0, 0.0, 4096.0])

# H_n(0) values
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 3024.0
H12_0 = 10321920.0

def _poly_mul(p, q):
    """Multiply two polynomials represented as coefficient arrays [c0, c1, ...]"""
    if len(p) == 0 or len(q) == 0:
        return np.array([0.0])
    return np.convolve(p, q)

def _poly_deriv(p):
    """Derivative of polynomial [c0, c1, ...] -> [c1, 2c2, ...]"""
    if len(p) <= 1:
        return np.array([0.0])
    return np.arange(1, len(p)) * p[1:]

def _poly_mul_by_x(p):
    """Multiply polynomial by x: [0, c0, c1, ...]"""
    return np.concatenate(([0.0], p))

def _integrate_poly_gaussian(p):
    """
    Compute integral of P(x) * exp(-x^2) dx from -inf to inf.
    """
    result = 0.0
    sqrt_pi = np.sqrt(np.pi)
    for k, c in enumerate(p):
        if c == 0.0:
            continue
        if k % 2 != 0:
            continue
        m = k // 2
        if m == 0:
            i_k = sqrt_pi
        else:
            double_fact = 1.0
            for i in range(1, 2*m, 2):
                double_fact *= i
            i_k = double_fact * sqrt_pi / (2.0 ** m)
        result += c * i_k
    return result

def _uncertainty_product(a0, a4, a8):
    """
    Compute sigma_x * sigma_p for g(x) = P(x) exp(-x^2/2).
    """
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    max_len = max(len(H0), len(H4), len(H8), len(H12))
    P = np.zeros(max_len)
    P[:len(H0)] += a0 * H0
    P[:len(H4)] += a4 * H4
    P[:len(H8)] += a8 * H8
    P[:len(H12)] += a12 * H12
    
    if np.allclose(P, 0.0):
        return 1e10
        
    P_sq = _poly_mul(P, P)
    x2_P_sq = _poly_mul_by_x(_poly_mul_by_x(P_sq))
    
    num_x = _integrate_poly_gaussian(x2_P_sq)
    den = _integrate_poly_gaussian(P_sq)
    
    if den <= 1e-10:
        return 1e10
        
    sigma_x_sq = num_x / den
    
    P_deriv = _poly_deriv(P)
    x_P = _poly_mul_by_x(P)
    
    max_len_q = max(len(P_deriv), len(x_P))
    Q = np.zeros(max_len_q)
    Q[:len(P_deriv)] += P_deriv
    Q[:len(x_P)] -= x_P
    
    Q_sq = _poly_mul(Q, Q)
    num_p = _integrate_poly_gaussian(Q_sq)
    
    sigma_p_sq = num_p / den
    
    if sigma_x_sq < 0 or sigma_p_sq < 0:
        return 1e10
        
    return np.sqrt(sigma_x_sq * sigma_p_sq)

def _normalize_and_eval(a0, a4, a8):
    """Normalize to unit norm and evaluate, to handle scale invariance."""
    norm = np.sqrt(a0**2 + a4**2 + a8**2)
    if norm < 1e-15:
        return 1e10
    return _uncertainty_product(a0/norm, a4/norm, a8/norm)

def construct():
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_result = seed_result.copy()
    best_score = _uncertainty_product(*seed_result)
    
    # Normalize seed
    seed_norm = np.linalg.norm(seed_result)
    seed_normalized = seed_result / seed_norm
    
    # 1. Multiple random restarts with local optimization
    rng = np.random.default_rng(42)
    n_restarts = 20
    
    for i in range(n_restarts):
        if _time_left() < 15:
            break
        
        if i == 0:
            start_point = seed_normalized.copy()
        else:
            # Random initialization on the unit sphere
            start_point = rng.normal(size=3)
            start_point /= np.linalg.norm(start_point)
        
        try:
            res = minimize(
                _uncertainty_product,
                start_point,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # 2. Global search with Differential Evolution
    if _time_left() > 30:
        try:
            # Use normalized bounds for global search
            bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
            
            de_result = differential_evolution(
                _uncertainty_product,
                bounds,
                seed=42,
                maxiter=300,
                tol=1e-10,
                polish=False,
                init='sobol',
                popsize=30
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
    
    # 3. Additional local refinement from best found
    if _time_left() > 10:
        try:
            res = minimize(
                _uncertainty_product,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # 4. Final high-precision polish
    if _time_left() > 5:
        try:
            res = minimize(
                _uncertainty_product,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result
