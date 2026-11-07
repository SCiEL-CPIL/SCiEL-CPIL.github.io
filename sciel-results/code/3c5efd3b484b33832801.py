"""Combined approach: y=x^2 substitution + multi-start local optimization."""
import numpy as np
import time
from fractions import Fraction
from scipy.optimize import minimize

# Precompute Hermite polynomial coefficients exactly using Fractions
def _hermite_coeffs(n):
    """Return list of Fractions for H_n(x) = sum a_k x^k (k=0..n)."""
    if n == 0:
        return [Fraction(1)]
    if n == 1:
        return [Fraction(0), Fraction(2)]
    c1 = _hermite_coeffs(n - 1)
    c2 = _hermite_coeffs(n - 2)
    a = [Fraction(0)] + [Fraction(2) * v for v in c1]
    b = [Fraction(-2 * (n - 1)) * v for v in c2]
    m = max(len(a), len(b))
    res = [Fraction(0)] * m
    for i, v in enumerate(a):
        res[i] += v
    for i, v in enumerate(b):
        res[i] += v
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res

# Precompute once
_HC = {n: _hermite_coeffs(n) for n in (0, 4, 8, 12)}
_HF = {n: np.array([float(v) for v in c], dtype=np.float64) for n, c in _HC.items()}
_H0_0 = float(_HC[0][0])
_H4_0 = float(_HC[4][0])
_H8_0 = float(_HC[8][0])
_H12_0 = float(_HC[12][0])

def _build_q_y_coeffs(c0, c1, c2):
    """
    Build Q(y) = q(sqrt(y)) where q(x) = g(x)/x^2.
    Q(y) is degree 5 in y. Returns coefficients ascending: [Q_0, Q_1, ..., Q_5].
    If leading coefficient is negative, flip all signs.
    """
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    # g_coeffs[k] = coefficient of x^k in g(x)
    g_coeffs = c0 * _HF[0] + c1 * _HF[4] + c2 * _HF[8] + c3 * _HF[12]
    # q(x) = g(x)/x^2, so q(x) = sum_{k=2}^{12} g_k x^{k-2}
    # Since g is even (only even powers from Hermite polys of even degree),
    # q(x) = sum_{j=0}^{5} g_{2j+2} x^{2j}
    # Q(y) = sum_{j=0}^{5} g_{2j+2} y^j
    Q = np.array([g_coeffs[2], g_coeffs[4], g_coeffs[6], g_coeffs[8], g_coeffs[10], g_coeffs[12]], dtype=np.float64)
    # If leading coefficient (of g, i.e., x^12 term) is negative, flip
    if g_coeffs[12] < 0:
        Q = -Q
    return Q

def _compute_value(c0, c1, c2):
    """
    Compute r^2/(2*pi) where r is the largest real root of q(x) at which q changes sign.
    Since q(x) = Q(x^2), positive roots of q correspond to positive roots of Q in y=x^2.
    r^2 = y where y is the largest positive root of Q at which Q changes sign.
    """
    Q = _build_q_y_coeffs(c0, c1, c2)
    # Find roots of Q(y) = 0
    try:
        roots_y = np.roots(Q[::-1])  # np.roots expects descending order
    except Exception:
        return np.inf
    
    real_pos = []
    for r in roots_y:
        if abs(r.imag) <= 1e-10 * max(1.0, abs(r.real)):
            if r.real > 1e-12:
                real_pos.append(float(r.real))
    
    if not real_pos:
        return 1e10
    
    # For each candidate, check sign change
    def Qval(y):
        return np.polyval(Q[::-1], y)
    
    best_y = None
    for y in sorted(real_pos, reverse=True):
        h = 1e-7 * max(1.0, abs(y))
        left = Qval(y - h)
        right = Qval(y + h)
        if left * right < 0:
            best_y = y
            break
    
    if best_y is None:
        # Try with larger h
        for y in sorted(real_pos, reverse=True):
            for h_factor in [1e-6, 1e-5, 1e-4, 1e-3]:
                h = h_factor * max(1.0, abs(y))
                left = Qval(y - h)
                right = Qval(y + h)
                if left * right < 0:
                    best_y = y
                    break
            if best_y is not None:
                break
    
    if best_y is None:
        return 1e10
    
    return best_y / (2.0 * np.pi)

def _objective(params):
    """Objective for optimizer."""
    c0, c1, c2 = params
    val = _compute_value(c0, c1, c2)
    if np.isinf(val) or np.isnan(val) or val > 1e9:
        return 1e10
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s kill
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_value = _compute_value(*seed)
    best_params = seed.copy()
    best_value = seed_value
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    start_points = [seed.copy()]
    
    # Perturbations around seed
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
        for _ in range(3):
            perturbation = rng.normal(0, scale, size=3)
            start_points.append(seed + perturbation)
    
    # Some hand-crafted points exploring different regions
    hand_points = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, -0.1, 0.0],
        [0.0, 0.0, 0.01],
        [0.0, 0.0, -0.01],
        [0.5, 0.1, 0.01],
        [-0.5, 0.1, 0.01],
        [0.5, -0.1, 0.01],
        [-0.5, -0.1, 0.01],
        [1.0, 0.5, 0.05],
        [-1.0, 0.5, 0.05],
        [1.0, -0.5, 0.05],
        [-1.0, -0.5, 0.05],
        [0.2, 0.05, 0.005],
        [-0.2, 0.05, 0.005],
        [0.2, -0.05, 0.005],
        [-0.2, -0.05, 0.005],
        [2.0, 1.0, 0.1],
        [-2.0, 1.0, 0.1],
        [2.0, -1.0, 0.1],
        [-2.0, -1.0, 0.1],
    ]
    for hp in hand_points:
        start_points.append(np.array(hp, dtype=np.float64))
    
    methods = ['Nelder-Mead', 'Powell', 'COBYLA']
    
    for method in methods:
        for sp in start_points:
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            try:
                if method == 'Nelder-Mead':
                    result = minimize(_objective, sp, method=method,
                                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                elif method == 'Powell':
                    result = minimize(_objective, sp, method=method,
                                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                else:  # COBYLA
                    result = minimize(_objective, sp, method=method,
                                      options={'maxiter': 2000, 'rhobeg': 1e-6, 'tol': 1e-12})
                
                if np.isfinite(result.fun) and result.fun < best_value:
                    best_value = float(result.fun)
                    best_params = result.x.copy()
            except Exception:
                pass
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
    
    # Final verification
    final_value = _compute_value(*best_params)
    if not np.isfinite(final_value) or final_value > seed_value:
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    return np.array(best_params)
