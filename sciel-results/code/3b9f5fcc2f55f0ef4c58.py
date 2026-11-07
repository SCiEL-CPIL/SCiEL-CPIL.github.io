"""Combined solver for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Precompute Hermite polynomial coefficients in terms of y = x^2
# H_n(x) for even n is a polynomial in y=x^2 of degree n/2
# We'll compute these once at module load time.

def _hermite_y_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial Hn(x) as polynomial in y=x^2.
    Returns list [c0, c1, ..., c_{n/2}] where Hn(x) = sum c_i * y^i."""
    if n % 2 != 0:
        raise ValueError("Only even n supported")
    m = n // 2
    # Build Hn in x, then extract even powers
    if n == 0:
        return [1.0]
    if n == 2:
        # H2(x) = 4x^2 - 2 => in y: 4y - 2
        return [-2.0, 4.0]
    
    # Recurrence: H_{k+1}(x) = 2x*H_k(x) - 2k*H_{k-1}(x)
    # We'll build up to n, keeping only even-power coefficients
    # H_k for even k is a polynomial in y of degree k/2
    # H_k for odd k is x * (polynomial in y of degree (k-1)/2)
    
    # Let's just compute the full x-polynomial and extract
    # H_0 = 1
    # H_1 = 2x
    # H_k = 2x*H_{k-1} - 2(k-1)*H_{k-2}
    
    # Represent H_k as list of coefficients [c0, c1, ..., ck] for x^0, x^1, ..., x^k
    def build_hermite(n):
        if n == 0:
            return [1.0]
        if n == 1:
            return [0.0, 2.0]
        prev = [1.0]  # H_0
        curr = [0.0, 2.0]  # H_1
        for k in range(2, n + 1):
            new = [0.0] * (k + 1)
            # 2x * H_{k-1}: shift curr by 1, multiply by 2
            for i in range(k):
                new[i + 1] += 2.0 * curr[i]
            # -2*(k-1) * H_{k-2}
            for i in range(k - 1):
                new[i] -= 2.0 * (k - 1) * prev[i]
            prev = curr
            curr = new
        return curr
    
    x_coeffs = build_hermite(n)
    # Extract even powers: y_coeffs[i] = x_coeffs[2*i]
    y_coeffs = [x_coeffs[2 * i] for i in range(m + 1)]
    return y_coeffs

# Precompute
_H0_y = _hermite_y_coeffs(0)   # [1]
_H4_y = _hermite_y_coeffs(4)   # degree 2 in y
_H8_y = _hermite_y_coeffs(8)   # degree 4 in y
_H12_y = _hermite_y_coeffs(12) # degree 6 in y

# Pad to degree 6
def _pad(coeffs, length=7):
    result = [0.0] * length
    for i in range(min(len(coeffs), length)):
        result[i] = coeffs[i]
    return result

_H0_y = _pad(_H0_y)
_H4_y = _pad(_H4_y)
_H8_y = _pad(_H8_y)
_H12_y = _pad(_H12_y)

# H_n(0) values
_H0_0 = _H0_y[0]   # 1
_H4_0 = _H4_y[0]   # 12
_H8_0 = _H8_y[0]   # 1008
_H12_0 = _H12_y[0] # 103950


def _compute_value_fast(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) for given coefficients using fast numpy."""
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    g0 = c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0
    c3 = -g0 / _H12_0
    
    # g_y = c0*H0_y + c1*H4_y + c2*H8_y + c3*H12_y
    g_y = (c0 * np.array(_H0_y) + c1 * np.array(_H4_y) + 
           c2 * np.array(_H8_y) + c3 * np.array(_H12_y))
    
    # q(y) = g_y(y) / y, but g_y[0] should be 0
    # q(y) = g_y[1] + g_y[2]*y + ... + g_y[6]*y^5
    q_coeffs = g_y[1:7]  # degree 5 in y
    
    # Check if all zero
    if np.all(np.abs(q_coeffs) < 1e-15):
        return float('inf')
    
    # Find roots of q(y)
    try:
        roots = np.roots(q_coeffs)
    except Exception:
        return float('inf')
    
    # Filter real roots with y >= 0
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            ry = r.real
            if ry >= -1e-10:
                real_roots.append(max(0.0, ry))
    
    if not real_roots:
        return float('inf')
    
    # For each real root, check sign change
    def q_eval(y):
        return np.sum(q_coeffs * y ** np.arange(len(q_coeffs)))
    
    eps = 1e-8
    sign_change_roots = []
    
    for y_r in real_roots:
        if y_r < 1e-10:
            q_at_0 = q_coeffs[0]
            q_at_eps = q_eval(eps)
            if q_at_0 * q_at_eps < 0:
                sign_change_roots.append(y_r)
        else:
            y_left = max(0.0, y_r - eps)
            y_right = y_r + eps
            q_left = q_eval(y_left)
            q_right = q_eval(y_right)
            if q_left * q_right < 0:
                sign_change_roots.append(y_r)
    
    if not sign_change_roots:
        return float('inf')
    
    max_y = max(sign_change_roots)
    r = np.sqrt(max_y)
    value = r ** 2 / (2.0 * np.pi)
    return value


def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 210s limit
    
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = seed.copy()
    best_value = _compute_value_fast(*seed)
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60.0:
        def objective(x):
            val = _compute_value_fast(x[0], x[1], x[2])
            if val == float('inf'):
                return 1e10
            return val
        
        # Bounds for c0, c1, c2
        bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.5, 0.5)]
        
        try:
            result = differential_evolution(
                objective, bounds,
                seed=42, maxiter=100, popsize=15,
                tol=1e-6, polish=False,
                init='sobol'
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 20.0:
        def objective_local(x):
            val = _compute_value_fast(x[0], x[1], x[2])
            if val == float('inf'):
                return 1e10
            return val
        
        try:
            result = minimize(
                objective_local, best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine local search with small steps
    if time.time() < deadline - 5.0:
        np.random.seed(123)
        step_size = 1e-5
        for _ in range(1000):
            if time.time() > deadline - 3.0:
                break
            noise = np.random.normal(0, step_size, 3)
            candidate = best_coeffs + noise
            val = _compute_value_fast(*candidate)
            if val < best_value:
                best_value = val
                best_coeffs = candidate.copy()
            else:
                step_size *= 0.98
                if step_size < 1e-8:
                    break
    
    return best_coeffs
# EVOLVE-BLOCK-END
