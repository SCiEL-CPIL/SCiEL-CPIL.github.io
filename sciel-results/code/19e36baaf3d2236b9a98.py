"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_poly_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n(x) as a list [a0, a1, ..., an]."""
    if n == 0:
        return [1]
    if n == 1:
        return [0, 2]
    # H_0 = 1, H_1 = 2x
    h_prev = [1]  # H_0
    h_curr = [0, 2]  # H_1
    for k in range(2, n + 1):
        # H_k = 2x * H_{k-1} - 2(k-1) H_{k-2}
        h_next = [0] * (k + 1)
        # 2x * H_{k-1}
        for i in range(len(h_curr)):
            h_next[i + 1] += 2 * h_curr[i]
        # -2(k-1) H_{k-2}
        for i in range(len(h_prev)):
            h_next[i] -= 2 * (k - 1) * h_prev[i]
        h_prev = h_curr
        h_curr = h_next
    return h_curr


def _poly_mul(a, b):
    """Multiply two polynomials given as coefficient lists (ascending order)."""
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def _poly_add(a, b):
    """Add two polynomials given as coefficient lists (ascending order)."""
    n = max(len(a), len(b))
    result = [0] * n
    for i in range(n):
        if i < len(a):
            result[i] += a[i]
        if i < len(b):
            result[i] += b[i]
    return result


def _poly_scalar_mul(a, c):
    """Multiply polynomial by scalar."""
    return [c * x for x in a]


def _poly_div_by_x2(coeffs):
    """Divide polynomial by x^2, assuming it divides exactly. Returns list of coefficients."""
    # coeffs[0] and coeffs[1] should be 0
    return coeffs[2:]


def _poly_eval(coeffs, x):
    """Evaluate polynomial at x."""
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * (x ** i)
    return result


def _poly_roots(coeffs):
    """Find real roots of polynomial given as coefficient list (ascending order)."""
    # Convert to numpy format (descending order)
    if len(coeffs) == 0:
        return []
    # Remove leading zeros
    c = list(coeffs)
    while len(c) > 1 and abs(c[-1]) < 1e-15:
        c = c[:-1]
    if len(c) == 1:
        return []  # Constant polynomial, no roots
    # np.roots expects descending order
    c_np = np.array(c[::-1], dtype=np.float64)
    roots = np.roots(c_np)
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(np.imag(r)) < 1e-10:
            real_roots.append(float(np.real(r)))
    return real_roots


def _compute_value(c0, c1, c2):
    """Compute the value for given coefficients c0, c1, c2."""
    H0 = _hermite_poly_coeffs(0)
    H4 = _hermite_poly_coeffs(4)
    H8 = _hermite_poly_coeffs(8)
    H12 = _hermite_poly_coeffs(12)
    
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    H0_0 = _poly_eval(H0, 0)
    H4_0 = _poly_eval(H4, 0)
    H8_0 = _poly_eval(H8, 0)
    H12_0 = _poly_eval(H12, 0)
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    g = _poly_add(_poly_add(_poly_scalar_mul(H0, c0), _poly_scalar_mul(H4, c1)),
                  _poly_add(_poly_scalar_mul(H8, c2), _poly_scalar_mul(H12, c3)))
    
    # Check if g tends to negative limit at infinity
    # Leading coefficient of g
    leading_coeff = g[-1] if len(g) > 0 else 0
    if leading_coeff < 0:
        g = _poly_scalar_mul(g, -1)
    
    # q = g / x^2
    if len(g) < 3:
        return np.inf
    q = _poly_div_by_x2(g)
    
    # Find largest real root of q at which q changes sign
    roots = _poly_roots(q)
    if not roots:
        return np.inf
    
    # Check sign change at each root
    best_r = None
    for r in sorted(roots):
        # Check sign change
        eps = 1e-8
        val_left = _poly_eval(q, r - eps)
        val_right = _poly_eval(q, r + eps)
        if val_left * val_right < 0:
            # Sign change at r
            if best_r is None or r > best_r:
                best_r = r
    
    if best_r is None:
        return np.inf
    
    return best_r ** 2 / (2 * np.pi)


def construct():
    """Return three coefficients [c0, c1, c2] that minimize the value."""
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s kill
    
    # Seed values
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_value = _compute_value(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    best_value = seed_value
    
    # Local optimization using Nelder-Mead
    def objective(x):
        return _compute_value(x[0], x[1], x[2])
    
    # Try optimization with different initial points
    initial_points = [seed]
    
    # Add some perturbed points
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbation = rng.normal(0, 0.1, size=3)
        initial_points.append(seed + perturbation)
    
    for x0 in initial_points:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # More thorough search if time permits
    if time.time() < deadline - 60:
        # Grid-like search around best
        for scale in [0.5, 0.2, 0.1]:
            if time.time() > deadline - 30:
                break
            for i in range(-3, 4):
                for j in range(-3, 4):
                    for k in range(-3, 4):
                        if time.time() > deadline - 30:
                            break
                        x0 = best_coeffs + scale * np.array([i, j, k])
                        val = _compute_value(x0[0], x0[1], x0[2])
                        if val < best_value:
                            best_value = val
                            best_coeffs = x0.copy()
    
    return best_coeffs
# EVOLVE-BLOCK-END
