"""Search for coefficients minimizing the uncertainty-inequality constant."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
from math import pi


def _hermite_coeffs(n):
    """Return list of float coefficients for H_n(x) = sum a_k x^k (ascending)."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    # H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
    prev = _hermite_coeffs(n - 1)
    prev2 = _hermite_coeffs(n - 2)
    res = [0.0] * (n + 1)
    for k in range(n - 1):
        res[k + 1] += 2.0 * prev[k]
    for k in range(n - 2):
        res[k] -= 2.0 * (n - 1) * prev2[k]
    return res


def _poly_mul(a, b):
    """Multiply two polynomials given as ascending coefficient lists."""
    n, m = len(a), len(b)
    out = [0.0] * (n + m - 1)
    for i in range(n):
        if a[i] == 0.0:
            continue
        for j in range(m):
            out[i + j] += a[i] * b[j]
    return out


def _poly_add(a, b):
    n = max(len(a), len(b))
    out = [0.0] * n
    for i in range(len(a)):
        out[i] += a[i]
    for i in range(len(b)):
        out[i] += b[i]
    return out


def _poly_sub(a, b):
    n = max(len(a), len(b))
    out = [0.0] * n
    for i in range(len(a)):
        out[i] += a[i]
    for i in range(len(b)):
        out[i] -= b[i]
    return out


def _poly_scale(a, s):
    return [v * s for v in a]


def _poly_div_x2(p):
    """Divide polynomial by x^2; returns None if not divisible."""
    if len(p) < 3 or abs(p[0]) > 1e-14 or abs(p[1]) > 1e-14:
        return None
    return p[2:]


def _poly_eval(p, x):
    """Horner evaluation."""
    r = 0.0
    for c in reversed(p):
        r = r * x + c
    return r


def _sign_change(p, a, b):
    fa = _poly_eval(p, a)
    fb = _poly_eval(p, b)
    return fa * fb < 0


def _find_largest_sign_change_root(p, x_max=200.0):
    """Find the largest real root of p at which it changes sign, using bisection."""
    if p is None or len(p) == 0:
        return None
    # Scan for sign changes
    n_pts = 4000
    xs = np.linspace(0.0, x_max, n_pts)
    vals = np.array([_poly_eval(p, x) for x in xs])
    # Find intervals where sign changes
    roots = []
    for i in range(n_pts - 1):
        if vals[i] == 0.0:
            roots.append(xs[i])
        elif vals[i] * vals[i + 1] < 0:
            # Bisection
            a, b = xs[i], xs[i + 1]
            fa = vals[i]
            for _ in range(80):
                m = 0.5 * (a + b)
                fm = _poly_eval(p, m)
                if fa * fm <= 0:
                    b = m
                else:
                    a = m
                    fa = fm
            roots.append(0.5 * (a + b))
    if not roots:
        return None
    return max(roots)


def _compute_value(c0, c1, c2):
    """Compute the objective value for given coefficients."""
    H0 = _hermite_coeffs(0)
    H4 = _hermite_coeffs(4)
    H8 = _hermite_coeffs(8)
    H12 = _hermite_coeffs(12)

    H0_0 = H0[0]
    H4_0 = H4[0]
    H8_0 = H8[0]
    H12_0 = H12[0]

    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0

    # g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    g = _poly_add(_poly_scale(H0, c0), _poly_scale(H4, c1))
    g = _poly_add(g, _poly_scale(H8, c2))
    g = _poly_add(g, _poly_scale(H12, c3))

    # Check sign at infinity: leading coefficient of g
    # H12 leading coeff is 2^12, so leading term of g is (c3 * 2^12) x^12
    leading = c3 * (2.0 ** 12)
    if leading < 0:
        g = _poly_scale(g, -1.0)

    # q = g / x^2
    q = _poly_div_x2(g)
    if q is None:
        return float('inf')

    r = _find_largest_sign_change_root(q)
    if r is None or r <= 0:
        return float('inf')

    return r * r / (2.0 * pi)


def _objective(params):
    c0, c1, c2 = params
    return _compute_value(c0, c1, c2)


def construct():
    t_start = time.time()
    deadline = t_start + 190.0  # safety margin before 210s kill

    # Seed
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_value(*best_params)

    # Multi-start Nelder-Mead
    rng = np.random.default_rng(42)

    # Generate diverse starting points
    starts = [best_params.copy()]
    # Add perturbations around seed
    for _ in range(30):
        delta = rng.normal(0, 0.5, size=3)
        starts.append(best_params + delta)
    # Add some wider starts
    for _ in range(20):
        delta = rng.normal(0, 2.0, size=3)
        starts.append(best_params + delta)
    # Add some structured starts
    for s0 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        for s1 in [-0.1, 0.0, 0.1, 0.2]:
            for s2 in [0.0, 0.01, 0.02]:
                starts.append(np.array([s0, s1, s2]))

    # Evaluate all starts
    start_vals = []
    for s in starts:
        v = _compute_value(*s)
        start_vals.append(v)
        if v < best_val:
            best_val = v
            best_params = s.copy()

    # Sort starts by value, take top 20 for local refinement
    order = np.argsort(start_vals)
    top_starts = [starts[i] for i in order[:20] if start_vals[i] < float('inf')]

    # Local Nelder-Mead refinement on top starts
    for i, s in enumerate(top_starts):
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(
                _objective,
                s,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # High-precision polish: more iterations, tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final check
    final_val = _compute_value(*best_params)
    if final_val < best_val:
        best_val = final_val

    return np.array(best_params)
# EVOLVE-BLOCK-END
