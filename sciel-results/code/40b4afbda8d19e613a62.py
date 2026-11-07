"""Search for better coefficients via exact polynomial root finding and Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 180.0
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

# Coefficients for physicists' Hermite polynomials H0, H4, H8, H12 as polynomials in x
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1008
# H12(x) = 4096x^12 - 98304x^10 + 1128960x^8 - 5529600x^6 + 10321920x^4 - 7526400x^2 + 10395

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _compute_q_coeffs(c0, c1, c2):
    """Compute coefficients of q(y) = g(x)/x^2 where y = x^2.
    q(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    Returns array [a5, a4, a3, a2, a1, a0] (highest degree first for numpy.roots).
    """
    H12_0 = 10395.0
    c3 = -(c0 + 12.0 * c1 + 1008.0 * c2) / H12_0

    # q(y) coefficients from g(x)/x^2 with y = x^2
    # a5 = c3 * 4096
    # a4 = c3 * (-98304)
    # a3 = c3 * 1128960 + c2 * 256
    # a2 = c3 * (-5529600) + c2 * (-3072)
    # a1 = c3 * 10321920 + c2 * 11520 + c1 * 16
    # a0 = c3 * (-7526400) + c2 * (-13440) + c1 * (-48)

    a5 = c3 * 4096.0
    a4 = c3 * (-98304.0)
    a3 = c3 * 1128960.0 + c2 * 256.0
    a2 = c3 * (-5529600.0) + c2 * (-3072.0)
    a1 = c3 * 10321920.0 + c2 * 11520.0 + c1 * 16.0
    a0 = c3 * (-7526400.0) + c2 * (-13440.0) + c1 * (-48.0)

    return np.array([a5, a4, a3, a2, a1, a0])


def _compute_value(c0, c1, c2):
    """Compute the task value for given coefficients. Returns None if invalid."""
    q_coeffs = _compute_q_coeffs(c0, c1, c2)
    
    # Find roots of q(y)
    try:
        roots = np.roots(q_coeffs)
    except Exception:
        return None

    # Filter for positive real roots
    pos_real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 1e-12:
            pos_real_roots.append(r.real)

    if not pos_real_roots:
        # No positive real roots - check if q is always positive or always negative for y>0
        # Evaluate q at a few points
        y_test = np.array([0.1, 1.0, 10.0, 100.0])
        q_vals = np.polyval(q_coeffs, y_test)
        if np.all(q_vals > 0) or np.all(q_vals < 0):
            return 0.0
        else:
            # There's a sign change but no detected root - numerical issue
            return None

    # For each positive real root, check if q changes sign
    # A root y0 is a sign-change root if q has odd multiplicity at y0
    # We can check by evaluating q on both sides of y0
    sign_change_roots = []
    eps = 1e-8
    for y0 in pos_real_roots:
        y_left = max(y0 - eps, 1e-12)
        y_right = y0 + eps
        q_left = np.polyval(q_coeffs, y_left)
        q_right = np.polyval(q_coeffs, y_right)
        if q_left * q_right < 0:
            sign_change_roots.append(y0)

    if not sign_change_roots:
        return 0.0

    r_squared = max(sign_change_roots)
    return r_squared / (2.0 * np.pi)


def _objective(c):
    """Objective: value + penalty for issues."""
    c0, c1, c2 = c
    val = _compute_value(c0, c1, c2)

    if val is None:
        # Penalty for invalid
        return 10.0

    return val


def construct():
    best_c = _SEED.copy()
    best_val = _compute_value(*_SEED)
    if best_val is None:
        best_val = float('inf')

    # Phase 1: Nelder-Mead search from seed
    try:
        res = minimize(
            _objective,
            x0=_SEED,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if _time_left() < 10.0:
            return best_c

        c_new = res.x
        val_new = _compute_value(*c_new)
        if val_new is not None and val_new < best_val:
            best_c = c_new
            best_val = val_new
    except Exception:
        pass

    # Phase 2: Local refinement with tighter tolerance
    if _time_left() > 20.0:
        try:
            res2 = minimize(
                _objective,
                x0=best_c,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if _time_left() < 5.0:
                return best_c

            c_new2 = res2.x
            val_new2 = _compute_value(*c_new2)
            if val_new2 is not None and val_new2 < best_val:
                best_c = c_new2
                best_val = val_new2
        except Exception:
            pass

    # Phase 3: Multi-start with diverse initial points
    if _time_left() > 30.0:
        rng = np.random.default_rng(42)
        # Generate diverse starting points
        starts = [best_c.copy()]
        for _ in range(10):
            if _time_left() < 15.0:
                break
            # Random perturbation with varying scales
            scale = rng.choice([0.001, 0.01, 0.1, 0.5])
            c_start = best_c + rng.normal(0, scale, size=3)
            starts.append(c_start)

        for c_start in starts[1:]:
            if _time_left() < 10.0:
                break
            try:
                res3 = minimize(
                    _objective,
                    x0=c_start,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                c_new3 = res3.x
                val_new3 = _compute_value(*c_new3)
                if val_new3 is not None and val_new3 < best_val:
                    best_c = c_new3
                    best_val = val_new3
            except Exception:
                pass

    return np.array(best_c)
# EVOLVE-BLOCK-END
