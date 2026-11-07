"""Structural approach: Newton root-finding on double-root constraint in coefficient space."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

_DEADLINE = 175.0
_t_start = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _t_start)


def _q_coefficients(c0, c1, c2):
    """Return [b0..b5] where q(y) = b0 + b1*y + ... + b5*y^5, y=x^2."""
    c3 = -(c0 + 12.0 * c1 + 168.0 * c2) / 18480.0
    b0 = -48.0 * c1 - 40320.0 * c2 - 5109120.0 * c3
    b1 = 16.0 * c1 + 26880.0 * c2 + 9072000.0 * c3
    b2 = -4608.0 * c2 - 6005760.0 * c3
    b3 = 256.0 * c2 + 1424640.0 * c3
    b4 = -142464.0 * c3
    b5 = 4096.0 * c3
    return np.array([b0, b1, b2, b3, b4, b5])


def _q_eval(coeffs, y):
    """Horner evaluation of q(y)."""
    r = 0.0
    for b in coeffs:
        r = r * y + b
    return r


def _q_deriv(coeffs, y):
    """Derivative q'(y) via Horner."""
    r = 0.0
    for i in range(len(coeffs) - 1, 0, -1):
        r = r * y + i * coeffs[i]
    return r


def _find_largest_sign_change_root(coeffs):
    """Find largest real positive root where q changes sign. Returns y or None."""
    poly_coeffs = np.array(coeffs[::-1])
    roots = np.roots(poly_coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9 and r.real > 1e-12:
            real_roots.append(r.real)
    if not real_roots:
        return None
    real_roots.sort(reverse=True)
    for y in real_roots:
        eps = max(1e-7, abs(y) * 1e-7)
        qb = _q_eval(coeffs, y - eps)
        qa = _q_eval(coeffs, y + eps)
        if qb * qa < 0:
            return y
    return None


def _evaluate_value(c0, c1, c2):
    """Compute value = r^2/(2*pi)."""
    coeffs = _q_coefficients(c0, c1, c2)
    y_root = _find_largest_sign_change_root(coeffs)
    if y_root is None:
        return np.inf
    return y_root / (2.0 * np.pi)


# Seed
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_seed_value = _evaluate_value(_SEED[0], _SEED[1], _SEED[2])


def _double_root_objective(x):
    """
    Objective: minimize y (the double root location) subject to q(y)=0 and q'(y)=0.
    x = [c0, c1, c2, log_y]
    Returns y + penalty if constraints not satisfied.
    """
    c0, c1, c2, log_y = x
    y = np.exp(log_y)
    coeffs = _q_coefficients(c0, c1, c2)
    f1 = _q_eval(coeffs, y)
    f2 = _q_deriv(coeffs, y)
    # Penalty for constraint violation
    penalty = 1e4 * (f1**2 + f2**2)
    return y + penalty


def _double_root_gradient(x):
    """
    Gradient of _double_root_objective.
    """
    c0, c1, c2, log_y = x
    y = np.exp(log_y)
    coeffs = _q_coefficients(c0, c1, c2)
    f1 = _q_eval(coeffs, y)
    f2 = _q_deriv(coeffs, y)
    
    dc3_dc0 = -1.0 / 18480.0
    dc3_dc1 = -12.0 / 18480.0
    dc3_dc2 = -168.0 / 18480.0
    
    # db_k/dc_j for each k, j
    # b0 = -48*c1 - 40320*c2 - 5109120*c3
    # b1 = 16*c1 + 26880*c2 + 9072000*c3
    # b2 = -4608*c2 - 6005760*c3
    # b3 = 256*c2 + 1424640*c3
    # b4 = -142464*c3
    # b5 = 4096*c3
    # db_k/dc0 = db_k/dc3 * dc3/dc0
    # db_k/dc1 = db_k/dc1_direct + db_k/dc3 * dc3/dc1
    # db_k/dc2 = db_k/dc2_direct + db_k/dc3 * dc3/dc2
    db_dc = np.zeros((6, 3))
    # dc0 column
    db_dc[0, 0] = -5109120.0 * dc3_dc0
    db_dc[1, 0] = 9072000.0 * dc3_dc0
    db_dc[2, 0] = -6005760.0 * dc3_dc0
    db_dc[3, 0] = 1424640.0 * dc3_dc0
    db_dc[4, 0] = -142464.0 * dc3_dc0
    db_dc[5, 0] = 4096.0 * dc3_dc0
    # dc1 column
    db_dc[0, 1] = -48.0 + (-5109120.0) * dc3_dc1
    db_dc[1, 1] = 16.0 + 9072000.0 * dc3_dc1
    db_dc[2, 1] = -6005760.0 * dc3_dc1
    db_dc[3, 1] = 1424640.0 * dc3_dc1
    db_dc[4, 1] = -142464.0 * dc3_dc1
    db_dc[5, 1] = 4096.0 * dc3_dc1
    # dc2 column
    db_dc[0, 2] = -40320.0 + (-5109120.0) * dc3_dc2
    db_dc[1, 2] = 26880.0 + 9072000.0 * dc3_dc2
    db_dc[2, 2] = -4608.0 + (-6005760.0) * dc3_dc2
    db_dc[3, 2] = 256.0 + 1424640.0 * dc3_dc2
    db_dc[4, 2] = -142464.0 * dc3_dc2
    db_dc[5, 2] = 4096.0 * dc3_dc2
    
    # df1/dc_j = sum_k db_k/dc_j * y^k
    # df2/dc_j = sum_k k * db_k/dc_j * y^(k-1)
    grad = np.zeros(4)
    for j in range(3):
        df1 = 0.0
        df2 = 0.0
        for k in range(6):
            df1 += db_dc[k, j] * (y ** k)
            if k > 0:
                df2 += k * db_dc[k, j] * (y ** (k - 1))
        # d(y+penalty)/dc_j = 1e4 * (2*f1*df1 + 2*f2*df2)
        grad[j] = 1e4 * (2.0 * f1 * df1 + 2.0 * f2 * df2)
    
    # d/d(log_y) = y + 1e4 * (2*f1 * df1/dlog_y + 2*f2 * df2/dlog_y)
    # df1/dlog_y = df1/dy * y = sum_k k * b_k * y^k
    # df2/dlog_y = df2/dy * y
    df1_dlogy = 0.0
    df2_dlogy = 0.0
    for k in range(6):
        if k > 0:
            df1_dlogy += k * coeffs[k] * (y ** k)
        if k > 1:
            df2_dlogy += k * (k - 1) * coeffs[k] * (y ** (k - 1))
    grad[3] = y + 1e4 * (2.0 * f1 * df1_dlogy + 2.0 * f2 * df2_dlogy)
    
    return grad


def _verify_sign_change(c0, c1, c2, y_target, tol=1e-8):
    """Check that y_target is a sign-change root of q."""
    coeffs = _q_coefficients(c0, c1, c2)
    eps = max(1e-6, abs(y_target) * 1e-6)
    qb = _q_eval(coeffs, y_target - eps)
    qa = _q_eval(coeffs, y_target + eps)
    return qb * qa < 0


def construct():
    """
    Search for coefficients [c0, c1, c2] minimizing value = r^2/(2*pi).
    Uses scipy.optimize.minimize on a smooth double-root objective.
    """
    global _t_start
    _t_start = time.time()

    best_params = _SEED.copy()
    best_value = _seed_value

    # Find the seed's y root
    seed_coeffs = _q_coefficients(_SEED[0], _SEED[1], _SEED[2])
    seed_y = _find_largest_sign_change_root(seed_coeffs)
    if seed_y is None:
        seed_y = 1.0
    seed_log_y = np.log(seed_y)

    # Phase 1: Multi-start scipy minimize on double-root objective
    # x = [c0, c1, c2, log_y]
    initial_points = [
        [_SEED[0], _SEED[1], _SEED[2], seed_log_y],
    ]
    # Add diverse starting points
    for c0_off in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        for c1_off in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            for c2_off in [-0.01, -0.005, 0.0, 0.005, 0.01]:
                c0_init = _SEED[0] + c0_off
                c1_init = _SEED[1] + c1_off
                c2_init = _SEED[2] + c2_off
                initial_points.append([c0_init, c1_init, c2_init, seed_log_y])
    
    # Deduplicate
    initial_points = list(set(tuple(p) for p in initial_points))
    
    for x0 in initial_points:
        if _remaining() < 10:
            break
        try:
            res = minimize(
                _double_root_objective,
                np.array(x0),
                jac=_double_root_gradient,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if _remaining() < 5:
                break
            c0, c1, c2, log_y = res.x
            y = np.exp(log_y)
            # Verify sign change
            if _verify_sign_change(c0, c1, c2, y):
                val = _evaluate_value(c0, c1, c2)
                if np.isfinite(val) and val < best_value:
                    best_value = val
                    best_params = np.array([c0, c1, c2])
        except Exception:
            pass

    # Phase 2: Local refinement with tighter tolerance
    if _remaining() > 10 and best_value < np.inf:
        y_current = best_value * 2.0 * np.pi
        log_y_current = np.log(y_current)
        x0 = np.array([best_params[0], best_params[1], best_params[2], log_y_current])
        try:
            res = minimize(
                _double_root_objective,
                x0,
                jac=_double_root_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            c0, c1, c2, log_y = res.x
            y = np.exp(log_y)
            if _verify_sign_change(c0, c1, c2, y):
                val = _evaluate_value(c0, c1, c2)
                if np.isfinite(val) and val < best_value:
                    best_value = val
                    best_params = np.array([c0, c1, c2])
        except Exception:
            pass

    # Phase 3: Try to push y even lower from best point
    if _remaining() > 5 and best_value < np.inf:
        y_current = best_value * 2.0 * np.pi
        for factor in [0.999, 0.998, 0.995, 0.99, 0.98, 0.95]:
            y_t = y_current * factor
            log_y_t = np.log(y_t)
            x0 = np.array([best_params[0], best_params[1], best_params[2], log_y_t])
            try:
                res = minimize(
                    _double_root_objective,
                    x0,
                    jac=_double_root_gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-16}
                )
                c0, c1, c2, log_y = res.x
                y = np.exp(log_y)
                if _verify_sign_change(c0, c1, c2, y):
                    val = _evaluate_value(c0, c1, c2)
                    if np.isfinite(val) and val < best_value:
                        best_value = val
                        best_params = np.array([c0, c1, c2])
            except Exception:
                pass
            if _remaining() < 3:
                break

    return np.array(best_params)
# EVOLVE-BLOCK-END
