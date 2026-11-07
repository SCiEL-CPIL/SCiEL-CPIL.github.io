"""Structural approach: Newton root-finding on double-root constraint in coefficient space."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

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


def _q_second(coeffs, y):
    """Second derivative q''(y)."""
    r = 0.0
    for i in range(len(coeffs) - 1, 1, -1):
        r = r * y + i * (i - 1) * coeffs[i]
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


def _newton_double_root(c0, c1, c2, y_target, max_iter=50, tol=1e-14):
    """
    Newton's method to find (c1, c2) such that q(y_target)=0 and q'(y_target)=0,
    holding c0 fixed. Returns (c1, c2, success).
    """
    c1_cur = c1
    c2_cur = c2
    for _ in range(max_iter):
        coeffs = _q_coefficients(c0, c1_cur, c2_cur)
        f1 = _q_eval(coeffs, y_target)
        f2 = _q_deriv(coeffs, y_target)
        if abs(f1) < tol and abs(f2) < tol:
            return c1_cur, c2_cur, True
        # Jacobian: dF/dc1, dF/dc2
        # F = [f1, f2], vars = [c1, c2]
        # df1/dc1 = b0_c1 + b1_c1*y + ... where b_k_c1 = dB_k/dC1
        # b0_c1 = -48, b1_c1 = 16, b2_c1 = 0, b3_c1 = 0, b4_c1 = 0, b5_c1 = 0
        # b0_c2 = -40320, b1_c2 = 26880, b2_c2 = -4608, b3_c2 = 256, b4_c2 = 0, b5_c2 = 0
        # But also c3 depends on c1, c2, so we need full chain rule.
        # c3 = -(c0 + 12*c1 + 168*c2)/18480
        # dc3/dc1 = -12/18480, dc3/dc2 = -168/18480
        
        dc3_dc1 = -12.0 / 18480.0
        dc3_dc2 = -168.0 / 18480.0
        
        # b_k = A_k*c1 + B_k*c2 + C_k*c3
        # b0 = -48*c1 - 40320*c2 - 5109120*c3
        # b1 = 16*c1 + 26880*c2 + 9072000*c3
        # b2 = 0*c1 - 4608*c2 - 6005760*c3
        # b3 = 0*c1 + 256*c2 + 1424640*c3
        # b4 = 0*c1 + 0*c2 - 142464*c3
        # b5 = 0*c1 + 0*c2 + 4096*c3
        
        # db0/dc1 = -48 + (-5109120)*dc3_dc1
        # db1/dc1 = 16 + 9072000*dc3_dc1
        # db2/dc1 = 0 + (-6005760)*dc3_dc1
        # db3/dc1 = 0 + 1424640*dc3_dc1
        # db4/dc1 = 0 + (-142464)*dc3_dc1
        # db5/dc1 = 0 + 4096*dc3_dc1
        
        # Similarly for dc2
        
        db0_dc1 = -48.0 + (-5109120.0) * dc3_dc1
        db1_dc1 = 16.0 + 9072000.0 * dc3_dc1
        db2_dc1 = 0.0 + (-6005760.0) * dc3_dc1
        db3_dc1 = 0.0 + 1424640.0 * dc3_dc1
        db4_dc1 = 0.0 + (-142464.0) * dc3_dc1
        db5_dc1 = 0.0 + 4096.0 * dc3_dc1
        
        db0_dc2 = -40320.0 + (-5109120.0) * dc3_dc2
        db1_dc2 = 26880.0 + 9072000.0 * dc3_dc2
        db2_dc2 = -4608.0 + (-6005760.0) * dc3_dc2
        db3_dc2 = 256.0 + 1424640.0 * dc3_dc2
        db4_dc2 = 0.0 + (-142464.0) * dc3_dc2
        db5_dc2 = 0.0 + 4096.0 * dc3_dc2
        
        y = y_target
        # df1/dc1 = db0_dc1 + db1_dc1*y + db2_dc1*y^2 + db3_dc1*y^3 + db4_dc1*y^4 + db5_dc1*y^5
        df1_dc1 = db0_dc1 + db1_dc1 * y + db2_dc1 * y**2 + db3_dc1 * y**3 + db4_dc1 * y**4 + db5_dc1 * y**5
        df1_dc2 = db0_dc2 + db1_dc2 * y + db2_dc2 * y**2 + db3_dc2 * y**3 + db4_dc2 * y**4 + db5_dc2 * y**5
        
        # df2/dc1 = derivative of q'(y) w.r.t. c1 = db0_dc1*0 + db1_dc1*1*y^0 + db2_dc1*2*y^1 + ...
        # q'(y) = b1 + 2*b2*y + 3*b3*y^2 + 4*b4*y^3 + 5*b5*y^4
        df2_dc1 = db1_dc1 + 2.0 * db2_dc1 * y + 3.0 * db3_dc1 * y**2 + 4.0 * db4_dc1 * y**3 + 5.0 * db5_dc1 * y**4
        df2_dc2 = db1_dc2 + 2.0 * db2_dc2 * y + 3.0 * db3_dc2 * y**2 + 4.0 * db4_dc2 * y**3 + 5.0 * db5_dc2 * y**4
        
        J = np.array([[df1_dc1, df1_dc2],
                      [df2_dc1, df2_dc2]])
        F = np.array([f1, f2])
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-30:
            return c1_cur, c2_cur, False
        J_inv = np.array([[J[1, 1], -J[0, 1]],
                          [-J[1, 0], J[0, 0]]]) / det
        delta = J_inv @ F
        # Damped Newton: step in direction -delta
        c1_cur -= delta[0]
        c2_cur -= delta[1]
        # Damping if step too large
        if abs(delta[0]) > 10.0 or abs(delta[1]) > 10.0:
            c1_cur += 0.5 * delta[0]
            c2_cur += 0.5 * delta[1]
    coeffs = _q_coefficients(c0, c1_cur, c2_cur)
    f1 = _q_eval(coeffs, y_target)
    f2 = _q_deriv(coeffs, y_target)
    return c1_cur, c2_cur, (abs(f1) < 1e-10 and abs(f2) < 1e-10)


def construct():
    """
    Search for coefficients [c0, c1, c2] minimizing value = r^2/(2*pi).
    Uses Newton's method on double-root constraint with bisection over target y*.
    """
    global _t_start
    _t_start = time.time()

    best_params = _SEED.copy()
    best_value = _seed_value

    # Strategy: for various c0 values, use Newton to find (c1, c2) giving double root at y_target.
    # Then verify it's a sign-change root and compute actual value.
    # Bisection: smaller y_target → smaller value (monotone in y_target for fixed c0).
    
    # First, try to improve from seed using direct Newton with various y_targets
    # The seed has some value; let's find what y_target corresponds to the seed's root.
    seed_coeffs = _q_coefficients(_SEED[0], _SEED[1], _SEED[2])
    seed_y = _find_largest_sign_change_root(seed_coeffs)
    
    # For a range of c0 values, try Newton from various starting points
    c0_values = []
    # Around seed c0
    for dc0 in np.linspace(-0.5, 0.5, 21):
        c0_values.append(_SEED[0] + dc0)
    # Broader range
    for c0 in np.linspace(-3.0, 3.0, 13):
        c0_values.append(c0)
    # Remove duplicates approximately
    c0_values = list(set(round(c, 6) for c in c0_values))
    c0_values.sort()
    
    # For each c0, try multiple y_targets and starting points
    # y_target range: from very small to moderate
    y_targets = np.logspace(-4, 1.5, 30)  # 1e-4 to ~31.6
    
    for c0 in c0_values:
        if _remaining() < 5:
            break
        for y_t in y_targets:
            if _remaining() < 3:
                break
            # Try Newton from several starting points
            for c1_init in [-0.5, 0.0, 0.5, _SEED[1]]:
                for c2_init in [-0.1, 0.0, 0.1, _SEED[2]]:
                    if _remaining() < 1:
                        break
                    c1_new, c2_new, success = _newton_double_root(
                        c0, c1_init, c2_init, y_t, max_iter=40, tol=1e-13)
                    if success:
                        val = _evaluate_value(c0, c1_new, c2_new)
                        if np.isfinite(val) and val < best_value:
                            best_value = val
                            best_params = np.array([c0, c1_new, c2_new])
    
    # Phase 2: Local refinement around best found point
    if _remaining() > 10 and best_value < np.inf:
        # Try small perturbations of best params and re-run Newton
        for dc0 in np.linspace(-0.01, 0.01, 5):
            for dc1 in np.linspace(-0.01, 0.01, 5):
                for dc2 in np.linspace(-0.01, 0.01, 5):
                    if _remaining() < 2:
                        break
                    c0_p = best_params[0] + dc0
                    c1_p = best_params[1] + dc1
                    c2_p = best_params[2] + dc2
                    # Try Newton with y_target slightly below current best
                    y_t = best_value * 2.0 * np.pi * 0.99  # slightly smaller
                    c1_new, c2_new, success = _newton_double_root(
                        c0_p, c1_p, c2_p, y_t, max_iter=50, tol=1e-14)
                    if success:
                        val = _evaluate_value(c0_p, c1_new, c2_new)
                        if np.isfinite(val) and val < best_value:
                            best_value = val
                            best_params = np.array([c0_p, c1_new, c2_new])
    
    # Phase 3: Final polish - try to push y even lower from best point
    if _remaining() > 5 and best_value < np.inf:
        y_current = best_value * 2.0 * np.pi
        # Try targets slightly below current
        for factor in [0.999, 0.998, 0.995, 0.99, 0.98]:
            y_t = y_current * factor
            c1_new, c2_new, success = _newton_double_root(
                best_params[0], best_params[1], best_params[2], y_t, max_iter=60, tol=1e-15)
            if success:
                val = _evaluate_value(best_params[0], c1_new, c2_new)
                if np.isfinite(val) and val < best_value:
                    best_value = val
                    best_params = np.array([best_params[0], c1_new, c2_new])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
