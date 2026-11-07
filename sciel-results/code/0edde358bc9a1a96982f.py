"""
Refined search for the uncertainty inequality constant C4.

Strategy:
  - Fast NumPy-based evaluation of the objective (largest sign-changing root of the
    degree-5 z-polynomial, value = r^2/(2*pi)).
  - Multi-start Nelder-Mead global search with structured perturbations.
  - High-precision polish passes with tighter tolerances.
  - A final coordinate-descent micro-refinement to squeeze out any remaining error.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def hermite_coeffs(n):
    """Return coefficients of H_n(x) as a list [c0, c1, ..., cn] where c_i is coeff of x^i."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]

    H_nm2 = [1.0]
    H_nm1 = [0.0, 2.0]

    for k in range(2, n + 1):
        term1 = [0.0] + [2.0 * c for c in H_nm1]
        term2 = [-2.0 * (k - 1) * c for c in H_nm2]

        max_len = max(len(term1), len(term2))
        while len(term1) < max_len:
            term1.append(0.0)
        while len(term2) < max_len:
            term2.append(0.0)

        H_n = [t1 + t2 for t1, t2 in zip(term1, term2)]

        H_nm2 = H_nm1
        H_nm1 = H_n

    return H_nm1


# Precompute Hermite polynomial coefficients
H0_coeffs = np.array(hermite_coeffs(0))
H4_coeffs = np.array(hermite_coeffs(4))
H8_coeffs = np.array(hermite_coeffs(8))
H12_coeffs = np.array(hermite_coeffs(12))

H0_0 = H0_coeffs[0]
H4_0 = H4_coeffs[0]
H8_0 = H8_coeffs[0]
H12_0 = H12_coeffs[0]


def get_q_coeffs(c0, c1, c2):
    """Return coefficients [q0, q1, ..., q10] of q(x) = g(x)/x^2."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0

    g_coeffs = np.zeros(13)
    g_coeffs[:len(H0_coeffs)] += c0 * H0_coeffs
    g_coeffs[:len(H4_coeffs)] += c1 * H4_coeffs
    g_coeffs[:len(H8_coeffs)] += c2 * H8_coeffs
    g_coeffs[:len(H12_coeffs)] += c3 * H12_coeffs

    if g_coeffs[12] < 0:
        g_coeffs = -g_coeffs

    q_coeffs = g_coeffs[2:]
    return q_coeffs


def evaluate_value(c0, c1, c2):
    """
    Compute the value for given coefficients c0, c1, c2.
    Returns the value (float) or None if invalid.
    """
    try:
        c0_f = float(c0)
        c1_f = float(c1)
        c2_f = float(c2)
    except Exception:
        return None

    q_coeffs = get_q_coeffs(c0_f, c1_f, c2_f)

    # Extract even-indexed coefficients for z-polynomial (z = x^2)
    z_coeffs = q_coeffs[0::2]  # degree 5 in z

    z_reversed = z_coeffs[::-1]

    try:
        z_roots = np.roots(z_reversed)
    except Exception:
        return None

    positive_real_roots = []
    for zr in z_roots:
        if abs(zr.imag) < 1e-10 and zr.real > 1e-12:
            positive_real_roots.append(zr.real)

    if not positive_real_roots:
        return None

    positive_real_roots.sort(reverse=True)

    def q_z_val(z):
        return np.polyval(z_coeffs[::-1], z)

    r_best = None
    for z_star in positive_real_roots:
        eps = 1e-8 * max(1.0, abs(z_star))
        qz_left = q_z_val(z_star - eps)
        qz_right = q_z_val(z_star + eps)

        if qz_left * qz_right < 0:
            r = np.sqrt(z_star)
            r_best = r
            break

    if r_best is None:
        return None

    value = (r_best ** 2) / (2 * np.pi)
    return value


def _coordinate_descent(c, tol=1e-14, max_iter=5000):
    """
    Simple coordinate-descent refinement: for each coordinate, try small steps
    and keep the best. This can squeeze out tiny improvements that Nelder-Mead
    might miss due to its simplex geometry.
    """
    best = np.array(c, dtype=float)
    best_val = evaluate_value(best[0], best[1], best[2])
    if best_val is None:
        return best

    step = 1e-3
    for _ in range(max_iter):
        improved = False
        for i in range(3):
            for direction in [-1, 1]:
                trial = best.copy()
                trial[i] += direction * step
                val = evaluate_value(trial[0], trial[1], trial[2])
                if val is not None and val < best_val - 1e-16:
                    best = trial
                    best_val = val
                    improved = True
        if not improved:
            step *= 0.5
            if step < tol:
                break
    return best


def construct():
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = evaluate_value(best_c[0], best_c[1], best_c[2])

    if best_val is None:
        return best_c

    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin: 210s limit, 15s buffer

    if not HAS_SCIPY:
        # Fallback: simple random local search
        rng = np.random.default_rng(42)
        scale = 0.05
        for _ in range(5000):
            if time.time() > deadline:
                break
            perturbation = rng.normal(0, scale, size=3)
            candidate = best_c + perturbation
            val = evaluate_value(candidate[0], candidate[1], candidate[2])
            if val is not None and val < best_val:
                best_c = candidate
                best_val = val
            else:
                scale *= 0.999
        return best_c

    def objective(x):
        val = evaluate_value(x[0], x[1], x[2])
        if val is None:
            return 1e10
        return val

    # Multi-start Nelder-Mead with diverse starting points
    rng = np.random.default_rng(42)
    starts = [best_c.copy()]

    # Structured perturbations at multiple scales
    for scale in [0.05, 0.02, 0.01, 0.005, 0.001]:
        for _ in range(4):
            perturbation = rng.normal(0, scale, size=3)
            starts.append(best_c + perturbation)

    # Larger jumps to escape local minima
    for _ in range(6):
        perturbation = rng.normal(0, 0.15, size=3)
        starts.append(best_c + perturbation)

    # Some asymmetric starts
    for delta0 in [-0.1, -0.05, 0.05]:
        for delta1 in [-0.02, 0.02]:
            starts.append(best_c + np.array([delta0, delta1, 0.0]))

    for x0 in starts:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True}
            )
            if time.time() > deadline - 10:
                break
            val_new = evaluate_value(res.x[0], res.x[1], res.x[2])
            if val_new is not None and val_new < best_val:
                best_c = res.x.copy()
                best_val = val_new
        except Exception:
            pass

    # High-precision polish pass
    if time.time() < deadline - 25:
        try:
            res_polish = minimize(
                objective, best_c, method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-13, 'fatol': 1e-15,
                         'adaptive': True}
            )
            val_polish = evaluate_value(res_polish.x[0], res_polish.x[1], res_polish.x[2])
            if val_polish is not None and val_polish < best_val:
                best_c = res_polish.x.copy()
                best_val = val_polish
        except Exception:
            pass

    # Final coordinate-descent micro-refinement
    if time.time() < deadline - 10:
        try:
            best_c = _coordinate_descent(best_c, tol=1e-15, max_iter=8000)
            val_final = evaluate_value(best_c[0], best_c[1], best_c[2])
            if val_final is not None and val_final < best_val:
                best_val = val_final
        except Exception:
            pass

    # One more Nelder-Mead polish if time remains
    if time.time() < deadline - 5:
        try:
            res_last = minimize(
                objective, best_c, method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16,
                         'adaptive': True}
            )
            val_last = evaluate_value(res_last.x[0], res_last.x[1], res_last.x[2])
            if val_last is not None and val_last < best_val:
                best_c = res_last.x.copy()
                best_val = val_last
        except Exception:
            pass

    return best_c
# EVOLVE-BLOCK-END
