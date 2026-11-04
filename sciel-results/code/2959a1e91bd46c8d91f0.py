"""Combined: efficient polynomial C4 + analytic gradient + alpha init + multi-start + high-precision polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre

_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(c):
    """Integral of poly (ascending) * e^{-x} dx = sum c[k]*k!."""
    n = min(len(c), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(c[:n], _FACTS[:n]))

def _build_p2_coeffs(roots):
    """Build P(x)^2 where P(x) = prod (x-xi)^2. Returns ascending coeffs of P^2 (degree 48)."""
    # First build P = prod (x-xi)^2
    p = np.array([1.0])
    for r in roots:
        # multiply by (x-r)^2 = x^2 - 2rx + r^2
        new = np.zeros(len(p) + 2)
        new[2:] += p
        new[1:-1] += -2.0 * r * p
        new[:len(p)] += r * r * p
        p = new
    # Now P^2 = P * P
    p2 = np.convolve(p, p)
    return p2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2/I1^2 where P=prod(x-xi)^2, I1=int P e^{-x}, I2=int P^2 e^{-x}."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    # Build P = prod (x-xi)^2
    p = np.array([1.0])
    for r in roots:
        new = np.zeros(len(p) + 2)
        new[2:] += p
        new[1:-1] += -2.0 * r * p
        new[:len(p)] += r * r * p
        p = new

    p2 = np.convolve(p, p)

    I1 = _poly_moment(p)
    I2 = _poly_moment(p2)

    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Analytic gradient:
    # P = prod_j (x-x_j)^2
    # dP/dx_i = P * (-2/(x-x_i)) ... but we need integral form
    # dP/dx_i = -2 * (x-x_i) * prod_{j!=i}(x-x_j)^2 = -2 * (x-x_i) * R_i
    # where R_i = prod_{j!=i}(x-x_j)^2
    # dI1/dx_i = -2 * int (x-x_i)*R_i * e^{-x} dx
    # P^2 = prod_j (x-x_j)^4
    # d(P^2)/dx_i = -4*(x-x_i) * prod_{j!=i}(x-x_j)^4 = -4*(x-x_i)*S_i
    # where S_i = prod_{j!=i}(x-x_j)^4
    # dI2/dx_i = -4 * int (x-x_i)*S_i * e^{-x} dx
    # dC4/dx_i = dI2/I1^2 - 2*I2*dI1/I1^3

    grad = np.zeros(n)
    I1_sq = I1 * I1
    I1_cu = I1_sq * I1

    for i in range(n):
        # Build R_i = prod_{j!=i}(x-x_j)^2
        Ri = np.array([1.0])
        for j in range(n):
            if j == i:
                continue
            rj = roots[j]
            new = np.zeros(len(Ri) + 2)
            new[2:] += Ri
            new[1:-1] += -2.0 * rj * Ri
            new[:len(Ri)] += rj * rj * Ri
            Ri = new

        # int (x-x_i)*R_i * e^{-x} dx
        # (x-x_i)*R_i: multiply Ri by (x - x_i)
        xRi = np.zeros(len(Ri) + 1)
        xRi[1:] += Ri          # x * Ri
        xRi[:len(Ri)] += -roots[i] * Ri  # -x_i * Ri
        dI1_i = -2.0 * _poly_moment(xRi)

        # Build S_i = prod_{j!=i}(x-x_j)^4
        # S_i = R_i^2
        Si = np.convolve(Ri, Ri)

        # int (x-x_i)*S_i * e^{-x} dx
        xSi = np.zeros(len(Si) + 1)
        xSi[1:] += Si
        xSi[:len(Si)] += -roots[i] * Si
        dI2_i = -4.0 * _poly_moment(xSi)

        grad[i] = dI2_i / I1_sq - 2.0 * I2 * dI1_i / I1_cu

    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _c4_from_alpha(alpha):
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 40.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: 1D search over alpha
    best_alpha = 0.0
    best_c4_alpha = np.inf

    # Coarse log grid
    log_alphas = np.linspace(math.log(1e-4), math.log(1e5), 200)
    for la in log_alphas:
        if not time_ok():
            break
        a = np.exp(la)
        c4_val = _c4_from_alpha(a)
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = a

    # Refine
    if time_ok():
        try:
            def _log_obj(la):
                a = np.exp(la)
                if a < 1e-5 or a > 1e6:
                    return np.inf
                return _c4_from_alpha(a)
            lo = math.log(max(best_alpha * 0.3, 1e-5))
            hi = math.log(min(best_alpha * 3.0, 1e6))
            res_a = minimize_scalar(_log_obj, bounds=(lo, hi), method='bounded',
                                     options={'xatol': 1e-12, 'maxiter': 800})
            if res_a.success and np.isfinite(res_a.fun) and res_a.fun < best_c4_alpha:
                best_alpha = np.exp(res_a.x)
                best_c4_alpha = res_a.fun
        except Exception:
            pass

    # Get initial roots
    xg, _ = roots_genlaguerre(12, best_alpha)
    best_roots = np.sort(xg)
    best_c4 = _compute_c4(best_roots)

    # Also try standard Laguerre
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Phase 2: L-BFGS-B refinement
    if time_ok():
        try:
            res = minimize(_objective_grad, best_roots, jac=True, method='L-BFGS-B',
                           bounds=[(1e-10, None)] * 12,
                           options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 50})
            if np.all(res.x > 0) and np.isfinite(res.fun):
                cand = np.sort(res.x)
                c4 = _compute_c4(cand)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = cand.copy()
        except Exception:
            pass

    # Phase 3: Multi-start with diverse initializations
    if time_ok():
        rng = np.random.default_rng(12345)
        # Various perturbation strategies
        strategies = []
        # Gaussian perturbation
        for scale in [0.05, 0.1, 0.2, 0.5]:
            pert = best_roots * (1.0 + scale * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-4)
            strategies.append(np.sort(pert))
        # Log-uniform random
        for _ in range(4):
            r = np.sort(np.exp(rng.uniform(math.log(0.1), math.log(100.0), 12)))
            strategies.append(r)
        # Shifted
        for shift in [0.5, 2.0, 0.2]:
            pert = np.sort(best_roots * shift)
            strategies.append(pert)

        for init in strategies:
            if not time_ok():
                break
            if not np.all(np.isfinite(init)) or not np.all(init > 0):
                continue
            try:
                res = minimize(_objective_grad, init, jac=True, method='L-BFGS-B',
                               bounds=[(1e-10, None)] * 12,
                               options={'maxiter': 1500, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 40})
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    cand = np.sort(res.x)
                    c4 = _compute_c4(cand)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = cand.copy()
            except Exception:
                pass

    # Phase 4: Try neighboring alpha basins
    if time_ok():
        for factor in [0.3, 0.5, 2.0, 3.0, 0.1, 5.0, 0.7, 1.4]:
            if not time_ok():
                break
            a_try = best_alpha * factor
            if a_try < 1e-5 or a_try > 1e6:
                continue
            try:
                xg2, _ = roots_genlaguerre(12, a_try)
                xg2 = np.sort(xg2)
                if not np.all(xg2 > 0):
                    continue
                res2 = minimize(_objective_grad, xg2, jac=True, method='L-BFGS-B',
                                bounds=[(1e-10, None)] * 12,
                                options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 40})
                if np.all(res2.x > 0) and np.isfinite(res2.fun):
                    cand = np.sort(res2.x)
                    c4 = _compute_c4(cand)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = cand.copy()
            except Exception:
                pass

    # Phase 5: Nelder-Mead polish (derivative-free, can find slightly better minima)
    if time_ok():
        try:
            res_nm = minimize(_objective, best_roots, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16,
                                       'adaptive': True})
            if np.all(res.x > 0) and np.isfinite(res_nm.fun):
                cand = np.sort(res_nm.x)
                c4 = _compute_c4(cand)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = cand.copy()
        except Exception:
            pass

    # Phase 6: Final ultra-high-precision L-BFGS-B
    for _ in range(3):
        if not time_ok():
            break
        try:
            res = minimize(_objective_grad, best_roots, jac=True, method='L-BFGS-B',
                           bounds=[(1e-12, None)] * 12,
                           options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-17, 'maxls': 30})
            if np.all(res.x > 0) and np.isfinite(res.fun):
                cand = np.sort(res.x)
                c4 = _compute_c4(cand)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = cand.copy()
                else:
                    break
        except Exception:
            break

    # Fallback
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
