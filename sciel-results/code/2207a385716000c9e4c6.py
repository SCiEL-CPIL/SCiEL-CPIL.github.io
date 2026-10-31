"""Analytic Hermite-Gaussian uncertainty bound with generalized eigenvalue + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh

_DEADLINE = time.time() + 195  # 15s safety margin before the 210s kill


def _bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty product and its gradient with respect to (a0, a4, a8).
    a12 is determined by the constraint g(0) = 0: a12 = -(a0 + 3*a4 + 105*a8)/945.
    Key insight: vF = n2/n0 = 7/3 (constant), so bound = sqrt(7/3 * m2/m0).
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    g2 = 2.0 * a0 + 210.0 * a8
    sqrt_pi = np.sqrt(np.pi)

    m0_inner = a0 * a0 + 3.0 * a4 * a4 + 105.0 * a8 * a8 + 945.0 * a12 * a12
    m0 = sqrt_pi * m0_inner

    m2_inner = (a0 * (3.0 * a0 + 12.0 * a4)
                + a4 * (12.0 * a0 + 105.0 * a4 + 840.0 * a8)
                + a8 * (840.0 * a4 + 2940.0 * a8)
                + 11340.0 * a8 * a12)
    m2 = sqrt_pi * m2_inner

    n0 = 18.0 * np.pi * g2 * g2
    n2 = 42.0 * np.pi * g2 * g2

    if m0 <= 0.0 or n0 <= 0.0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([0.0, 0.0, 0.0])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0.0 or vF <= 0.0:
        return 1e18, np.array([0.0, 0.0, 0.0])

    bound = np.sqrt(vf * vF)

    # Gradient
    dm0_da0 = sqrt_pi * (2.0 * a0 + 2.0 * 945.0 * a12 * (-1.0 / 945.0))
    dm0_da4 = sqrt_pi * (6.0 * a4 + 2.0 * 945.0 * a12 * (-3.0 / 945.0))
    dm0_da8 = sqrt_pi * (210.0 * a8 + 2.0 * 945.0 * a12 * (-105.0 / 945.0))

    dm2_inner_da0 = 6.0 * a0 + 24.0 * a4
    dm2_inner_da4 = 24.0 * a0 + 210.0 * a4 + 1680.0 * a8
    dm2_inner_da8 = 1680.0 * a4 + 4620.0 * a8 + 11340.0 * a12

    dm2_da0 = sqrt_pi * dm2_inner_da0
    dm2_da4 = sqrt_pi * dm2_inner_da4
    dm2_da8 = sqrt_pi * dm2_inner_da8

    dn0_da0 = 72.0 * np.pi * g2
    dn0_da4 = 0.0
    dn0_da8 = 7560.0 * np.pi * g2
    dn2_da0 = 168.0 * np.pi * g2
    dn2_da4 = 0.0
    dn2_da8 = 17640.0 * np.pi * g2

    dvf_da0 = (dm2_da0 * m0 - m2 * dm0_da0) / (m0 * m0)
    dvf_da4 = (dm2_da4 * m0 - m2 * dm0_da4) / (m0 * m0)
    dvf_da8 = (dm2_da8 * m0 - m2 * dm0_da8) / (m0 * m0)
    dvF_da0 = (dn2_da0 * n0 - n2 * dn0_da0) / (n0 * n0)
    dvF_da4 = (dn2_da4 * n0 - n2 * dn0_da4) / (n0 * n0)
    dvF_da8 = (dn2_da8 * n0 - n2 * dn0_da8) / (n0 * n0)

    grad_da0 = 0.5 / bound * (vF * dvf_da0 + vf * dvF_da0)
    grad_da4 = 0.5 / bound * (vF * dvf_da4 + vf * dvF_da4)
    grad_da8 = 0.5 / bound * (vF * dvf_da8 + vf * dvF_da8)

    return bound, np.array([grad_da0, grad_da4, grad_da8])


def _objective(params):
    val, _ = _bound_and_grad(params[0], params[1], params[2])
    return val if np.isfinite(val) else 1e18


def _objective_grad(params):
    val, grad = _bound_and_grad(params[0], params[1], params[2])
    if not np.isfinite(val):
        return 1e18, np.array([0.0, 0.0, 0.0])
    return val, grad


def construct():
    """
    Key insight: vF = n2/n0 = 7/3 is constant (given g(0)=0 constraint).
    So bound = sqrt(7/3 * m2/m0). Minimizing m2/m0 is a Rayleigh quotient problem.
    Set up symmetric matrices A, B such that m0_inner = x^T A x, m2_inner = x^T B x.
    Solve B v = lambda A v; smallest eigenvalue gives the minimum ratio.
    """
    # Build symmetric matrices A and B
    # m0_inner = a0^2 + 3*a4^2 + 105*a8^2 + 945*a12^2
    # where a12 = -(a0+3*a4+105*a8)/945
    # 945*a12^2 = (a0+3*a4+105*a8)^2 / 945
    # Expanding: (a0^2 + 9*a4^2 + 11025*a8^2 + 6*a0*a4 + 210*a0*a8 + 630*a4*a8)/945

    A = np.array([
        [1.0 + 1.0/945.0,       3.0/945.0,       105.0/945.0],
        [3.0/945.0,              3.0 + 9.0/945.0, 315.0/945.0],
        [105.0/945.0,            315.0/945.0,     105.0 + 11025.0/945.0]
    ])

    # m2_inner = 3*a0^2 + 24*a0*a4 + 105*a4^2 + 1680*a4*a8 + 2940*a8^2 + 11340*a8*a12
    # 11340*a8*a12 = -12*a8*a0 - 36*a8*a4 - 1260*a8^2
    # m2_inner = 3*a0^2 + 24*a0*a4 + 105*a4^2 + 1644*a4*a8 + 1680*a8^2 - 12*a0*a8

    B = np.array([
        [3.0,     12.0,   -6.0],
        [12.0,    105.0,  822.0],
        [-6.0,    822.0,  1680.0]
    ])

    # Verify matrix construction with seed
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240.0])
    a0, a4, a8 = seed
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    m0_direct = a0*a0 + 3.0*a4*a4 + 105.0*a8*a8 + 945.0*a12*a12
    m2_direct = (a0*(3.0*a0+12.0*a4) + a4*(12.0*a0+105.0*a4+840.0*a8)
                 + a8*(840.0*a4+2940.0*a8) + 11340.0*a8*a12)
    m0_mat = seed @ A @ seed
    m2_mat = seed @ B @ seed

    # Solve generalized eigenvalue problem B v = lambda A v
    best = seed.copy()
    best_val, _ = _bound_and_grad(*seed)

    try:
        eigenvalues, eigenvectors = eigh(B, A)
        # Smallest eigenvalue gives minimum of x^T B x / x^T A x
        min_ratio = eigenvalues[0]
        v_min = eigenvectors[:, 0]

        # Check g2 != 0
        g2_min = 2.0 * v_min[0] + 210.0 * v_min[2]
        if abs(g2_min) > 1e-12:
            # Scale eigenvector to reasonable magnitude
            norm_v = np.linalg.norm(v_min)
            if norm_v > 1e-15:
                v_scaled = v_min / norm_v * 0.5
                val_scaled, _ = _bound_and_grad(*v_scaled)
                if np.isfinite(val_scaled) and val_scaled < best_val:
                    best_val = val_scaled
                    best = v_scaled.copy()
    except Exception:
        pass

    # Multi-start L-BFGS-B for robustness with log-magnitude parameterization
    rng = np.random.default_rng(42)

    # Transform to log-magnitude + sign parameterization
    # params = [sign0*log|a0|, sign4*log|a4|, sign8*log|a8|]
    # where sign is in {-1, 1} and log|a| is the log magnitude
    # This makes the search space unbounded and allows exploring orders of magnitude

    def _log_params_to_coeffs(lp):
        """Convert log-magnitude+sign params to coefficients."""
        coeffs = np.zeros(3)
        for i in range(3):
            sign = np.sign(lp[i]) if abs(lp[i]) > 1e-10 else 1.0
            # Use tanh to smoothly map sign
            s = np.tanh(lp[i])
            mag = np.exp(np.abs(lp[i]))
            coeffs[i] = s * mag
        return coeffs

    def _coeffs_to_log_params(c):
        """Convert coefficients to log-magnitude+sign params."""
        lp = np.zeros(3)
        for i in range(3):
            if abs(c[i]) < 1e-15:
                lp[i] = 0.0
            else:
                s = np.sign(c[i])
                lp[i] = s * np.log(abs(c[i]))
        return lp

    def _log_objective(lp):
        """Objective in log-parameter space."""
        coeffs = _log_params_to_coeffs(lp)
        val, _ = _bound_and_grad(coeffs[0], coeffs[1], coeffs[2])
        return val if np.isfinite(val) else 1e18

    def _log_objective_grad(lp):
        """Objective and gradient in log-parameter space (numerical gradient)."""
        coeffs = _log_params_to_coeffs(lp)
        val, grad = _bound_and_grad(coeffs[0], coeffs[1], coeffs[2])
        if not np.isfinite(val):
            return 1e18, np.array([0.0, 0.0, 0.0])

        # Compute gradient in log-space via chain rule
        lp_grad = np.zeros(3)
        for i in range(3):
            if abs(lp[i]) > 1e-10:
                s = np.tanh(lp[i])
                mag = np.exp(np.abs(lp[i]))
                # d(coeff)/d(lp) = s * mag * (tanh'(lp) * sign(lp) + tanh(lp) * sign(lp))
                # tanh'(x) = 1 - tanh(x)^2
                dcoeff_dlp = s * mag * ((1 - s*s) * np.sign(lp[i]) + s * np.sign(lp[i]))
                lp_grad[i] = grad[i] * dcoeff_dlp
            else:
                # At lp=0, coeff=0, gradient is tricky; use finite difference
                eps = 1e-7
                lp_plus = lp.copy()
                lp_plus[i] += eps
                lp_minus = lp.copy()
                lp_minus[i] -= eps
                val_plus = _log_objective(lp_plus)
                val_minus = _log_objective(lp_minus)
                lp_grad[i] = (val_plus - val_minus) / (2 * eps)

        return val, lp_grad

    # Start from best in log-space
    lp_best = _coeffs_to_log_params(best)

    # Multi-start in log-space
    starts = [lp_best.copy()]
    # Add seed in log-space
    starts.append(_coeffs_to_log_params(seed))

    # Add random starts in log-space
    for i in range(30):
        scale = 2.0 / (i + 1)
        starts.append(lp_best + rng.normal(0.0, scale, 3))

    # Also try direct coefficient space starts
    direct_starts = [best.copy(), seed.copy()]
    for i in range(15):
        scale = 0.5 / (i + 1)
        direct_starts.append(best + rng.normal(0.0, scale, 3))

    # Optimize in log-space
    for s in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_log_objective_grad, s, method='L-BFGS-B',
                           options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                coeffs = _log_params_to_coeffs(res.x)
                val_check, _ = _bound_and_grad(coeffs[0], coeffs[1], coeffs[2])
                if np.isfinite(val_check) and val_check < best_val:
                    best_val = val_check
                    best = coeffs.copy()
        except Exception:
            pass

    # Optimize in direct coefficient space
    for s in direct_starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                           options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # High-precision polish in both spaces
    if time.time() < _DEADLINE - 5:
        # Polish in log-space
        lp_best = _coeffs_to_log_params(best)
        try:
            res = minimize(_log_objective_grad, lp_best, method='L-BFGS-B',
                           options={'maxiter': 20000, 'gtol': 1e-16, 'ftol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                coeffs = _log_params_to_coeffs(res.x)
                val_check, _ = _bound_and_grad(coeffs[0], coeffs[1], coeffs[2])
                if np.isfinite(val_check) and val_check < best_val:
                    best_val = val_check
                    best = coeffs.copy()
        except Exception:
            pass

        # Polish in direct space
        if time.time() < _DEADLINE - 2:
            try:
                res = minimize(_objective_grad, best, method='L-BFGS-B',
                               options={'maxiter': 20000, 'gtol': 1e-16, 'ftol': 1e-16})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best = res.x.copy()
            except Exception:
                pass

    return np.array(best)
# EVOLVE-BLOCK-END
