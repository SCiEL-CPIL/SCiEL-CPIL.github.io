"""Classical-gas equilibrium approach for 12 double roots minimizing C4.

The 12 roots x_1 < ... < x_12 are modeled as the equilibrium configuration of a
1D classical gas with:
  - Repulsive 1/(x_i - x_j)^2 pairwise potential (Jostensberg-type)
  - Logarithmic confining potential V(x) = a * ln(x)

At equilibrium, each root satisfies:
  sum_{j != i} 2/(x_i - x_j)^2  +  a/x_i  =  0   (for interior, with sign care)

Actually for a gas with repulsive 1/r^2 and confining a*ln(x):
  Force on particle i: sum_{j!=i} 2/(x_i-x_j)^2 * sign(x_j - x_i)  +  a/x_i = 0

We solve the 12 coupled nonlinear equations via Newton-Raphson, then tune the
scalar parameter `a` (and optionally a per-root shift) to minimize C4.

This is a fundamentally different representation: instead of directly optimizing
12 free variables, we exploit the algebraic structure of the equilibrium.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar, fsolve
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """int poly(x) e^{-x} dx over [0,inf) for ascending coefficient order."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 80)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_p_squared_coeffs(roots):
    """Build P(x)=prod(x-x_i)^2 and P2=P^2 in ascending coeff order."""
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2


def _equilibrium_residuals(roots, a):
    """Compute the 12 equilibrium residuals for the classical gas model.
    
    For roots x_1 < ... < x_12 with repulsive 1/(x_i-x_j)^2 potential
    and confining potential a*ln(x), the force balance at each particle is:
    
    R_i = sum_{j!=i} 2/(x_i - x_j)^2 * sign(x_j - x_i) + a / x_i = 0
    
    Note: for x_j > x_i, the repulsive force pushes i leftward (negative),
    so the contribution is -2/(x_j - x_i)^2.
    For x_j < x_i, the repulsive force pushes i rightward (positive),
    so the contribution is +2/(x_i - x_j)^2.
    
    Net repulsive: sum_{j!=i} 2*(x_j - x_i) / (x_i - x_j)^3 = -sum_{j!=i} 2/(x_i-x_j)^2 * sign(x_i-x_j)
    
    Let's use: R_i = -sum_{j!=i} 2/(x_i - x_j)^2 * sign(x_i - x_j) + a/x_i
    For x_i > x_j: sign=+1, term = -2/(x_i-x_j)^2 (pushes left, correct)
    For x_i < x_j: sign=-1, term = +2/(x_j-x_i)^2 (pushes right, correct)
    """
    n = len(roots)
    R = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            diff = roots[i] - roots[j]
            s += 2.0 / (diff * diff) * (1.0 if diff > 0 else -1.0)
        R[i] = -s + a / roots[i]
    return R


def _equilibrium_jacobian(roots, a):
    """Jacobian of equilibrium residuals w.r.t. roots."""
    n = len(roots)
    J = np.zeros((n, n))
    for i in range(n):
        # dR_i/dx_i: from the sum term and a/x_i
        # d/dx_i [-sum_{j!=i} 2/(x_i-x_j)^2 * sign(x_i-x_j)] 
        # For j < i (x_i > x_j): term = -2/(x_i-x_j)^2, d/dx_i = 4/(x_i-x_j)^3
        # For j > i (x_i < x_j): term = +2/(x_j-x_i)^2, d/dx_i = 4/(x_j-x_i)^3
        # So d/dx_i of sum = sum_{j!=i} 4/|x_i-x_j|^3
        # d/dx_i of (-sum) = -sum_{j!=i} 4/|x_i-x_j|^3
        # d/dx_i of a/x_i = -a/x_i^2
        diag_term = -a / (roots[i] * roots[i])
        for j in range(n):
            if i == j:
                continue
            d = abs(roots[i] - roots[j])
            diag_term -= 4.0 / (d * d * d)
        J[i, i] = diag_term
        
        # dR_i/dx_k for k != i
        for k in range(n):
            if k == i:
                continue
            diff = roots[i] - roots[k]
            ad = abs(diff)
            # d/dx_k of [-2/(x_i-x_k)^2 * sign(x_i-x_k)]
            # If x_i > x_k: term = -2/(x_i-x_k)^2, d/dx_k = -4/(x_i-x_k)^3
            # If x_i < x_k: term = +2/(x_k-x_i)^2, d/dx_k = -4/(x_k-x_i)^3
            # In both cases: -4/|x_i-x_k|^3 * sign(x_i - x_k) ... let me recompute
            # term = -2 * sign(diff) / diff^2  where diff = x_i - x_k
            # d/dx_k: -2 * sign(diff) * d/dx_k (1/diff^2) = -2 * sign(diff) * 2/diff^3 * (-1)
            #        = 4 * sign(diff) / diff^3
            # sign(diff)/diff^3 = 1/|diff|^3
            J[i, k] = 4.0 / (ad * ad * ad)
    return J


def _solve_equilibrium(a, x0):
    """Solve the 12 equilibrium equations for given parameter a, starting from x0.
    
    Returns the solution roots or None if it fails.
    """
    try:
        sol, info, ier, msg = fsolve(
            lambda x: _equilibrium_residuals(x, a),
            x0,
            fprime=lambda x: _equilibrium_jacobian(x, a),
            full_output=True,
            xtol=1e-14,
            maxfev=500
        )
        if ier == 1 and np.all(sol > 0) and np.all(np.isfinite(sol)):
            # Verify residuals are small
            res = _equilibrium_residuals(sol, a)
            if np.max(np.abs(res)) < 1e-6:
                return np.sort(sol)
        return None
    except Exception:
        return None

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2/I1^2 with optional gradient."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P, P2 = _build_p_squared_coeffs(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Gradient via prefix/suffix products
    n = len(roots)
    prefix = [None] * n
    prefix[0] = np.ones(1)
    for j in range(n - 1):
        prefix[j + 1] = polymul(prefix[j], np.array([roots[j]**2, -2.0*roots[j], 1.0]))

    suffix = [None] * n
    suffix[n - 1] = np.ones(1)
    for j in range(n - 2, -1, -1):
        suffix[j] = polymul(suffix[j + 1], np.array([roots[j]**2, -2.0*roots[j], 1.0]))

    grad = np.zeros(n)
    for i in range(n):
        if i == 0:
            Q_i = suffix[0].copy()
        elif i == n - 1:
            Q_i = prefix[n - 1].copy()
        else:
            Q_i = polymul(prefix[i], suffix[i])

        int_Qi = _poly_moment(Q_i)
        xQi = polymul(Q_i, np.array([0.0, 1.0]))
        int_xQi = _poly_moment(xQi)
        int_P_over_xi = int_xQi - roots[i] * int_Qi
        dI1_i = 2.0 * int_P_over_xi

        S_i = polymul(Q_i, Q_i)
        x3Si = polymul(S_i, np.array([0.0, 0.0, 0.0, 1.0]))
        x2Si = polymul(S_i, np.array([0.0, 0.0, 1.0]))
        xSi = polymul(S_i, np.array([0.0, 1.0]))
        int_x3Si = _poly_moment(x3Si)
        int_x2Si = _poly_moment(x2Si)
        int_xSi = _poly_moment(xSi)
        int_Si = _poly_moment(S_i)
        dI2_i = 4.0 * (int_x3Si - 3.0*roots[i]*int_x2Si + 3.0*roots[i]**2*int_xSi - roots[i]**3*int_Si)

        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)

    return float(c4), grad


def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^(alpha)."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf


def _c4_from_gas(a, x0):
    """Solve equilibrium for parameter a, then compute C4."""
    roots = _solve_equilibrium(a, x0)
    if roots is None:
        return np.inf, None
    c4 = _compute_c4(roots)
    return c4, roots

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _exp_objective_grad(log_roots):
    """Objective in log-space: x_i = exp(log_roots_i), removes positivity constraint."""
    x = np.exp(log_roots)
    c4, grad_x = _compute_c4(x, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(log_roots)
    grad_log = x * grad_x
    return c4, grad_log


def _roots_from_power_sums(power_sums):
    """Reconstruct roots from first 12 power sums using Newton's identities.
    
    Given p_k = sum(x_i^k) for k=1..12, find monic polynomial whose roots are x_i.
    Returns None if no valid positive real roots exist.
    """
    n = 12
    ps = np.asarray(power_sums, dtype=np.float64)
    if len(ps) < n:
        return None
    
    # Newton's identities: e_k = (1/k) * sum_{j=1}^{k} (-1)^{j-1} * e_{k-j} * p_j
    # where e_0 = 1, e_k are elementary symmetric polynomials
    e = np.zeros(n + 1)
    e[0] = 1.0
    for k in range(1, n + 1):
        s = 0.0
        for j in range(1, k + 1):
            s += ((-1) ** (j - 1)) * e[k - j] * ps[j - 1]
        e[k] = s / k
    
    # Build monic polynomial: x^n - e_1*x^{n-1} + e_2*x^{n-2} - ... + (-1)^n*e_n
    # In descending order for np.roots
    coeffs_desc = np.zeros(n + 1)
    coeffs_desc[0] = 1.0
    for k in range(1, n + 1):
        coeffs_desc[k] = ((-1) ** k) * e[n - k + 1] if (n - k + 1) >= 0 else 0.0
    
    # Actually, let's be more careful:
    # P(x) = x^n - e_1 x^{n-1} + e_2 x^{n-2} - e_3 x^{n-3} + ... + (-1)^n e_n
    # coeffs in descending order: [1, -e_1, e_2, -e_3, ..., (-1)^n e_n]
    coeffs_desc = np.zeros(n + 1)
    coeffs_desc[0] = 1.0
    for k in range(1, n + 1):
        coeffs_desc[k] = ((-1) ** k) * e[k]
    
    # Find roots
    try:
        roots = np.roots(coeffs_desc)
    except Exception:
        return None
    
    # Filter for positive real roots
    real_roots = []
    for r in roots:
        if abs(np.imag(r)) < 1e-8 and np.real(r) > 1e-10:
            real_roots.append(np.real(r))
    
    if len(real_roots) != n:
        return None
    
    return np.sort(np.array(real_roots))


def _c4_from_power_sums(power_sums):
    """Compute C4 given power sums; returns inf if invalid roots."""
    roots = _roots_from_power_sums(power_sums)
    if roots is None:
        return np.inf
    return _compute_c4(roots)


def _c4_from_power_sums_grad(power_sums):
    """Compute C4 and gradient w.r.t. power sums using finite differences."""
    n = 12
    ps = np.asarray(power_sums, dtype=np.float64)
    c4_val = _c4_from_power_sums(ps)
    if not np.isfinite(c4_val):
        return np.inf, np.zeros(n)
    
    grad = np.zeros(n)
    h = 1e-7
    for i in range(n):
        ps_plus = ps.copy()
        ps_plus[i] += h
        c4_plus = _c4_from_power_sums(ps_plus)
        if np.isfinite(c4_plus):
            grad[i] = (c4_plus - c4_val) / h
        else:
            grad[i] = 0.0
    
    return c4_val, grad


def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 0: Get baseline from generalized Laguerre search (fast, reliable)
    best_alpha = 0.0
    best_c4_alpha = np.inf
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 40)
    for la in log_alphas:
        if not time_ok():
            break
        c4_val = _c4_from_alpha(np.exp(la))
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = np.exp(la)

    xg, _ = roots_genlaguerre(12, best_alpha)
    best_roots = np.sort(xg)
    best_c4 = _compute_c4(best_roots)

    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Phase 1: Classical gas equilibrium optimization
    # Search over the scalar parameter a using the equilibrium solver
    # The gas model gives a 1-parameter family of 12-root configurations
    
    # Coarse scan over log(a)
    best_gas_a = None
    best_gas_c4 = best_c4
    best_gas_roots = best_roots.copy()
    
    log_a_grid = np.linspace(math.log(0.01), math.log(100.0), 80)
    for la in log_a_grid:
        if not time_ok():
            break
        a = math.exp(la)
        # Use scaled Laguerre roots as initial guess
        x0_init = x0 * max(1.0, a * 0.05)
        c4_gas, roots_gas = _c4_from_gas(a, x0_init)
        if np.isfinite(c4_gas) and c4_gas < best_gas_c4:
            best_gas_c4 = c4_gas
            best_gas_a = a
            best_gas_roots = roots_gas.copy()

    # Fine scan around best gas a
    if best_gas_a is not None and time_ok():
        local_lo = math.log(max(best_gas_a * 0.3, 0.001))
        local_hi = math.log(min(best_gas_a * 3.0, 1000.0))
        if local_hi > local_lo:
            log_a_fine = np.linspace(local_lo, local_hi, 100)
            for la in log_a_fine:
                if not time_ok():
                    break
                a = math.exp(la)
                x0_init = best_gas_roots * (a / best_gas_a)
                c4_gas, roots_gas = _c4_from_gas(a, x0_init)
                if np.isfinite(c4_gas) and c4_gas < best_gas_c4:
                    best_gas_c4 = c4_gas
                    best_gas_a = a
                    best_gas_roots = roots_gas.copy()

    # Scalar minimization on log(a)
    if best_gas_a is not None and time_ok():
        try:
            def gas_obj(la):
                a = math.exp(la)
                x0_init = best_gas_roots * (a / best_gas_a)
                c4, _ = _c4_from_gas(a, x0_init)
                return c4 if np.isfinite(c4) else 1e10
            
            res_a = minimize_scalar(
                gas_obj,
                bounds=(math.log(max(best_gas_a * 0.1, 0.001)), math.log(min(best_gas_a * 10.0, 10000.0))),
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 200}
            )
            if res_a.success and np.isfinite(res_a.fun) and res_a.fun < best_gas_c4:
                best_gas_a = math.exp(res_a.x)
                best_gas_c4 = res_a.fun
                x0_init = best_gas_roots * (best_gas_a / best_gas_a)  # same scale
                _, roots_new = _c4_from_gas(best_gas_a, best_gas_roots)
                if roots_new is not None:
                    best_gas_roots = roots_new.copy()
        except Exception:
            pass

    if best_gas_c4 < best_c4:
        best_c4 = best_gas_c4
        best_roots = best_gas_roots.copy()

    # Phase 2: Direct 12-DOF L-BFGS-B polish from best gas solution
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 60}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Log-space L-BFGS-B polish
    if time_ok():
        try:
            log_init = np.log(np.maximum(best_roots, 1e-10))
            res_log = minimize(
                _exp_objective_grad,
                log_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 60}
            )
            if np.isfinite(res_log.fun):
                candidate = np.sort(np.exp(res_log.x))
                if np.all(candidate > 0) and np.all(np.isfinite(candidate)):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: Multi-start with diverse gas parameters + perturbations
    rng = np.random.default_rng(123)
    diverse_starts = []
    
    # From different gas parameters
    for scale_factor in [0.3, 0.5, 0.7, 1.3, 1.5, 2.0]:
        if not time_ok():
            break
        if best_gas_a is not None:
            a_try = best_gas_a * scale_factor
            x0_try = best_gas_roots * scale_factor
            roots_try = _solve_equilibrium(a_try, x0_try)
            if roots_try is not None:
                diverse_starts.append(roots_try)
    
    # Perturbed versions
    for _ in range(8):
        if not time_ok():
            break
        pert = best_roots * (1.0 + 0.2 * rng.standard_normal(12))
        pert = np.maximum(pert, 1e-3)
        pert = np.sort(pert)
        diverse_starts.append(pert)

    for init_pt in diverse_starts:
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                init_pt,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 5: Final high-precision polish iterations
    for polish_iter in range(10):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-12, None)] * 12,
                options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 80}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass
        if not time_ok():
            break
        try:
            log_roots_init = np.log(best_roots)
            res_log = minimize(
                _exp_objective_grad,
                log_roots_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 80}
            )
            if np.isfinite(res_log.fun):
                candidate = np.sort(np.exp(res_log.x))
                if np.all(candidate > 0) and np.all(np.isfinite(candidate)):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Fallback
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
