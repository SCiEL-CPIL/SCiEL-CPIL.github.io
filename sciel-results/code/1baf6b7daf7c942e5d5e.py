# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv
from math import lgamma

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1
_DEADLINE_OFFSET = 590.0
_SAFETY = 20.0

def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        log_ratio = lgamma(_K - am + 1) - lgamma(_K + am + 1)
        fact_ratio = np.exp(log_ratio)
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W

_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)

def _objective(c):
    """Minimize R^4 = <f^4> / <f^2>^2. Scale-invariant."""
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)

def _gradient(c):
    """Analytic gradient of R^4 = <f^4>/<f^2>^2 w.r.t. c."""
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad

def _z_to_c(z):
    """Convert complex amplitude vector z = [c_0, z_1, ..., z_10] to 21-coeff vector.
    z_m = c_{-m} + i*c_m for m >= 1, z_0 = c_0."""
    c = np.zeros(_NC)
    c[_K] = z[0]  # c_0
    for m in range(1, _K + 1):
        z_m = z[m]
        c[_K - m] = z_m.real   # c_{-m}
        c[_K + m] = z_m.imag   # c_{m}
    return c

def _c_to_z(c):
    """Convert 21-coeff vector to complex amplitude vector."""
    z = np.zeros(_K + 1, dtype=np.complex128)
    z[0] = c[_K]  # c_0
    for m in range(1, _K + 1):
        z[m] = c[_K - m] + 1j * c[_K + m]
    return z

def _objective_z(z):
    """Objective in complex amplitude space (unconstrained)."""
    c = _z_to_c(z)
    return _objective(c)

def _gradient_z(z):
    """Gradient in complex amplitude space via chain rule."""
    c = _z_to_c(z)
    gc = _gradient(c)
    gz = np.zeros(_K + 1, dtype=np.complex128)
    gz[0] = gc[_K]  # dc_0/dz_0 = 1
    for m in range(1, _K + 1):
        # z_m = c_{-m} + i*c_m
        # dL/dz_m = dL/dc_{-m} * dc_{-m}/dz_m + dL/dc_m * dc_m/dz_m
        # dc_{-m}/dz_m = 1 (real part), dc_m/dz_m = i (imag part)
        # But we need the Wirtinger derivative for the real-valued objective
        # For real-valued L, dL/dz = dL/dc_{-m} + i*dL/dc_m (treating z as independent)
        # Actually for real-valued function L(c_{-m}, c_m) with z = c_{-m} + i*c_m:
        # dL/dz_real = dL/dc_{-m}, dL/dz_imag = dL/dc_m
        # So gradient in (Re(z), Im(z)) space is (dL/dc_{-m}, dL/dc_m)
        gz[m] = gc[_K - m] + 1j * gc[_K + m]
    return gz

def _objective_z_real(zr):
    """Objective where zr is real vector of length 2*(K+1): [c0, Re(z1), Im(z1), ..., Re(z10), Im(z10)]."""
    z = np.zeros(_K + 1, dtype=np.complex128)
    z[0] = zr[0]
    for m in range(1, _K + 1):
        z[m] = zr[2*m - 1] + 1j * zr[2*m]
    c = _z_to_c(z)
    return _objective(c)

def _gradient_z_real(zr):
    """Gradient in real vector representation of complex amplitudes."""
    z = np.zeros(_K + 1, dtype=np.complex128)
    z[0] = zr[0]
    for m in range(1, _K + 1):
        z[m] = zr[2*m - 1] + 1j * zr[2*m]
    c = _z_to_c(z)
    gc = _gradient(c)
    gzr = np.zeros(2 * (_K + 1))
    gzr[0] = gc[_K]
    for m in range(1, _K + 1):
        gzr[2*m - 1] = gc[_K - m]   # dL/dRe(z_m) = dL/dc_{-m}
        gzr[2*m] = gc[_K + m]        # dL/dIm(z_m) = dL/dc_m
    return gzr

def _zr_to_c(zr):
    """Convert real vector representation to 21-coeff vector."""
    z = np.zeros(_K + 1, dtype=np.complex128)
    z[0] = zr[0]
    for m in range(1, _K + 1):
        z[m] = zr[2*m - 1] + 1j * zr[2*m]
    return _z_to_c(z)

def _c_to_zr(c):
    """Convert 21-coeff vector to real vector representation."""
    zr = np.zeros(2 * (_K + 1))
    zr[0] = c[_K]
    for m in range(1, _K + 1):
        zr[2*m - 1] = c[_K - m]
        zr[2*m] = c[_K + m]
    return zr

def _norm_constraint(c):
    return c @ c - 1.0

def _norm_constraint_jac(c):
    return 2.0 * c

def _projected_cg(c, deadline, max_iter=2000):
    """Projected conjugate gradient on the unit sphere with backtracking line search."""
    c = c.copy()
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return c, 1e30
    val = float(np.dot(_Wflat, f2 * f2)) / (l2sq * l2sq)
    
    # Projected gradient: remove radial component
    g = _gradient(c)
    g = g - np.dot(g, c) * c  # project onto tangent space
    
    d = -g
    d = d - np.dot(d, c) * c
    dn = np.linalg.norm(d)
    if dn < 1e-15:
        return c, val
    d /= dn
    
    for it in range(max_iter):
        if time.time() > deadline - _SAFETY:
            break
        
        # Backtracking line search on the sphere
        alpha = 0.1
        best_alpha = alpha
        best_val = val
        for _ in range(30):
            # Move on sphere: c_new = (c + alpha*d) / ||c + alpha*d||
            c_new = c + alpha * d
            nrm = np.linalg.norm(c_new)
            if nrm < 1e-14:
                alpha *= 0.5
                continue
            c_new /= nrm
            v_new = _objective(c_new)
            if v_new < best_val:
                best_val = v_new
                best_alpha = alpha
                best_c_new = c_new.copy()
            if v_new >= val * (1.0 - 1e-4 * alpha):
                # Sufficient decrease
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break
        
        if best_val >= val - 1e-16:
            break
        
        c = best_c_new
        val = best_val
        
        # Update gradient
        g_new = _gradient(c)
        g_new = g_new - np.dot(g_new, c) * c
        
        # Fletcher-Reeves beta
        g_norm_sq = np.dot(g, g)
        if g_norm_sq > 1e-30:
            beta = np.dot(g_new, g_new) / g_norm_sq
        else:
            beta = 0.0
        if beta < 0:
            beta = 0.0
        
        d_new = -g_new + beta * d
        d_new = d_new - np.dot(d_new, c) * c
        dn = np.linalg.norm(d_new)
        if dn < 1e-15:
            break
        d = d_new / dn
        g = g_new
    
    return c, val

def _riemannian_lbfgs(c, deadline, max_iter=3000):
    """Riemannian L-BFGS on the unit sphere using exponential map (retraction)."""
    c = np.asarray(c, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return c, 1e30
    c = c / nrm

    val = _objective(c)
    g = _gradient(c)
    g = g - np.dot(g, c) * c  # project to tangent

    # L-BFGS history
    m_lbfgs = 10
    s_list = []
    y_list = []
    rho_list = []

    d = -g.copy()
    d = d - np.dot(d, c) * c
    dn = np.linalg.norm(d)
    if dn < 1e-15:
        return c, val

    for it in range(max_iter):
        if time.time() > deadline - _SAFETY:
            break

        # Line search on sphere
        alpha = 1.0
        best_alpha = alpha
        best_val = val
        best_c_new = None
        for _ls in range(40):
            c_trial = c + alpha * d
            nt = np.linalg.norm(c_trial)
            if nt < 1e-14:
                alpha *= 0.5
                continue
            c_trial /= nt
            v_trial = _objective(c_trial)
            if v_trial < best_val:
                best_val = v_trial
                best_alpha = alpha
                best_c_new = c_trial.copy()
            # Armijo condition
            if v_trial <= val + 1e-4 * alpha * np.dot(g, d):
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break

        if best_c_new is None or best_val >= val - 1e-16:
            break

        c_new = best_c_new
        val_new = best_val

        # Compute new gradient
        g_new = _gradient(c_new)
        g_new = g_new - np.dot(g_new, c_new) * c_new

        # Store history for L-BFGS
        s = best_alpha * d
        y = g_new - g
        sy = np.dot(s, y)
        if sy > 1e-12:
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)
            if len(s_list) > m_lbfgs:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)

        # L-BFGS two-loop recursion for new direction
        if len(s_list) > 0:
            q = g_new.copy()
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                ai = rho_list[i] * np.dot(s_list[i], q)
                alphas.append(ai)
                q -= ai * y_list[i]
            # Initial Hessian approximation
            if len(s_list) == 1:
                gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
            else:
                gamma = 1.0 / (np.dot(y_list[-1], y_list[-1]) + 1e-15)
            r = gamma * q
            for i in range(len(s_list)):
                bi = rho_list[i] * np.dot(y_list[i], r)
                r += (alphas[i] - bi) * s_list[i]
            d_new = -r
        else:
            d_new = -g_new

        # Project to tangent
        d_new = d_new - np.dot(d_new, c_new) * c_new
        dn = np.linalg.norm(d_new)
        if dn < 1e-15:
            c = c_new
            val = val_new
            break
        d = d_new / dn
        g = g_new
        c = c_new
        val = val_new

    return c, val

def _optimize_from(c0, deadline):
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm

    def fun(x):
        return _objective(x)

    def jac(x):
        return _gradient(x)

    constraints = [{'type': 'eq', 'fun': _norm_constraint, 'jac': _norm_constraint_jac}]

    best_c = c.copy()
    best_val = fun(c)

    # Phase 1: SLSQP with unit-norm constraint in full 21-dim space
    try:
        res = minimize(fun, c, jac=jac, method="SLSQP", constraints=constraints,
                       options={"maxiter": 300, "ftol": 1e-14, "disp": False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    # Re-normalize after SLSQP
    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    best_c = best_c / nrm
    best_val = fun(best_c)

    if time.time() > deadline - _SAFETY:
        return best_c, best_val

    # Phase 1.5: Riemannian L-BFGS
    try:
        c_lbfgs, val_lbfgs = _riemannian_lbfgs(best_c, deadline, max_iter=3000)
        if val_lbfgs < best_val:
            best_val = val_lbfgs
            best_c = c_lbfgs.copy()
    except Exception:
        pass

    if time.time() > deadline - _SAFETY:
        nrm = np.linalg.norm(best_c)
        if nrm < 1e-14:
            return None
        best_c = best_c / nrm
        return best_c, float(_objective(best_c))

    # Phase 1.7: Projected CG
    try:
        c_cg, val_cg = _projected_cg(best_c, deadline, max_iter=1500)
        if val_cg < best_val:
            best_val = val_cg
            best_c = c_cg.copy()
    except Exception:
        pass

    if time.time() > deadline - _SAFETY:
        nrm = np.linalg.norm(best_c)
        if nrm < 1e-14:
            return None
        best_c = best_c / nrm
        return best_c, float(_objective(best_c))

    # Phase 2: Nelder-Mead polish
    try:
        res2 = minimize(fun, best_c, method="Nelder-Mead",
                        options={"maxiter": 2000, "xatol": 1e-14,
                                 "fatol": 1e-16, "adaptive": True})
        if res2.fun < best_val:
            best_val = res2.fun
            best_c = res2.x.copy()
    except Exception:
        pass

    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    best_c = best_c / nrm
    return best_c, float(_objective(best_c))

def _zr_to_c_11(zr):
    """Convert 11-dim reduced representation [c0, r1,theta1,...,r10,theta10] to 21-coeff.
    For m>=1: c_{-m} = r_m*cos(theta_m), c_m = r_m*sin(theta_m).
    The objective is invariant under global phase, so this 11-dim space is sufficient."""
    c = np.zeros(_NC)
    c[_K] = zr[0]
    for m in range(1, _K + 1):
        r = zr[2*m - 1]
        th = zr[2*m]
        c[_K - m] = r * np.cos(th)
        c[_K + m] = r * np.sin(th)
    return c

def _c_to_zr_11(c):
    """Convert 21-coeff to 11-dim reduced representation."""
    zr = np.zeros(2 * _K + 1)
    zr[0] = c[_K]
    for m in range(1, _K + 1):
        a = c[_K - m]
        b = c[_K + m]
        r = np.sqrt(a*a + b*b)
        if r > 1e-15:
            th = np.arctan2(b, a)
        else:
            th = 0.0
        zr[2*m - 1] = r
        zr[2*m] = th
    return zr

def _objective_11(zr):
    c = _zr_to_c_11(zr)
    return _objective(c)

def _gradient_11(zr):
    c = _zr_to_c_11(zr)
    gc = _gradient(c)
    gzr = np.zeros(2 * _K + 1)
    gzr[0] = gc[_K]
    for m in range(1, _K + 1):
        a = c[_K - m]
        b = c[_K + m]
        r = np.sqrt(a*a + b*b)
        if r > 1e-15:
            ct = a / r
            st = b / r
        else:
            ct, st = 1.0, 0.0
        # dc_{-m}/dr = ct, dc_{-m}/dth = -r*st
        # dc_m/dr = st, dc_m/dth = r*ct
        gzr[2*m - 1] = gc[_K - m] * ct + gc[_K + m] * st
        gzr[2*m] = gc[_K - m] * (-r * st) + gc[_K + m] * (r * ct)
    return gzr

def _optimize_from_11(zr0, deadline):
    """Optimize in the 11-dim phase-reduced space, then polish in 21-dim."""
    zr = np.asarray(zr0, dtype=np.float64).copy()
    nrm = np.linalg.norm(zr)
    if nrm < 1e-14:
        return None
    zr = zr / nrm

    def fun(x):
        return _objective_11(x)

    def jac(x):
        return _gradient_11(x)

    best_zr = zr.copy()
    best_val = fun(zr)

    # SLSQP in 11-dim with unit-norm constraint
    def norm_con(x):
        return x @ x - 1.0

    def norm_con_jac(x):
        return 2.0 * x

    constraints = [{'type': 'eq', 'fun': norm_con, 'jac': norm_con_jac}]

    try:
        res = minimize(fun, zr, jac=jac, method="SLSQP", constraints=constraints,
                       options={"maxiter": 400, "ftol": 1e-14, "disp": False})
        if res.fun < best_val:
            best_val = res.fun
            best_zr = res.x.copy()
    except Exception:
        pass

    nrm = np.linalg.norm(best_zr)
    if nrm < 1e-14:
        return None
    best_zr = best_zr / nrm
    best_val = fun(best_zr)

    c = _zr_to_c_11(best_zr)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm

    if time.time() > deadline - _SAFETY:
        return c, float(_objective(c))

    # Polish in full 21-dim space
    try:
        c_cg, val_cg = _projected_cg(c, deadline, max_iter=2000)
        if val_cg < best_val:
            best_val = val_cg
            c = c_cg.copy()
    except Exception:
        pass

    if time.time() > deadline - _SAFETY:
        nrm = np.linalg.norm(c)
        if nrm < 1e-14:
            return None
        c = c / nrm
        return c, float(_objective(c))

    # Final Nelder-Mead in 21-dim
    try:
        res2 = minimize(_objective, c, method="Nelder-Mead",
                        options={"maxiter": 1500, "xatol": 1e-14,
                                 "fatol": 1e-16, "adaptive": True})
        if res2.fun < best_val:
            best_val = res2.fun
            c = res2.x.copy()
    except Exception:
        pass

    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm
    return c, float(_objective(c))

def _make_seeds(rng, n_seeds):
    seeds = []
    for i in range(n_seeds):
        c = np.zeros(_NC)
        if i == 0:
            c[_K] = 1.0
        elif i == 1:
            c[_K + _K] = 1.0
        elif i == 2:
            c[_K - _K] = 1.0
        elif i == 3:
            for m in range(-_K, _K + 1):
                c[m + _K] = np.cos(0.5 * m)
        elif i == 4:
            for m in range(-_K, _K + 1):
                c[m + _K] = (-1) ** m
        elif i == 5:
            c[:] = rng.standard_normal(_NC)
        elif i == 6:
            for m in range(-_K, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 7:
            for m in range(-_K + 1, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 8:
            c[_K] = 1.0
            c[_K + _K] = rng.standard_normal()
            c[_K - _K] = rng.standard_normal()
        elif i == 9:
            c[_K + 5] = rng.standard_normal()
            c[_K - 5] = rng.standard_normal()
        elif i == 10:
            for m in range(0, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = v
        elif i == 11:
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = -v
        elif i == 12:
            for m in range(-3, 4):
                c[m + _K] = rng.standard_normal()
        elif i == 13:
            for m in list(range(-10, -6)) + list(range(7, 11)):
                c[m + _K] = rng.standard_normal()
        elif i == 14:
            c[_K] = 1.0
            c[_K + 5] = 0.5 * rng.standard_normal()
            c[_K - 5] = 0.5 * rng.standard_normal()
            c[_K + 10] = 0.1 * rng.standard_normal()
            c[_K - 10] = 0.1 * rng.standard_normal()
        elif i == 15:
            for m in range(-_K, _K + 1):
                c[m + _K] = rng.standard_normal()
        elif i == 16:
            c[_K] = 2.0
            c[_K + 1] = rng.standard_normal()
            c[_K - 1] = rng.standard_normal()
            c[_K + 2] = 0.3 * rng.standard_normal()
            c[_K - 2] = 0.3 * rng.standard_normal()
        elif i == 17:
            c[_K + 10] = 2.0
            c[_K - 10] = 2.0
            c[_K + 9] = 0.5 * rng.standard_normal()
            c[_K - 9] = 0.5 * rng.standard_normal()
            c[_K] = 0.3 * rng.standard_normal()
        elif i == 18:
            for m in range(-_K, _K + 1):
                c[m + _K] = rng.standard_normal() * (0.7 ** abs(m))
        elif i == 19:
            c[_K] = 1.5
            c[_K + 10] = 1.0 * rng.standard_normal()
            c[_K - 10] = 1.0 * rng.standard_normal()
            for m in range(-4, 5):
                if m != 0:
                    c[m + _K] = 0.2 * rng.standard_normal()
        elif i == 20:
            for m in range(-_K, _K + 1):
                c[m + _K] = rng.standard_normal() / (1.0 + 0.3 * abs(m))
        elif i == 21:
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = v
            c[_K] = rng.standard_normal()
        elif i == 22:
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = -v
        else:
            c[:] = rng.standard_normal(_NC)
            c[_K] += 0.5
        nrm = np.linalg.norm(c)
        if nrm > 1e-14:
            c /= nrm
        seeds.append(c)
    return seeds

def _make_zr11_seeds(rng, n_seeds):
    """Generate seeds in the 11-dim reduced space [c0, r1,th1,...,r10,th10]."""
    seeds = []
    for i in range(n_seeds):
        zr = np.zeros(2 * _K + 1)
        if i == 0:
            zr[0] = 1.0
        elif i == 1:
            zr[2*_K] = np.pi/2  # r10=0, but set angle; actually set r10
            zr[2*_K - 1] = 1.0
            zr[2*_K] = np.pi/2
        elif i == 2:
            zr[2*_K - 1] = 1.0
            zr[2*_K] = 0.0
        elif i == 3:
            for m in range(1, _K + 1):
                zr[2*m - 1] = 1.0
                zr[2*m] = 0.5 * m
        elif i == 4:
            for m in range(1, _K + 1):
                zr[2*m - 1] = 1.0
                zr[2*m] = np.pi * (m % 2)
        elif i == 5:
            zr[:] = rng.standard_normal(2 * _K + 1)
        elif i == 6:
            for m in range(1, _K + 1, 2):
                zr[2*m - 1] = abs(rng.standard_normal())
                zr[2*m] = rng.standard_normal() * np.pi
        elif i == 7:
            for m in range(2, _K + 1, 2):
                zr[2*m - 1] = abs(rng.standard_normal())
                zr[2*m] = rng.standard_normal() * np.pi
        elif i == 8:
            zr[0] = 1.0
            zr[2*_K - 1] = abs(rng.standard_normal())
            zr[2*_K] = rng.standard_normal() * np.pi
        elif i == 9:
            zr[2*5 - 1] = abs(rng.standard_normal())
            zr[2*5] = rng.standard_normal() * np.pi
        elif i == 10:
            for m in range(1, _K + 1):
                zr[2*m - 1] = abs(rng.standard_normal())
                zr[2*m] = 0.0
        elif i == 11:
            for m in range(1, _K + 1):
                zr[2*m - 1] = abs(rng.standard_normal())
                zr[2*m] = np.pi / 2
        elif i == 12:
            for m in range(1, 4):
                zr[2*m - 1] = abs(rng.standard_normal())
                zr[2*m] = rng.standard_normal() * np.pi
        elif i == 13:
            for m in [7, 8, 9, 10]:
                zr[2*m - 1] = abs(rng.standard_normal())
                zr[2*m] = rng.standard_normal() * np.pi
        elif i == 14:
            zr[0] = 1.0
            zr[2*5 - 1] = 0.5
            zr[2*5] = rng.standard_normal() * np.pi
            zr[2*10 - 1] = 0.1
            zr[2*10] = rng.standard_normal() * np.pi
        elif i == 15:
            zr[:] = rng.standard_normal(2 * _K + 1)
        elif i == 16:
            zr[0] = 2.0
            zr[1] = abs(rng.standard_normal())
            zr[2] = rng.standard_normal() * np.pi
            zr[3] = 0.3 * abs(rng.standard_normal())
            zr[4] = rng.standard_normal() * np.pi
        elif i == 17:
            zr[2*10 - 1] = 2.0
            zr[2*10] = np.pi / 4
            zr[2*9 - 1] = 0.5
            zr[2*9] = rng.standard_normal() * np.pi
            zr[0] = 0.3 * rng.standard_normal()
        elif i == 18:
            for m in range(1, _K + 1):
                zr[2*m - 1] = (0.7 ** m) * abs(rng.standard_normal())
                zr[2*m] = rng.standard_normal() * np.pi
        elif i == 19:
            zr[0] = 1.5
            zr[2*10 - 1] = abs(rng.standard_normal())
            zr[2*10] = rng.standard_normal() * np.pi
            for m in range(1, 5):
                zr[2*m - 1] = 0.2 * abs(rng.standard_normal())
                zr[2*m] = rng.standard_normal() * np.pi
        elif i == 20:
            for m in range(1, _K + 1):
                zr[2*m - 1] = abs(rng.standard_normal()) / (1.0 + 0.3 * m)
                zr[2*m] = rng.standard_normal() * np.pi
        elif i == 21:
            for m in range(1, _K + 1):
                zr[2*m - 1] = abs(rng.standard_normal())
                zr[2*m] = 0.0
            zr[0] = rng.standard_normal()
        elif i == 22:
            for m in range(1, _K + 1):
                zr[2*m - 1] = abs(rng.standard_normal())
                zr[2*m] = np.pi / 2
        else:
            zr[:] = rng.standard_normal(2 * _K + 1)
            zr[0] += 0.5
        nrm = np.linalg.norm(zr)
        if nrm > 1e-14:
            zr /= nrm
        seeds.append(zr)
    return seeds

def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + _DEADLINE_OFFSET
    rng = np.random.default_rng(42)

    best_c = None
    best_val = np.inf

    # Phase A: Optimize in full 21-dim space from structured seeds
    seeds_21 = _make_seeds(rng, 24)
    for i, c0 in enumerate(seeds_21):
        if time.time() > deadline - _SAFETY:
            break
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Phase B: Optimize in 11-dim reduced space from structured seeds
    if time.time() < deadline - 60.0:
        seeds_zr11 = _make_zr11_seeds(rng, 24)
        for i, zr0 in enumerate(seeds_zr11):
            if time.time() > deadline - _SAFETY:
                break
            try:
                out = _optimize_from_11(zr0, deadline)
            except Exception:
                continue
            if out is None:
                continue
            c, val = out
            if val < best_val:
                best_val = val
                best_c = c.copy()
        # Additional structured seeds in 21-dim: emphasize different m combinations
        extra_seeds = []
        # Seed with only m=±10
        c_extra = np.zeros(_NC)
        c_extra[_K + 10] = 1.0
        c_extra[_K - 10] = 1.0
        c_extra /= np.linalg.norm(c_extra)
        extra_seeds.append(c_extra)
        # Seed with m=0 and m=±10
        c_extra = np.zeros(_NC)
        c_extra[_K] = 1.0
        c_extra[_K + 10] = 0.7
        c_extra[_K - 10] = 0.7
        c_extra /= np.linalg.norm(c_extra)
        extra_seeds.append(c_extra)
        # Seed with m=±8, ±10
        c_extra = np.zeros(_NC)
        c_extra[_K + 8] = 1.0
        c_extra[_K - 8] = 1.0
        c_extra[_K + 10] = 0.5
        c_extra[_K - 10] = 0.5
        c_extra /= np.linalg.norm(c_extra)
        extra_seeds.append(c_extra)
        # Seed with m=±4, ±6, ±8, ±10 (even m only, nonzero)
        c_extra = np.zeros(_NC)
        for m in [4, 6, 8, 10]:
            c_extra[_K + m] = 1.0 / m
            c_extra[_K - m] = 1.0 / m
        c_extra /= np.linalg.norm(c_extra)
        extra_seeds.append(c_extra)
        # Seed with m=±1, ±3, ±5, ±7, ±9 (odd m only)
        c_extra = np.zeros(_NC)
        for m in [1, 3, 5, 7, 9]:
            c_extra[_K + m] = 1.0 / m
            c_extra[_K - m] = 1.0 / m
        c_extra /= np.linalg.norm(c_extra)
        extra_seeds.append(c_extra)
        
        for c0 in extra_seeds:
            if time.time() > deadline - _SAFETY:
                break
            try:
                out = _optimize_from(c0, deadline)
            except Exception:
                continue
            if out is None:
                continue
            c, val = out
            if val < best_val:
                best_val = val
                best_c = c.copy()

    # Phase C: Random restarts in 21-dim space
    n_extra = 0
    while time.time() < deadline - 50.0:
        c0 = rng.standard_normal(_NC)
        nrm = np.linalg.norm(c0)
        if nrm < 1e-14:
            continue
        c0 /= nrm
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()
        n_extra += 1

    # Phase D: Random restarts in 11-dim reduced space
    while time.time() < deadline - 40.0:
        zr0 = rng.standard_normal(2 * _K + 1)
        nrm = np.linalg.norm(zr0)
        if nrm < 1e-14:
            continue
        zr0 /= nrm
        try:
            out = _optimize_from_11(zr0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Final high-precision polish if time permits
    if best_c is not None and time.time() < deadline - 15.0:
        constraints = [{'type': 'eq', 'fun': _norm_constraint, 'jac': _norm_constraint_jac}]
        try:
            res = minimize(_objective, best_c, jac=_gradient, method="SLSQP",
                           constraints=constraints,
                           options={"maxiter": 500, "ftol": 1e-16, "disp": False})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
        
        # Projected CG polish (very tight)
        if time.time() < deadline - 10.0:
            try:
                c_cg, val_cg = _projected_cg(best_c, deadline, max_iter=5000)
                if val_cg < best_val:
                    best_val = val_cg
                    best_c = c_cg.copy()
            except Exception:
                pass

        # One more Nelder-Mead polish
        if time.time() < deadline - 5.0:
            try:
                res2 = minimize(_objective, best_c, method="Nelder-Mead",
                                options={"maxiter": 3000, "xatol": 1e-15,
                                         "fatol": 1e-17, "adaptive": True})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_c = res2.x.copy()
            except Exception:
                pass

    if best_c is None:
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)

    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    return best_c
# EVOLVE-BLOCK-END
