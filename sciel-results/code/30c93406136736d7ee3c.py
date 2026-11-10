# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, NonlinearConstraint

_DEADLINE = time.time() + 610.0  # safety margin under the 630s kill
_K = 10
_NTH = 70
_NPHI = 130
_DIM = 2 * _K + 1  # 21


def _build_harmonics():
    """Return Y_flat (N, 2k+1), w_flat (N,) where N = NTH*NPHI, and dphi."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y_flat = np.zeros((_NTH * _NPHI, _DIM), dtype=np.float64)
    
    cos_table = np.zeros((k + 1, _NPHI))
    sin_table = np.zeros((k + 1, _NPHI))
    for m in range(k + 1):
        cos_table[m] = np.cos(m * phi)
        sin_table[m] = np.sin(m * phi)

    N_m = np.zeros(k + 1)
    P_m = np.zeros((k + 1, _NTH))
    for am in range(k + 1):
        N_m[am] = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                          * float(factorial(k - am, exact=True))
                          / float(factorial(k + am, exact=True)))
        P_m[am] = lpmv(am, k, mu)

    for th_idx in range(_NTH):
        for ph_idx in range(_NPHI):
            i = th_idx * _NPHI + ph_idx
            for m in range(-k, k + 1):
                idx = m + k
                am = abs(m)
                if m == 0:
                    Y_flat[i, idx] = N_m[0] * P_m[0, th_idx]
                elif m > 0:
                    Y_flat[i, idx] = np.sqrt(2.0) * N_m[m] * P_m[m, th_idx] * cos_table[m, ph_idx]
                else:
                    Y_flat[i, idx] = np.sqrt(2.0) * N_m[am] * P_m[am, th_idx] * sin_table[am, ph_idx]

    w_flat = np.tile(w, _NPHI)
    return Y_flat, w_flat, dphi


def _ratio_val(Y_flat, w_flat, dphi, c):
    """Return R = ||f||_4 / ||f||_2."""
    f = Y_flat @ c
    f2 = f * f
    f4 = f2 * f2
    n2 = np.sum(f2 * w_flat) * dphi
    n4 = np.sum(f4 * w_flat) * dphi
    if n2 < 1e-15:
        return np.inf
    return (n4 ** 0.25) / np.sqrt(n2)


def _objective_and_grad(Y_flat, w_flat, dphi, c):
    """Return (R, grad_R) for R = ||f||_4 / ||f||_2."""
    f = Y_flat @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2
    n2 = np.sum(f2 * w_flat) * dphi
    n4 = np.sum(f4 * w_flat) * dphi
    if n2 < 1e-15:
        return np.inf, np.zeros_like(c)
    R = (n4 ** 0.25) / np.sqrt(n2)
    
    term1 = Y_flat.T @ (f3 * w_flat)
    term2 = Y_flat.T @ (f * w_flat)
    grad_R = dphi * (n4 ** (-0.75)) * (n2 ** (-1.5)) * (n2 * term1 - n4 * term2)
    return R, grad_R


def _project_to_sphere(c):
    nrm = np.linalg.norm(c)
    if nrm < 1e-12:
        return c
    return c / nrm


def _make_inits(rng, k):
    """Diverse initial directions."""
    inits = []
    for _ in range(20):
        v = rng.standard_normal(_DIM)
        inits.append(v / np.linalg.norm(v))
    for m in range(-k, k + 1):
        v = np.zeros(_DIM)
        v[m + k] = 1.0
        inits.append(v)
    for m in range(1, k + 1):
        v = np.zeros(_DIM)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
    v = np.ones(_DIM)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    for _ in range(10):
        v = np.zeros(_DIM)
        idx = rng.choice(_DIM, size=rng.integers(2, 8), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        if np.linalg.norm(v) > 0:
            inits.append(v / np.linalg.norm(v))
    for m in [k, k-1, k-2]:
        v = np.zeros(_DIM)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
    return inits


def _optimize_from(Y_flat, w_flat, dphi, x0, maxiter=300):
    """Use scipy trust-constr with sphere constraint to minimize R."""
    sphere_eq = NonlinearConstraint(
        lambda c: np.dot(c, c), 1.0 - 1e-8, 1.0 + 1e-8,
        jac=lambda c: 2.0 * c
    )
    
    def obj(c):
        R, _ = _objective_and_grad(Y_flat, w_flat, dphi, c)
        return R
    
    def grad(c):
        _, g = _objective_and_grad(Y_flat, w_flat, dphi, c)
        return g
    
    try:
        res = minimize(
            obj, x0, jac=grad, method='trust-constr',
            constraints=[sphere_eq],
            options={'maxiter': maxiter, 'gtol': 1e-8, 'xtol': 1e-10,
                     'barrier_tol': 1e-8, 'verbose': 0}
        )
        c = _project_to_sphere(res.x)
        val = _ratio_val(Y_flat, w_flat, dphi, c)
        if np.isfinite(val):
            return c, val
    except Exception:
        pass
    
    # Fallback: simple projected gradient
    c = _project_to_sphere(x0.copy())
    best_val = _ratio_val(Y_flat, w_flat, dphi, c)
    lr = 0.01
    for it in range(200):
        R, g = _objective_and_grad(Y_flat, w_flat, dphi, c)
        if not np.isfinite(R):
            break
        g_tan = g - np.dot(g, c) * c
        gn = np.linalg.norm(g_tan)
        if gn < 1e-12:
            break
        c_new = _project_to_sphere(c - lr * g_tan)
        R_new = _ratio_val(Y_flat, w_flat, dphi, c_new)
        if R_new < best_val:
            c = c_new
            best_val = R_new
    return c, best_val


def rudin_problem10() -> np.ndarray:
    Y_flat, w_flat, dphi = _build_harmonics()
    k = _K
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = np.inf

    # Phase 1: trust-constr from diverse starts
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 120:
            break
        c, val = _optimize_from(Y_flat, w_flat, dphi, x0, maxiter=200)
        if val < best_val:
            best_val = val
            best_c = c.copy()
        scored.append((val, c))

    scored.sort(key=lambda t: t[0])
    top = [c for _, c in scored[:6]]
    if not top:
        top = [best_c]

    # Phase 2: refine top candidates with more iterations
    for x0 in top:
        if time.time() > _DEADLINE - 60:
            break
        c, val = _optimize_from(Y_flat, w_flat, dphi, x0, maxiter=500)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Phase 3: random restarts from perturbed best
    while time.time() < _DEADLINE - 20:
        x0 = best_c + 0.15 * rng.standard_normal(_DIM)
        x0 = _project_to_sphere(x0)
        c, val = _optimize_from(Y_flat, w_flat, dphi, x0, maxiter=300)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Phase 4: final high-precision polish
    if time.time() < _DEADLINE - 5:
        c, val = _optimize_from(Y_flat, w_flat, dphi, best_c, maxiter=1000)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    out = best_c.astype(np.float64)
    out = out / np.linalg.norm(out)
    return out


# EVOLVE-BLOCK-END
