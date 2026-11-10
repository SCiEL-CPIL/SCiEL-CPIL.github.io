import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_DEADLINE = time.time() + 615.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y_flat (NTH*NPHI, 2k+1) and w_flat (NTH*NPHI,) for quadrature."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NTH, _NPHI, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, :, idx] = N * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.sin(am * phi)[None, :]

    Y_flat = Y.reshape(_NTH * _NPHI, 2 * k + 1)
    w_flat = np.outer(w, np.full(_NPHI, dphi)).ravel()

    return Y_flat, w_flat


def _newton_on_sphere(c, Y, w, max_iter=100, tol=1e-12):
    """
    Minimize F(c) = sum(w * (Y@c)**4) subject to ||c||=1 using Newton's method.
    """
    n = len(c)
    c = c / np.linalg.norm(c)
    
    for _ in range(max_iter):
        f = Y @ c
        f2 = f * f
        f3 = f2 * f
        f4 = f2 * f2
        
        # Objective value (proportional to R^4)
        F = np.dot(w, f4)
        
        # Gradient: 4 * Y^T (w * f^3)
        grad = 4.0 * (Y.T @ (w * f3))
        
        # Hessian: 12 * Y^T (w * f^2) Y
        # Note: This is a 21x21 matrix
        H = 12.0 * (Y.T @ (w * f2)) @ Y
        
        # Project gradient to tangent space
        grad_t = grad - np.dot(grad, c) * c
        
        # Check convergence
        if np.linalg.norm(grad_t) < tol:
            break
            
        # Solve (H + lambda*I) delta = -grad_t
        # Start with small lambda, increase if singular
        lambda_reg = 1e-6
        for _ in range(10):
            try:
                delta = np.linalg.solve(H + lambda_reg * np.eye(n), -grad_t)
                break
            except np.linalg.LinAlgError:
                lambda_reg *= 10.0
        else:
            # Fallback to gradient descent step if Hessian is bad
            delta = -grad_t
            
        # Project delta to tangent space
        delta_t = delta - np.dot(delta, c) * c
        
        # Line search to ensure decrease
        step_size = 1.0
        F_curr = F
        for _ in range(20):
            c_new = c + step_size * delta_t
            norm_new = np.linalg.norm(c_new)
            if norm_new < 1e-12:
                step_size *= 0.5
                continue
            c_new = c_new / norm_new
            f_new = Y @ c_new
            F_new = np.dot(w, f_new * f_new * f_new * f_new)
            
            if F_new < F_curr:
                c = c_new
                break
            else:
                step_size *= 0.5
                
    return c, F


def _make_inits(rng, k):
    """Diverse initial directions on the unit sphere."""
    inits = []
    n = 2 * k + 1
    # Random
    for _ in range(50):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))
    # Single dominant m
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v)
    # Paired cos/sin
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
    # Alternating signs
    v = np.ones(n)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    # Random sparse
    for _ in range(20):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 9), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        if np.linalg.norm(v) > 1e-12:
            inits.append(v / np.linalg.norm(v))
    # High-m dominance
    for m in [10, 9, 8]:
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 0.5
        inits.append(v / np.linalg.norm(v))
    return inits


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_F = np.inf

    # Phase 1: Newton from diverse starts
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 60.0:
            break
        c_opt, F_opt = _newton_on_sphere(x0, Y, w, max_iter=50, tol=1e-12)
        if F_opt < best_F:
            best_F = F_opt
            best_c = c_opt.copy()

    # Phase 2: Refine best with higher tolerance
    if time.time() < _DEADLINE - 30.0:
        c_opt, F_opt = _newton_on_sphere(best_c, Y, w, max_iter=100, tol=1e-14)
        if F_opt < best_F:
            best_F = F_opt
            best_c = c_opt.copy()

    # Phase 3: Random restarts from perturbed best
    while time.time() < _DEADLINE - 10.0:
        x0 = best_c + 0.1 * rng.standard_normal(n)
        x0 = x0 / np.linalg.norm(x0)
        c_opt, F_opt = _newton_on_sphere(x0, Y, w, max_iter=50, tol=1e-12)
        if F_opt < best_F:
            best_F = F_opt
            best_c = c_opt.copy()

    # Final normalization
    out = best_c.astype(np.float64)
    nrm = np.linalg.norm(out)
    if nrm < 1e-12:
        out = np.ones(n, dtype=np.float64)
    else:
        out = out / nrm

    # Validation
    if out.shape != (21,):
        out = np.ones(21, dtype=np.float64)
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21, dtype=np.float64)

    max_abs = np.max(np.abs(out))
    if max_abs < 1e-3:
        out = out / max_abs * 1.0
    elif max_abs > 1e5:
        out = out / max_abs * 1.0

    return out
