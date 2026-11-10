import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_DEADLINE = time.time() + 610.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y (nphi*nth, 2k+1), weights (nphi*nth,) for quadrature."""
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


def _ratio_val_grad_hess(c, Y, w):
    """Compute R^4, gradient, and Hessian of S4/S2^2."""
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2

    S2 = np.dot(w, f2)
    S4 = np.dot(w, f4)

    if S2 < 1e-15:
        return 1e12, np.zeros_like(c), np.zeros((len(c), len(c)))

    val = S4 / (S2 * S2)

    # Gradient
    Yt_wf = Y.T @ (w * f)
    Yt_wf3 = Y.T @ (w * f3)

    dS2 = 2.0 * Yt_wf
    dS4 = 4.0 * Yt_wf3

    grad = dS4 / (S2 * S2) - 2.0 * S4 * dS2 / (S2 * S2 * S2)

    # Hessian
    # dS4/dci = 4 * sum_j w_j * 3*f_j^2 * Y_ji = 12 * (Y.T @ (w * f^2))_i
    # d2S4/dcidk = 12 * sum_j w_j * (3*f_j * Y_ji * Y_jk) + 12 * sum_j w_j * f_j^2 * 0 (since Y is linear)
    # Actually: S4 = sum_j w_j f_j^4, df_j/dc_i = Y_ji, d2f_j/dci dk = 0
    # dS4/dc_i = 4 * sum_j w_j f_j^3 Y_ji
    # d2S4/dc_i dc_k = 4 * sum_j w_j * 3*f_j^2 * Y_ji * Y_kj = 12 * (Y.T @ diag(w * f^2) @ Y)_{ik}
    
    # d2S2/dci dk = 2 * sum_j w_j Y_ji Y_kj = 2 * (Y.T @ diag(w) @ Y)_{ik}
    
    Yt_w = Y.T @ w  # shape (n,)
    # Actually need matrix products
    # Let A = Y.T @ diag(w) @ Y, B = Y.T @ diag(w*f^2) @ Y
    # d2S2 = 2*A, d2S4 = 12*B
    
    # For efficiency, compute A and B
    # A[i,k] = sum_j Y_ji * w_j * Y_jk
    # B[i,k] = sum_j Y_ji * w_j * f_j^2 * Y_jk
    
    # Use matrix multiplication
    Ytw = Y.T * w[None, :]  # (n, N) * (N,) -> (n, N)
    A = Ytw @ Y  # (n, N) @ (N, n) -> (n, n)
    Ytwf2 = Y.T * (w * f2)[None, :]
    B = Ytwf2 @ Y

    d2S2 = 2.0 * A
    d2S4 = 12.0 * B

    # Hessian of S4/S2^2:
    # H = d2S4/S2^2 - 2*S4*(dS2 dS2^T)/S2^3 - 2*S4*d2S2/S2^3 + 4*S4^2*(dS2 dS2^T)/S2^4
    # Wait, let me re-derive. Let g = S4, h = S2^2. Then F = g/h.
    # dF = dg/h - g dh/h^2
    # d2F = d2g/h - 2 dg dh/h^2 - g d2h/h^2 + 2g (dh)^2/h^3
    # dh = 2 S2 dS2
    # d2h = 2 (dS2 dS2^T + S2 d2S2)
    
    dh = 2.0 * S2 * dS2  # shape (n,)
    d2h = 2.0 * (np.outer(dS2, dS2) + S2 * d2S2)  # shape (n, n)
    
    H = d2S4 / (S2 * S2) - 2.0 * np.outer(dS4, dh) / (S2 * S2 * S2) \
        - val * d2h / (S2 * S2) + 2.0 * val * np.outer(dh, dh) / (S2 * S2 * S2)
    
    return val, grad, H


def _project_to_sphere(c):
    """Project c onto the unit sphere."""
    nrm = np.linalg.norm(c)
    if nrm < 1e-15:
        return c
    return c / nrm


def _projected_newton(c0, Y, w, maxiter=2000, tol=1e-14):
    """Projected Newton method on the unit sphere."""
    c = c0.copy()
    n = len(c)
    
    for it in range(maxiter):
        if time.time() > _DEADLINE - 30.0:
            break
        
        val, grad, H = _ratio_val_grad_hess(c, Y, w)
        
        if not np.isfinite(val):
            break
        
        # Project gradient to tangent space: grad_tan = grad - (c . grad) * c
        c_dot_grad = np.dot(c, grad)
        grad_tan = grad - c_dot_grad * c
        
        # Check convergence
        if np.linalg.norm(grad_tan) < tol:
            break
        
        # Newton step in tangent space: solve (P H P) d = -grad_tan
        # where P = I - c c^T is the projector onto tangent space
        # P H P = H - c (c^T H) - (H c) c^T + c (c^T H c) c^T
        Hc = H @ c
        cTHc = np.dot(c, Hc)
        PH = H - np.outer(c, Hc) - np.outer(Hc, c) + cTHc * np.outer(c, c)
        
        # Solve PH d = -grad_tan
        try:
            d = np.linalg.solve(PH, -grad_tan)
        except np.linalg.LinAlgError:
            # Fallback to gradient descent
            d = -grad_tan / max(np.linalg.norm(grad_tan), 1e-15)
        
        # Line search: find alpha such that f(c + alpha d) < f(c) and c+alpha d is on sphere
        # Use backtracking
        alpha = 1.0
        c_new = c + alpha * d
        c_new = _project_to_sphere(c_new)
        
        for ls in range(50):
            c_new = c + alpha * d
            c_new = _project_to_sphere(c_new)
            val_new, _, _ = _ratio_val_grad_hess(c_new, Y, w)
            
            if val_new < val:
                break
            alpha *= 0.5
        
        if alpha < 1e-15:
            break
        
        c = c_new
    
    return c


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    
    best_c = np.zeros(n)
    best_val = np.inf

    # 1. Symmetric Initial Guesses
    inits = []
    # m=0 only
    v0 = np.zeros(n); v0[k] = 1.0; inits.append(v0)
    # m=10 only (cos)
    v10 = np.zeros(n); v10[2*k] = 1.0; inits.append(v10)
    # m=10 only (sin)
    v10s = np.zeros(n); v10s[k-10] = 1.0; inits.append(v10s)
    # Uniform
    v_u = np.ones(n); inits.append(v_u / np.linalg.norm(v_u))
    # Random
    rng = np.random.default_rng(123)
    for _ in range(20):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))

    # 2. Projected Newton from each init
    for x0 in inits:
        if time.time() > _DEADLINE - 100.0:
            break
        try:
            c_opt = _projected_newton(x0, Y, w, maxiter=1000, tol=1e-15)
            val_opt, _, _ = _ratio_val_grad_hess(c_opt, Y, w)
            if val_opt < best_val and np.isfinite(val_opt):
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            pass

    # 3. Additional random restarts if time permits
    if time.time() < _DEADLINE - 60.0:
        rng2 = np.random.default_rng(456)
        for _ in range(10):
            if time.time() > _DEADLINE - 40.0:
                break
            v = rng2.standard_normal(n)
            v = v / np.linalg.norm(v)
            try:
                c_opt = _projected_newton(v, Y, w, maxiter=500, tol=1e-15)
                val_opt, _, _ = _ratio_val_grad_hess(c_opt, Y, w)
                if val_opt < best_val and np.isfinite(val_opt):
                    best_val = val_opt
                    best_c = c_opt.copy()
            except Exception:
                pass

    # 4. Final high-precision polish
    if time.time() < _DEADLINE - 20.0:
        try:
            c_polished = _projected_newton(best_c, Y, w, maxiter=5000, tol=1e-16)
            val_polished, _, _ = _ratio_val_grad_hess(c_polished, Y, w)
            if val_polished < best_val and np.isfinite(val_polished):
                best_val = val_polished
                best_c = c_polished.copy()
        except Exception:
            pass

    # Normalize and validate
    out = best_c.astype(np.float64)
    nrm = np.linalg.norm(out)
    if nrm < 1e-12:
        out = np.ones(n, dtype=np.float64)
    else:
        out = out / nrm

    if out.shape != (21,):
        out = np.ones(21, dtype=np.float64)
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(out))
    if max_abs < 1e-3:
        out = out / max_abs if max_abs > 0 else np.ones(21)
    if max_abs > 1e5:
        out = out / max_abs

    return out
