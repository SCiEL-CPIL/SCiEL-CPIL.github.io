# EVOLVE-BLOCK-START
import numpy as np


def _norms(n, m):
    """N_m = sqrt( (2n+1)/(4 pi) * (n-m)!/(n+m)! ) for |m| <= n."""
    m = abs(int(m))
    return np.sqrt((2 * n + 1) / (4.0 * np.pi) * float(np.math.factorial(n - m)) /
                   float(np.math.factorial(n + m)))


def _assoc_legendre(n, m, x):
    """Associated Legendre P_n^m(x) with Condon-Shortley phase, m >= 0, n >= m."""
    x = np.asarray(x, dtype=np.float64)
    Pmm = np.ones_like(x)
    if m > 0:
        fact = 1.0
        sq = np.sqrt(np.maximum(0.0, (1.0 - x) * (1.0 + x)))
        for i in range(1, m + 1):
            Pmm *= -fact * sq
            fact += 2.0
    if n == m:
        return Pmm
    Pmp1 = x * (2 * m + 1) * Pmm
    if n == m + 1:
        return Pmp1
    Pmm_prev = Pmm
    Pm = Pmp1
    for i in range(m + 2, n + 1):
        Pmm_prev, Pm = Pm, ((2 * i - 1) * x * Pm - (i + m - 1) * Pmm_prev) / (i - m)
    return Pm


def _build_quadrature(n_theta=70, n_phi=130):
    """Gauss-Legendre in cos(theta) x equally spaced phi. Returns weights, cos_t, sin_t, phi."""
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    cos_t = np.repeat(x, n_phi)
    phi_v = np.tile(phi, n_theta)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    sin_t = np.repeat(sin_t, n_phi)
    wpts = np.repeat(w * (2.0 * np.pi / n_phi), n_theta)
    return wpts, cos_t, sin_t, phi_v


def _make_Y(wpts, cos_t, sin_t, phi_v, n=10):
    """Build real orthonormal spherical harmonics matrix."""
    Npts = cos_t.shape[0]
    cos_phi = np.cos(phi_v)
    sin_phi = np.sin(phi_v)
    Y = np.zeros((2 * n + 1, Npts), dtype=np.float64)
    for m in range(0, n + 1):
        Pm = _assoc_legendre(n, m, cos_t)
        Nm = _norms(n, m)
        idx = n + m
        if m == 0:
            Y[idx, :] = Nm * Pm
        else:
            Y[idx, :] = np.sqrt(2.0) * Nm * Pm * cos_phi
            Y[n - m, :] = np.sqrt(2.0) * Nm * Pm * sin_phi
    return Y


def _objective_and_grad(c, Y, wpts):
    """Return (R, grad_R) where R = ||f||_4/||f||_2, f = Y^T c."""
    f = Y.T @ c
    f2 = f * f
    f4 = f2 * f2
    mass = wpts.sum()
    L2sq = (wpts * f2).sum() / mass
    L4 = (wpts * f4).sum() / mass
    L2 = np.sqrt(max(L2sq, 1e-300))
    L4v = L4 ** 0.25
    R = L4v / L2
    coeff = f ** 3 / (L2 * L4v) - L4v * f / (L2 * L2 * L2sq)
    g = coeff * Y
    grad = Y.T @ g
    return R, grad


def _hessian_vec(c, Y, wpts, v):
    """Compute Hessian-vector product via central finite differences of gradient."""
    eps = 1e-5
    _, g_plus = _objective_and_grad(c + eps * v, Y, wpts)
    _, g_minus = _objective_and_grad(c - eps * v, Y, wpts)
    return (g_plus - g_minus) / (2 * eps)


def _trust_region_step(c, Y, wpts, delta, max_inner=15):
    """
    Compute a trust-region step on the unit sphere using CG on projected Hessian.
    Returns tangent step direction.
    """
    R, g = _objective_and_grad(c, Y, wpts)
    g_tan = g - np.dot(g, c) * c
    if np.linalg.norm(g_tan) < 1e-12:
        return np.zeros(len(c))
    
    p = -g_tan
    p = p - np.dot(p, c) * c
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-15:
        return np.zeros(len(c))
    p = p / p_norm
    
    r = -g_tan
    r = r - np.dot(r, c) * c
    r_norm_sq = np.dot(r, r)
    p_prev = p.copy()
    x_new = np.zeros(len(c))
    
    for k in range(max_inner):
        Hp = _hessian_vec(c, Y, wpts, p)
        Hp = Hp - np.dot(Hp, c) * c
        pHp = np.dot(p, Hp)
        if pHp < 1e-15:
            break
        alpha = r_norm_sq / pHp
        x_new += alpha * p
        x_new = x_new - np.dot(x_new, c) * c
        
        r_new = r - alpha * Hp
        r_new = r_new - np.dot(r_new, c) * c
        r_new_norm_sq = np.dot(r_new, r_new)
        
        if np.sqrt(r_new_norm_sq) < 1e-12 * np.sqrt(r_norm_sq):
            break
        
        beta = r_new_norm_sq / r_norm_sq
        p_new = r_new + beta * p_prev
        p_new = p_new - np.dot(p_new, c) * c
        p_new_norm = np.linalg.norm(p_new)
        if p_new_norm < 1e-15:
            break
        p_prev = p
        p = p_new / p_new_norm
        r = r_new
        r_norm_sq = r_new_norm_sq
    
    step = x_new
    step_norm = np.linalg.norm(step)
    if step_norm > delta:
        step = step * (delta / step_norm)
    return step


def _optimize(n=10, n_theta=70, n_phi=130, max_iter=200, tol=1e-10):
    wpts, cos_t, sin_t, phi_v = _build_quadrature(n_theta, n_phi)
    Y = _make_Y(wpts, cos_t, sin_t, phi_v, n)
    dim = 2 * n + 1
    rng = np.random.default_rng(42)
    best_c = None
    best_val = np.inf
    
    starts = []
    starts.append(np.ones(dim, dtype=np.float64) / np.sqrt(dim))
    a = np.zeros(dim); a[n] = 1.0; starts.append(a)
    b = np.zeros(dim); b[2 * n] = 1.0; starts.append(b)
    d = np.zeros(dim); d[0] = 1.0; starts.append(d)
    for _ in range(4):
        r = rng.standard_normal(dim)
        r /= np.linalg.norm(r)
        starts.append(r)
    
    for c0 in starts:
        c = c0 / np.linalg.norm(c0)
        delta = 1.0
        
        for it in range(max_iter):
            R, g = _objective_and_grad(c, Y, wpts)
            g_tan = g - np.dot(g, c) * c
            if np.linalg.norm(g_tan) < tol:
                break
            
            step = _trust_region_step(c, Y, wpts, delta, max_inner=15)
            step_norm = np.linalg.norm(step)
            
            if step_norm < 1e-15:
                delta *= 0.5
                if delta < 1e-12:
                    break
                continue
            
            c_trial = c + step
            c_trial_norm = np.linalg.norm(c_trial)
            if c_trial_norm < 1e-15:
                delta *= 0.5
                continue
            c_trial = c_trial / c_trial_norm
            
            R_trial, _ = _objective_and_grad(c_trial, Y, wpts)
            
            if R_trial < R:
                c = c_trial
                R = R_trial
                pred = np.dot(g_tan, step)
                actual = R - R_trial
                ratio = actual / max(pred, 1e-300)
                if ratio > 0.75:
                    delta = min(2.0 * delta, 2.0)
                elif ratio < 0.25:
                    delta = 0.5 * delta
            else:
                delta = 0.5 * delta
                if delta < 1e-12:
                    break
        
        R_final, _ = _objective_and_grad(c, Y, wpts)
        if R_final < best_val:
            best_val = R_final
            best_c = c.copy()
    
    if best_c is None:
        best_c = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    return best_c


def rudin_problem10() -> np.ndarray:
    try:
        c = _optimize(n=10, n_theta=70, n_phi=130)
    except Exception:
        c = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    if c.shape[0] != 21:
        c = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    if not np.all(np.isfinite(c)):
        c = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    mx = np.max(np.abs(c))
    if mx < 1e-3:
        c = c / max(mx, 1e-12) * 0.5
    elif mx > 1e5:
        c = c / mx * 0.5
    return c


# EVOLVE-BLOCK-END
