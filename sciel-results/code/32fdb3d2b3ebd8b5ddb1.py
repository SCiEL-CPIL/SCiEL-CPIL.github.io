"""Spherical-harmonic critical-point method for minimizing log-energy of 150 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv

N = 150
MAX_DEG = 8


def _add_l_m(pts, l, m):
    """Add a band-limited (degree <= MAX_DEG) harmonic term with random coefficients."""
    for ell in range(1, MAX_DEG + 1):
        for mm in range(-ell, ell + 1):
            if np.random.random() < 0.3:
                pts["a_re"][ell, mm] = np.random.randn() * 0.5
                pts["a_im"][ell, mm] = np.random.randn() * 0.5
    return pts


def _spherical_harmonic(pts, l, m, theta, phi):
    """Evaluate Y_l^m(theta, phi) for m possibly negative. theta=polar, phi=azimuth."""
    import math
    if m < 0:
        norm = (-1) ** m * math.sqrt((2 * l + 1) / (4 * math.pi) *
                                      math.factorial(l + m) / math.factorial(l - m))
        Y = norm * lpmv(abs(m), l, np.cos(theta)) * np.exp(1j * m * phi)
    else:
        norm = math.sqrt((2 * l + 1) / (4 * math.pi) *
                         math.factorial(l - m) / math.factorial(l + m))
        Y = norm * lpmv(m, l, np.cos(theta)) * np.exp(1j * m * phi)
    return Y


def _eval_harmonic(pts, theta, phi):
    """Real-valued band-limited function value at (theta, phi) arrays."""
    val = np.zeros_like(theta, dtype=float)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    for ell in range(1, MAX_DEG + 1):
        for mm in range(-ell, ell + 1):
            are = pts["a_re"][ell, mm]
            aim = pts["a_im"][ell, mm]
            if are == 0.0 and aim == 0.0:
                continue
            Y = _spherical_harmonic(pts, ell, mm, theta, phi)
            val += are * Y.real + aim * Y.imag
    return val


def _grad_harmonic(pts, theta, phi):
    """Gradient (dtheta, dphi) of the band-limited function at (theta, phi) arrays."""
    import math
    gtheta = np.zeros_like(theta, dtype=float)
    gphi = np.zeros_like(theta, dtype=float)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    for ell in range(1, MAX_DEG + 1):
        for mm in range(-ell, ell + 1):
            are = pts["a_re"][ell, mm]
            aim = pts["a_im"][ell, mm]
            if are == 0.0 and aim == 0.0:
                continue
            Y = _spherical_harmonic(pts, ell, mm, theta, phi)
            # Compute derivative w.r.t. theta using associated Legendre derivative identity
            abs_mm = abs(mm)
            # d/dtheta [P_l^m(cos(theta))] = -sin(theta) * d/dx P_l^m(x) at x=cos(theta)
            # Using: d/dx P_l^m(x) = (l*x*P_l^m(x) - (l+m)*P_{l-1}^m(x)) / (x^2 - 1)
            # But we can use: P_l^{m+1}(x) = (2l+1)x P_l^m(x) - (l+m)(l-m+1) P_{l+1}^m(x)
            # Simpler: use numerical or the identity dP_l^m/dtheta = -sin(theta) * dP_l^m/dx
            # Let's use the recurrence: d/dtheta P_l^m(cos theta) = -sin(theta) * [l*P_l^m(cos theta)/sin(theta) - P_{l-1}^m(cos theta)/sin(theta)]
            # Actually: dP_l^m(x)/dx = (l*x*P_l^m(x) - (l+m)*P_{l-1}^m(x)) / (x^2-1) for x != ±1
            # So dP_l^m/dtheta = -sin(theta) * dP_l^m/dx = -sin(theta) * (l*cos(theta)*P_l^m - (l+m)*P_{l-1}^m) / (cos^2(theta)-1)
            # = -sin(theta) * (l*cos(theta)*P_l^m - (l+m)*P_{l-1}^m) / (-sin^2(theta))
            # = (l*cos(theta)*P_l^m - (l+m)*P_{l-1}^m) / sin(theta)
            if sin_t > 1e-10:
                P_l = lpmv(abs_mm, ell, cos_t)
                if ell > 0:
                    P_lm1 = lpmv(abs_mm, ell - 1, cos_t) if ell >= 1 else np.zeros_like(cos_t)
                    dP_dtheta = (ell * cos_t * P_l - (ell + abs_mm) * P_lm1) / sin_t
                else:
                    dP_dtheta = np.zeros_like(cos_t)
            else:
                dP_dtheta = np.zeros_like(cos_t)
            
            if mm < 0:
                norm = (-1) ** mm * math.sqrt((2 * ell + 1) / (4 * math.pi) *
                                              math.factorial(ell + mm) / math.factorial(ell - mm))
            else:
                norm = math.sqrt((2 * ell + 1) / (4 * math.pi) *
                                 math.factorial(ell - mm) / math.factorial(ell + mm))
            
            Y_t = norm * dP_dtheta * np.exp(1j * mm * phi)
            Y_p = 1j * mm * Y
            gtheta += are * Y_t.real + aim * Y_t.imag
            gphi += are * Y_p.real + aim * Y_p.imag
    return gtheta, gphi


def _fibonacci_grid(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])


def _to_sph(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    r = np.maximum(r, 1e-12)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi


def _log_energy(pts):
    """Compute log-energy: sum_{i<j} log(1/|x_i - x_j|)"""
    d2 = np.sum(pts * pts, axis=1)[:, None] + np.sum(pts * pts, axis=1)[None, :] - 2 * pts @ pts.T
    d2 = np.maximum(d2, 1e-18)
    d = np.sqrt(d2)
    iu = np.triu_indices(N, k=1)
    return np.sum(np.log(1.0 / d[iu]))


def _log_energy_grad(pts):
    """Compute gradient of log-energy w.r.t. point coordinates.
    d/dx_i sum_{j!=i} log(1/|x_i-x_j|) = sum_{j!=i} (x_i - x_j) / |x_i-x_j|^2
    """
    d2 = np.sum(pts * pts, axis=1)[:, None] + np.sum(pts * pts, axis=1)[None, :] - 2 * pts @ pts.T
    d2 = np.maximum(d2, 1e-18)
    d = np.sqrt(d2)
    # Gradient for each point i: sum_j (x_i - x_j) / d_ij^2 for j != i
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            grad[i] += diff / d2[i, j]
    return grad


def _find_critical_points(pts, n_init=2000):
    """Find local minima of the band-limited potential via Newton refinement."""
    theta, phi = _to_sph(_fibonacci_grid(n_init, offset=0.3))
    cands = []
    for t0, p0 in zip(theta, phi):
        try:
            res = minimize(
                lambda v: _eval_harmonic(pts, v[0:1], v[1:1])[0],
                x0=[t0, p0],
                method="L-BFGS-B",
                bounds=[(1e-4, np.pi - 1e-4), (-np.pi, np.pi)],
                options={"maxiter": 20, "ftol": 1e-10, "gtol": 1e-6},
            )
            if res.success or res.fun < 0:
                cands.append(res.x)
        except Exception:
            continue
    if len(cands) < N:
        return None
    cands = np.array(cands)
    # deduplicate close candidates
    cands = cands[np.argsort(cands[:, 0])]
    keep = [0]
    for i in range(1, len(cands)):
        dt = cands[i, 0] - cands[keep[-1], 0]
        dp = np.arctan2(np.sin(cands[i, 1] - cands[keep[-1], 1]),
                        np.cos(cands[i, 1] - cands[keep[-1], 1]))
        if np.hypot(dt, dp) > 0.05:
            keep.append(i)
    cands = cands[keep]
    if len(cands) < N:
        return None
    # pick N most negative-potential candidates
    vals = np.array([_eval_harmonic(pts, c[0:1], c[1:1])[0] for c in cands])
    idx = np.argsort(vals)[:N]
    cands = cands[idx]
    th = cands[:, 0]
    ph = cands[:, 1]
    xyz = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return xyz


def _refine_log_energy(xyz):
    """Local refinement of log-energy using analytical gradient in Cartesian coords."""
    xyz = xyz.copy()
    
    def energy(xyz):
        return _log_energy(xyz)
    
    def grad(xyz):
        return _log_energy_grad(xyz)
    
    # Optimize in Cartesian space with projection back to sphere
    best_x = xyz.copy()
    best_e = energy(best_x)
    
    for iteration in range(200):
        g = grad(best_x)
        # Project gradient onto tangent space: g_tangent = g - (g·x_hat)x_hat
        x_norm = np.linalg.norm(best_x, axis=1, keepdims=True)
        x_hat = best_x / np.maximum(x_norm, 1e-12)
        g_proj = g - np.sum(g * x_hat, axis=1, keepdims=True) * x_hat
        
        # Line search
        step = 0.01
        improved = False
        for _ in range(20):
            new_x = best_x + step * g_proj
            new_x = new_x / np.linalg.norm(new_x, axis=1, keepdims=True)
            new_e = energy(new_x)
            if new_e < best_e:
                best_x = new_x
                best_e = new_e
                improved = True
                break
            step *= 0.5
        
        if not improved or step < 1e-10:
            break
    
    return best_x


def construct():
    np.random.seed(12345)
    best = None
    
    for attempt in range(5):
        # Build a random band-limited potential
        pts = {"a_re": np.zeros((MAX_DEG + 1, 2 * MAX_DEG + 1)),
               "a_im": np.zeros((MAX_DEG + 1, 2 * MAX_DEG + 1))}
        _add_l_m(pts, 0, 0)
        
        # Find critical points
        xyz = _find_critical_points(pts)
        if xyz is None:
            continue
        
        # Refine
        xyz = _refine_log_energy(xyz)
        e = _log_energy(xyz)
        if best is None or e < best[0]:
            best = (e, xyz)
        
        # Also try direct refinement from fibonacci as baseline
        fib = _fibonacci_grid(N, offset=0.5)
        fib_refined = _refine_log_energy(fib)
        fib_e = _log_energy(fib_refined)
        if best is None or fib_e < best[0]:
            best = (fib_e, fib_refined)
    
    if best is None:
        return _fibonacci_grid(N, offset=0.5)

    xyz = best[1]
    # Normalize
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    # Validate: no points closer than 1e-9
    d2 = np.sum(xyz * xyz, axis=1)[:, None] + np.sum(xyz * xyz, axis=1)[None, :] - 2 * xyz @ xyz.T
    d = np.sqrt(np.maximum(d2, 0.0))
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 1e-9:
        return _fibonacci_grid(N, offset=0.5)
    return xyz
