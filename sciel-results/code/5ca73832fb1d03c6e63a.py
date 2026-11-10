import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0


def _normalize(points):
    pts = np.asarray(points, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _facet_alphas(points):
    """Return per-facet angular circumradius (radians). Returns [] if hull fails."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    try:
        simplices = hull.simplices
    except Exception:
        return []
    if simplices.size == 0:
        return []
    tri = points[simplices]
    p0 = tri[:, 0, :]
    p1 = tri[:, 1, :]
    p2 = tri[:, 2, :]
    nrm = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(nrm, axis=1)
    mask = nn > 1e-12
    if not np.any(mask):
        return []
    d = np.abs(nrm[mask] @ p0[mask].T) / nn[mask]
    d = np.clip(d, 0.0, 1.0)
    return np.arccos(d)


def _covering_radius(points):
    """Exact covering radius in radians (max facet circumradius)."""
    alphas = _facet_alphas(points)
    if alphas.size == 0:
        return np.pi
    return float(np.max(alphas))


def _softmax(values, beta):
    if values.size == 0:
        return np.pi
    m = np.max(values)
    return m + np.log(np.sum(np.exp(values - m))) / beta


def _smooth_obj(params, beta):
    pts = _normalize(params.reshape(N, 3))
    if not np.all(np.isfinite(pts)):
        return 1e6
    alphas = _facet_alphas(pts)
    if alphas.size == 0:
        return 1e6
    return _softmax(alphas, beta)


def _smooth_obj_grad(params, beta):
    """Compute gradient of softmax surrogate w.r.t. Cartesian coordinates.
    
    The covering radius is determined by the facet with maximum angular circumradius.
    For a facet with vertices a, b, c and normal n, the circumradius alpha satisfies
    cos(alpha) = |n·a| / |n|. The gradient of alpha w.r.t. each vertex involves
    how moving that vertex changes the plane and thus the distance from origin.
    """
    pts = _normalize(params.reshape(N, 3))
    if not np.all(np.isfinite(pts)):
        return 1e6, np.zeros_like(params)
    alphas = _facet_alphas(pts)
    if alphas.size == 0:
        return 1e6, np.zeros_like(params)
    obj = _softmax(alphas, beta)

    m = np.max(alphas)
    w = np.exp(alphas - m) / (beta * np.sum(np.exp(alphas - m)))

    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
    except Exception:
        return obj, np.zeros_like(params)

    grad = np.zeros((N, 3))
    for i, simp in enumerate(simplices):
        wi = w[i]
        if wi <= 1e-10:
            continue
        a, b, c = pts[simp]
        v1 = b - a
        v2 = c - a
        nrm = np.cross(v1, v2)
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        n = nrm / nn
        d = np.abs(n @ a)
        d = np.clip(d, 0.0, 1.0)
        denom = np.sqrt(max(1.0 - d * d, 1e-14))
        s = wi / denom
        
        # Gradient of alpha w.r.t. vertex position:
        # alpha = arccos(d), d = |n·a|
        # d(alpha)/d(a) = -1/sqrt(1-d^2) * d(d)/d(a)
        # For vertex a: moving a changes both n and the plane offset
        # Simplified: the dominant effect is through the normal direction
        # The gradient pushes vertices away from the facet normal direction
        # to increase the distance from origin to the plane
        grad[simp[0]] += -s * n
        grad[simp[1]] += -s * n
        grad[simp[2]] += -s * n
    
    return obj, grad.flatten()


def _repulsive_relaxation(points, steps=300, alpha=0.05, time_limit=None, start_time=None):
    """Repulsive force relaxation with adaptive step size."""
    if start_time is None:
        start_time = time.time()
    pts = points.copy()
    n = pts.shape[0]
    for _ in range(steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, 1.0)
        inv_d3 = 1.0 / (d2 * np.sqrt(d2 + 1e-15))
        forces = np.einsum('ij,ijk->ik', inv_d3, diff)
        # Adaptive: reduce step if forces are large
        force_mag = np.linalg.norm(forces, axis=1)
        max_f = np.max(force_mag) if force_mag.size > 0 else 1.0
        step = alpha / max(1.0, max_f * 0.1)
        pts = _normalize(pts + step * forces)
    return pts


def _fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def _spherical_to_cartesian(params):
    theta = params[0::2]
    phi = params[1::2]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def _cartesian_to_spherical(pts):
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    params = np.zeros(2 * N)
    params[0::2] = theta
    params[1::2] = phi
    return params


def construct():
    start = time.time()
    deadline = start + _DEADLINE

    def remaining():
        return deadline - time.time()

    best = _fibonacci()
    best_r = _covering_radius(best)

    def time_ok(budget=0.5):
        return remaining() > budget

    def local_optimize(pts, budget):
        nonlocal best, best_r
        pts = _normalize(pts)
        r0 = _covering_radius(pts)
        if r0 < best_r:
            best = pts.copy()
            best_r = r0

        # Phase A: gradient-based on smooth surrogate (coarse -> fine)
        for beta, maxiter in ((20.0, 250), (35.0, 350), (55.0, 450)):
            if not time_ok(2.0) or (time.time() - start) > budget:
                break
            try:
                res = minimize(_smooth_obj, pts.flatten(), method='L-BFGS-B',
                               jac=lambda x: _smooth_obj_grad(x, beta)[1],
                               options={'maxiter': maxiter, 'gtol': 1e-7,
                                        'ftol': 1e-9, 'maxfun': 1500, 'disp': False})
                if np.all(np.isfinite(res.x)):
                    cand = _normalize(res.x.reshape(N, 3))
                    rc = _covering_radius(cand)
                    if rc < best_r:
                        best = cand
                        best_r = rc
                    if rc < r0:
                        r0 = rc
                        pts = cand
            except Exception:
                pass

        # Phase B: Nelder-Mead in spherical coordinates with exact objective
        if time_ok(2.0) and (time.time() - start) < budget:
            try:
                params0 = _cartesian_to_spherical(best)
                
                def sphere_obj(p):
                    if time.time() > deadline:
                        return 1e6
                    pts_c = _spherical_to_cartesian(p)
                    if not np.all(np.isfinite(pts_c)):
                        return 1e6
                    return _covering_radius(pts_c)
                
                res = minimize(sphere_obj, params0, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11,
                                        'adaptive': True})
                if np.all(np.isfinite(res.x)) and res.fun < best_r:
                    best = _normalize(_spherical_to_cartesian(res.x))
                    best_r = float(res.fun)
            except Exception:
                pass
        return best_r

    # 1. Initial relaxation + first optimization
    if time_ok(1.0):
        rel = _repulsive_relaxation(best, steps=1200, alpha=0.04,
                                    time_limit=5.0, start_time=start)
        rr = _covering_radius(rel)
        if rr < best_r:
            best = rel
            best_r = rr
    local_optimize(best, budget=22.0)

    # 2. Adaptive random restarts with multiple perturbation scales
    stall = 0
    scale = 0.4
    max_stall = 10
    while time_ok(15.0):
        # Try different noise scales
        for s in [scale, scale * 0.6, scale * 1.5]:
            if not time_ok(10.0):
                break
            noise = np.random.normal(0.0, s, size=best.shape)
            pert = _normalize(best + noise)
            rel = _repulsive_relaxation(pert, steps=200, alpha=0.05,
                                        time_limit=2.0, start_time=start)
            rr = _covering_radius(rel)
            if rr < best_r:
                best = rel
                best_r = rr
                stall = 0
                scale = max(0.06, scale * 0.8)
                local_optimize(best, budget=10.0)
                break
            else:
                stall += 1
        else:
            # No improvement in this round
            scale = min(0.8, scale * 1.2)
            if stall >= max_stall:
                break

    # 3. Final tight polish in spherical coordinates
    if time_ok(8.0):
        try:
            params0 = _cartesian_to_spherical(best)
            
            def sphere_obj_final(p):
                if time.time() > deadline:
                    return 1e6
                pts_c = _spherical_to_cartesian(p)
                if not np.all(np.isfinite(pts_c)):
                    return 1e6
                return _covering_radius(pts_c)
            
            res = minimize(sphere_obj_final, params0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_r:
                best = _normalize(_spherical_to_cartesian(res.x))
                best_r = float(res.fun)
        except Exception:
            pass

    # 4. One more gradient polish if time permits
    if time_ok(5.0):
        try:
            res = minimize(_smooth_obj, best.flatten(), method='L-BFGS-B',
                           jac=lambda x: _smooth_obj_grad(x, 80.0)[1],
                           options={'maxiter': 300, 'gtol': 1e-9,
                                    'ftol': 1e-12, 'maxfun': 1000, 'disp': False})
            if np.all(np.isfinite(res.x)):
                cand = _normalize(res.x.reshape(N, 3))
                rc = _covering_radius(cand)
                if rc < best_r:
                    best = cand
                    best_r = rc
        except Exception:
            pass

    return best
