"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_grid(pts)
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            max_cr = max(max_cr, cr)
            continue
        u = v / norm_v
        n_face = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n_face)
        if norm_n < 1e-12:
            continue
        n_face = n_face / norm_n
        if np.dot(n_face, a) < 0:
            n_face = -n_face
        if np.dot(u, n_face) > 0:
            circum_center = u
        else:
            circum_center = -u
        d_a = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(circum_center, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(circum_center, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        max_cr = max(max_cr, cr)
    return max_cr

def _covering_radius_grid(pts, n_grid=120):
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _repulsion_energy(pts):
    pts = _normalize(pts)
    n = len(pts)
    if n < 2:
        return 0.0
    dots = pts @ pts.T
    dots = np.clip(dots, -1, 1)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, 1.0)
    p = 6
    energy = np.sum(1.0 / (dists ** p))
    return energy

def _repulsion_gradient(pts):
    pts = _normalize(pts)
    n = len(pts)
    grad = np.zeros_like(pts)
    if n < 2:
        return grad
    dots = pts @ pts.T
    dots = np.clip(dots, -1, 1)
    dists = np.arccos(dots)
    p = 6
    sin_dists = np.sqrt(np.maximum(1.0 - dots**2, 1e-12))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-6:
                continue
            sin_d = sin_dists[i, j]
            dot_ij = dots[i, j]
            vec = pts[j] - dot_ij * pts[i]
            grad_d = -vec / sin_d
            factor = -p * (d ** (-p - 1))
            grad[i] += factor * grad_d
    for i in range(n):
        grad[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
    return grad

def _soft_max_covering_radius(pts, beta=50.0):
    pts = _normalize(pts)
    n = len(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_grid(pts), np.zeros_like(pts)
    crs = np.zeros(len(hull.simplices))
    circumcenters = []
    simplices_list = []
    for i, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            crs[i] = cr
            circumcenters.append(None)
            simplices_list.append(simplex)
            continue
        u = v / norm_v
        n_face = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n_face)
        if norm_n < 1e-12:
            crs[i] = 0.0
            circumcenters.append(None)
            simplices_list.append(simplex)
            continue
        n_face = n_face / norm_n
        if np.dot(n_face, a) < 0:
            n_face = -n_face
        if np.dot(u, n_face) > 0:
            cc = u
        else:
            cc = -u
        d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        crs[i] = cr
        circumcenters.append(cc)
        simplices_list.append(simplex)
    max_cr = np.max(crs)
    if max_cr < 1e-10:
        return 0.0, np.zeros_like(pts)
    shifted = beta * (crs - max_cr)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_val = max_cr + log_sum / beta
    exp_shifted = np.exp(shifted)
    total_exp = np.sum(exp_shifted)
    if total_exp < 1e-30:
        return soft_val, np.zeros_like(pts)
    weights = exp_shifted / total_exp
    grad = np.zeros_like(pts)
    for i, (simplex, cc, cr) in enumerate(zip(simplices_list, circumcenters, crs)):
        if cc is None:
            continue
        w = weights[i]
        if w < 1e-10:
            continue
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
        max_d = max(d_a, d_b, d_c)
        def grad_arccos_cc_x(cc, x):
            dot_cx = np.dot(cc, x)
            dot_cx = np.clip(dot_cx, -1, 1)
            sin_term = np.sqrt(max(1.0 - dot_cx**2, 1e-12))
            vec = cc - dot_cx * x
            return -vec / sin_term
        if d_a >= max_d - 1e-10:
            g_a = grad_arccos_cc_x(cc, a)
            g_a = g_a - np.dot(g_a, a) * a
            grad[simplex[0]] += w * g_a
        if d_b >= max_d - 1e-10:
            g_b = grad_arccos_cc_x(cc, b)
            g_b = g_b - np.dot(g_b, b) * b
            grad[simplex[1]] += w * g_b
        if d_c >= max_d - 1e-10:
            g_c = grad_arccos_cc_x(cc, c)
            g_c = g_c - np.dot(g_c, c) * c
            grad[simplex[2]] += w * g_c
    for i in range(n):
        grad[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
    return soft_val, grad

def _polish_facets(pts, max_iter=300, step=0.15):
    pts = _normalize(pts.copy())
    n = len(pts)
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        weights_sum = np.zeros(n)
        vec_sum = np.zeros((n, 3))
        max_cr_global = 0.0
        facet_data = []
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            u = v / norm_v
            n_face = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n_face)
            if norm_n < 1e-12:
                continue
            n_face = n_face / norm_n
            if np.dot(n_face, a) < 0:
                n_face = -n_face
            if np.dot(u, n_face) > 0:
                cc = u
            else:
                cc = -u
            d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
            cr = max(d_a, d_b, d_c)
            max_cr_global = max(max_cr_global, cr)
            facet_data.append((simplex, cc, cr))
        if max_cr_global < 1e-7:
            break
        beta = 8.0
        for simplex, cc, cr in facet_data:
            w = math.exp(beta * (cr - max_cr_global))
            for idx in simplex:
                weights_sum[idx] += w
                vec_sum[idx] += w * cc
        moved = False
        for i in range(n):
            if weights_sum[i] > 1e-10:
                target = vec_sum[i] / weights_sum[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target = target / norm_t
                    vec = target - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    norm_vec = np.linalg.norm(vec_tan)
                    if norm_vec > 1e-10:
                        pts[i] = pts[i] + step * vec_tan
                        moved = True
        if not moved:
            break
        pts = _normalize(pts)
        step *= 0.97
        if step < 1e-8:
            break
    return pts

def _lloyd_iteration(pts, n_iter=200, step=0.1):
    pts = _normalize(pts.copy())
    n = len(pts)
    for it in range(n_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        weights_sum = np.zeros(n)
        vec_sum = np.zeros((n, 3))
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            u = v / norm_v
            n_face = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n_face)
            if norm_n < 1e-12:
                continue
            n_face = n_face / norm_n
            if np.dot(n_face, a) < 0:
                n_face = -n_face
            if np.dot(u, n_face) > 0:
                cc = u
            else:
                cc = -u
            for idx in simplex:
                weights_sum[idx] += 1.0
                vec_sum[idx] += cc
        for i in range(n):
            if weights_sum[i] > 1e-10:
                target = vec_sum[i] / weights_sum[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target = target / norm_t
                    vec = target - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    pts[i] = pts[i] + step * vec_tan
        pts = _normalize(pts)
        step *= 0.95
        if step < 1e-8:
            break
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    np.random.seed(42)
    initializations = []
    initializations.append(_fibonacci_points(N))
    for _ in range(5):
        noise = 0.5 * np.random.randn(N, 3)
        pts = _normalize(_fibonacci_points(N) + noise)
        initializations.append(pts)
    for _ in range(3):
        pts = np.random.randn(N, 3)
        pts = _normalize(pts)
        initializations.append(pts)
    for init_pts in initializations:
        if time.time() > deadline:
            break
        pts = init_pts.copy()
        x0 = pts.flatten()
        def objective(x):
            p = x.reshape(N, 3)
            p = _normalize(p)
            return _repulsion_energy(p)
        def gradient(x):
            p = x.reshape(N, 3)
            p = _normalize(p)
            g = _repulsion_gradient(p)
            return g.flatten()
        try:
            res = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            pts = _normalize(res.x.reshape(N, 3))
        except Exception:
            pass
        if time.time() < deadline:
            x1 = pts.flatten()
            def soft_obj(x):
                p = x.reshape(N, 3)
                p = _normalize(p)
                val, _ = _soft_max_covering_radius(p, beta=50.0)
                return val
            def soft_grad(x):
                p = x.reshape(N, 3)
                p = _normalize(p)
                _, g = _soft_max_covering_radius(p, beta=50.0)
                return g.flatten()
            try:
                res2 = minimize(
                    soft_obj,
                    x1,
                    jac=soft_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts = _normalize(res2.x.reshape(N, 3))
            except Exception:
                pass
        if time.time() < deadline:
            pts = _polish_facets(pts, max_iter=200, step=0.15)
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    if time.time() < deadline:
        best_pts = _polish_facets(best_pts, max_iter=300, step=0.1)
        best_cr = _covering_radius(best_pts)
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
