"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Structural change: Support-function parameterization of a convex body.
  - A strictly convex body is represented by its support function h(θ) = r0 + Σ a_k cos(kθ) + b_k sin(kθ)
  - 14 points are placed at equal angular intervals on this body
  - The body is always convex (for reasonable coefficients), guaranteeing all points are on the boundary
  - Optimization is performed on the smooth 14-dim coefficient space
  - L-BFGS-B with a smooth log-sum-exp objective for fast, precise convergence
  - Multiple restarts from diverse initial configurations
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14
K = 7  # number of Fourier modes

# Pre-compute all triangle index combinations
tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)

# Pre-compute the Fourier basis matrix
# For each of N points at angle θ_i, we need cos(k*θ_i) and sin(k*θ_i) for k=1..K
_theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
_basis = np.zeros((N, 2 * K))
for k in range(1, K + 1):
    _basis[:, 2*(k-1)] = np.cos(k * _theta)
    _basis[:, 2*(k-1)+1] = np.sin(k * _theta)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_triangle_areas(points):
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_soft(flat, beta=80.0):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min


def _decode_support(params):
    """
    Decode support function parameters to N points.
    params: [r0, a1, b1, a2, b2, ..., aK, bK]  (1 + 2*K = 15 params)
    
    The support function is h(θ) = r0 + Σ_{k=1}^{K} [a_k cos(kθ) + b_k sin(kθ)]
    A point on the boundary in direction θ is:
      x = h(θ) cos(θ) - h'(θ) sin(θ)
      y = h(θ) sin(θ) + h'(θ) cos(θ)
    where h'(θ) = Σ_{k=1}^{K} k * [-a_k sin(kθ) + b_k cos(kθ)]
    """
    r0 = params[0]
    coeffs = params[1:]  # 2*K values: a1, b1, a2, b2, ...
    
    h_vals = np.full(N, r0)
    hprime_vals = np.zeros(N)
    
    for k in range(1, K + 1):
        a = coeffs[2*(k-1)]
        b = coeffs[2*(k-1)+1]
        h_vals += a * np.cos(k * _theta) + b * np.sin(k * _theta)
        hprime_vals += k * (-a * np.sin(k * _theta) + b * np.cos(k * _theta))
    
    x = h_vals * np.cos(_theta) - hprime_vals * np.sin(_theta)
    y = h_vals * np.sin(_theta) + hprime_vals * np.cos(_theta)
    
    pts = np.column_stack([x, y])
    pts -= pts.mean(axis=0)
    
    return pts


def _objective_support(params, beta=80.0):
    """Smooth objective in support-function parameter space."""
    pts = _decode_support(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min


def _objective_support_hard(params):
    pts = _decode_support(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _generate_support_inits():
    """Generate diverse initial support-function parameter vectors."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Circle (r0=1, all coefficients 0)
    p = np.zeros(1 + 2*K)
    p[0] = 1.0
    inits.append(p.copy())
    
    # 2. Ellipses
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.3, 0.7), (0.7, 1.3), (1.1, 0.9), (1.0, 0.6), (0.6, 1.0)]:
        p = np.zeros(1 + 2*K)
        p[0] = (rx + ry) / 2
        # For an ellipse, h(θ) = sqrt(rx^2 cos^2 θ + ry^2 sin^2 θ)
        # Approximate with Fourier: h ≈ r0 + (rx-ry)/2 * cos(2θ)
        p[2] = (rx - ry) / 2  # a2
        inits.append(p.copy())
    
    # 3. Perturbed circles
    for js in [0.02, 0.05, 0.1, 0.15, 0.2]:
        p = np.zeros(1 + 2*K)
        p[0] = 1.0
        p[1:] = rng.normal(0, js, size=2*K)
        inits.append(p.copy())
    
    # 4. Random smooth shapes
    for seed in range(20):
        rs = np.random.default_rng(seed * 31 + 17)
        p = np.zeros(1 + 2*K)
        p[0] = rs.uniform(0.8, 1.2)
        # Decay coefficients for higher modes to keep it smooth/convex
        for k in range(1, K+1):
            scale = 0.15 / k
            p[2*(k-1)] = rs.normal(0, scale)
            p[2*(k-1)+1] = rs.normal(0, scale)
        inits.append(p.copy())
    
    # 5. Shapes with dominant low-frequency modes
    for k_dom in [2, 3, 4]:
        for amp in [0.1, 0.2, 0.3]:
            p = np.zeros(1 + 2*K)
            p[0] = 1.0
            p[2*(k_dom-1)] = amp
            inits.append(p.copy())
            p[2*(k_dom-1)+1] = amp
            inits.append(p.copy())
    
    # 6. Larger perturbations
    for seed in range(15):
        rs = np.random.default_rng(seed * 43 + 29)
        p = np.zeros(1 + 2*K)
        p[0] = rs.uniform(0.7, 1.3)
        for k in range(1, K+1):
            scale = 0.25 / k
            p[2*(k-1)] = rs.normal(0, scale)
            p[2*(k-1)+1] = rs.normal(0, scale)
        inits.append(p.copy())
    
    return inits


def _nm(flat0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None


def _powell(flat0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None


def _lbfgs(flat0, deadline, obj, max_iter=3000):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(obj, flat0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        return res
    except Exception:
        return None


def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points


def _hull_pull(points, alpha=0.3):
    """Expand convex hull outward to enlarge hull area."""
    try:
        hull = ConvexHull(points)
        hull_idx = set(hull.vertices.tolist())
    except Exception:
        return points.copy()
    centroid = points.mean(axis=0)
    new_pts = points.copy()
    for idx in hull_idx:
        v = points[idx] - centroid
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            new_pts[idx] = points[idx] + alpha * v
    return new_pts


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective_hard(pts.flatten())
    
    # Phase 1: L-BFGS-B on support function parameters (15D smooth space)
    support_inits = _generate_support_inits()
    
    for i, p0 in enumerate(support_inits):
        if time.time() > deadline - 60:
            break
        res = _lbfgs(p0, deadline, _objective_support, max_iter=2000)
        if res is not None:
            pts = _decode_support(res.x)
            update_best(pts)
    
    # Phase 2: NM refinement on support function parameters
    for i, p0 in enumerate(support_inits[:15]):
        if time.time() > deadline - 45:
            break
        res = _nm(p0, deadline, _objective_support_hard, max_iter=1500)
        if res is not None and res.fun < best_obj:
            pts = _decode_support(res.x)
            update_best(pts)
    
    # Phase 3: Cartesian local refinement from best found
    if best_pts is not None and time.time() < deadline - 35:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=3000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
        
        # Try hull pull
        if time.time() < deadline - 25:
            pulled = _hull_pull(best_pts, alpha=0.2)
            update_best(pulled)
            if time.time() < deadline - 20:
                res = _lbfgs(pulled.flatten(), deadline, _objective_soft, max_iter=2000)
                if res is not None and res.fun < best_obj:
                    update_best(res.x.reshape(N, 2))
    
    # Phase 4: NM on Cartesian
    if best_pts is not None and time.time() < deadline - 20:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=2000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 5: Differential Evolution in support-function space
    if time.time() < deadline - 15:
        try:
            bounds = [(0.5, 2.0)] + [(-0.3, 0.3)] * (2*K)
            de_res = differential_evolution(
                _objective_support_hard, bounds,
                seed=42, maxiter=25, tol=1e-8,
                popsize=10, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                pts = _decode_support(de_res.x)
                update_best(pts)
        except Exception:
            pass
    
    # Phase 6: Final polish with multiple methods
    if best_pts is not None and time.time() < deadline - 5:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=3000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
        
        if time.time() < deadline - 3:
            res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=1000)
            if res is not None and res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    best_pts = _normalize_unit_area(best_pts)
    
    return best_pts
