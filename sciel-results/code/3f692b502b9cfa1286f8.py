"""Heilbronn triangles: maximize the smallest triangle area among 11 points.

Structural approach:
- Barycentric coordinates (a, b, c) with a+b+c=1, all >= 0.  Store (b, c).
- Phase A: low-dimensional smooth mode parameterization.  The 11 points are
  placed at fixed "anchor" positions and each anchor is displaced by a linear
  combination of the first K Laplacian eigenmodes (smooth functions).  We
  optimize the mode coefficients with Nelder-Mead on a soft-min objective.
- Phase B: full 22-dim polish with coordinate descent and Nelder-Mead on the
  hard min objective.
"""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the unit-side equilateral triangle

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

DEADLINE = 145.0


def _barycentric_to_cartesian(b, c):
    """Convert barycentric coords (b, c) with a=1-b-c to cartesian."""
    a = 1.0 - b - c
    x = b + 0.5 * c
    y = c * S3H
    return np.column_stack([x, y])


def _cartesian_to_barycentric(x, y):
    """Convert cartesian to barycentric (b, c). a = 1-b-c."""
    c = y / S3H
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _project_bary_to_valid(bc):
    """Project barycentric coords (b, c) into the valid region a,b,c >= 0."""
    b = bc[:, 0].copy()
    c = bc[:, 1].copy()
    a = 1.0 - b - c
    # Simple iterative projection
    for _ in range(20):
        changed = False
        if np.any(a < 0):
            idx = np.where(a < 0)[0]
            a[idx] = 0.0
            changed = True
        if np.any(b < 0):
            idx = np.where(b < 0)[0]
            b[idx] = 0.0
            changed = True
        if np.any(c < 0):
            idx = np.where(c < 0)[0]
            c[idx] = 0.0
            changed = True
        if not changed:
            break
    return np.column_stack([b, c])


def _min_triangle_area(pts):
    """Compute minimum triangle area for N points in cartesian coords."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_triangle_areas(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas


def _soft_min_area(pts, beta=200.0):
    """Soft minimum of triangle areas using log-sum-exp."""
    areas = _all_triangle_areas(pts)
    # Shift for numerical stability
    a_min = np.min(areas)
    shifted = areas - a_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    return a_min + log_sum


def _push_apart(pts, min_idx, step=0.01):
    """Push vertices of the minimum-area triangle apart."""
    i, j, k = _T_I[min_idx], _T_J[min_idx], _T_K[min_idx]
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    centroid = (pi + pj + pk) / 3.0
    
    new_pts = pts.copy()
    for idx, p in [(i, pi), (j, pj), (k, pk)]:
        direction = p - centroid
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction /= norm
            new_pts[idx] += step * direction
    
    # Project back to triangle
    bc = _cartesian_to_barycentric(new_pts[:, 0], new_pts[:, 1])
    bc = _project_bary_to_valid(bc)
    return _barycentric_to_cartesian(bc[:, 0], bc[:, 1])


def _snap_to_symmetry(pts, strength=0.1):
    """Nudge points toward symmetric positions (3-fold symmetry around centroid)."""
    centroid = np.mean(pts, axis=0)
    # Rotate by 120 and 240 degrees around centroid
    angle = 2 * np.pi / 3
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # For each point, average with its rotated versions
    new_pts = pts.copy()
    for i in range(N):
        p = pts[i] - centroid
        p_rot1 = rot @ p
        p_rot2 = rot @ p_rot1
        # Only snap if the rotated positions are close to existing points
        for j in range(N):
            if j != i:
                for p_rot in [p_rot1, p_rot2]:
                    if np.linalg.norm(pts[j] - centroid - p_rot) < 0.15:
                        new_pts[i] = (1 - strength) * new_pts[i] + strength * (centroid + p_rot)
                        break
    return new_pts


def _random_bary(rng):
    """Generate random point in barycentric coords."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    return np.array([u, v])  # (b, c)


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=5.0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        bc = _cartesian_to_barycentric(pts[:, 0], pts[:, 1])
        bc = _project_bary_to_valid(bc)
        pts = _barycentric_to_cartesian(bc[:, 0], bc[:, 1])
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # ---- Initial configurations ----
    inits = []
    
    # Config 1: Corners + edge midpoints + interior points
    init1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    inits.append(init1)
    
    # Config 2: More symmetric interior placement
    init2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.5, S3H * 0.5],
        [0.25, S3H * 0.5],
        [0.75, S3H * 0.5],
        [0.5, S3H * 0.25],
        [0.5, S3H * 0.75],
        [0.5, S3H * 0.45],
    ], dtype=float)
    inits.append(init2)
    
    # Config 3: Based on 3x3 grid + corners
    init3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [1/3, 0.0],
        [2/3, 0.0],
        [0.5, S3H/3],
        [0.25, S3H/4],
        [0.75, S3H/4],
        [0.5, S3H/2],
        [0.333, S3H/3],
        [0.667, S3H/3],
    ], dtype=float)
    inits.append(init3)
    
    # Config 4: Quasi-lattice with 3-fold symmetry
    init4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 4],
        [0.25, S3H / 4],
        [0.5, S3H / 2],
        [0.25, S3H / 2],
        [0.75, S3H / 2],
        [0.333, S3H / 3],
        [0.667, S3H / 3],
    ], dtype=float)
    inits.append(init4)
    
    for init in inits:
        _update_best(init)
    
    # ---- Phase A: Multi-start with push-apart + coordinate descent ----
    n_starts = 50
    bases = list(inits)
    for _ in range(25):
        base_idx = rng.integers(0, len(inits))
        base = inits[base_idx] + rng.normal(0, 0.15, size=(N, 2))
        bc = _cartesian_to_barycentric(base[:, 0], base[:, 1])
        bc = _project_bary_to_valid(bc)
        bases.append(_barycentric_to_cartesian(bc[:, 0], bc[:, 1]))
    for _ in range(15):
        rand_pts = np.array([_barycentric_to_cartesian(*_random_bary(rng)) for _ in range(N)])
        bases.append(rand_pts)
    if best_pts is not None:
        bases.append(best_pts.copy())
    
    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        base = bases[s % len(bases)]
        pts = base + rng.normal(0, 0.03, size=(N, 2))
        bc = _cartesian_to_barycentric(pts[:, 0], pts[:, 1])
        bc = _project_bary_to_valid(bc)
        pts = _barycentric_to_cartesian(bc[:, 0], bc[:, 1])
        _update_best(pts)
        
        # Apply push-apart iterations
        for _ in range(30):
            if not _elapsed_ok(35):
                break
            areas = _all_triangle_areas(pts)
            min_idx = np.argmin(areas)
            min_a = areas[min_idx]
            new_pts = _push_apart(pts, min_idx, step=0.008)
            new_min = _min_triangle_area(new_pts)
            if new_min > min_a + 1e-10:
                pts = new_pts
            else:
                # Try smaller step
                new_pts = _push_apart(pts, min_idx, step=0.002)
                new_min = _min_triangle_area(new_pts)
                if new_min > min_a + 1e-10:
                    pts = new_pts
                else:
                    break
        _update_best(pts)
        
        # Coordinate descent polish
        if _elapsed_ok(30):
            pts = _coordinate_descent(pts, max_iter=50)
            _update_best(pts)
    
    # ---- Phase B: Nelder-Mead polish with soft-min objective ----
    if best_pts is not None and _elapsed_ok(25):
        bc_best = _cartesian_to_barycentric(best_pts[:, 0], best_pts[:, 1])
        
        for scale in [0.03, 0.015, 0.008, 0.004]:
            if not _elapsed_ok(15):
                break
            for _ in range(6):
                if not _elapsed_ok(10):
                    break
                pert = bc_best + rng.normal(0, scale, size=(N, 2))
                pert = _project_bary_to_valid(pert)
                
                def nm_obj(x):
                    bc = _project_bary_to_valid(x.reshape(-1, 2))
                    pts = _barycentric_to_cartesian(bc[:, 0], bc[:, 1])
                    return -_soft_min_area(pts, beta=150.0)
                
                try:
                    res = minimize(nm_obj, pert.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-14})
                    bc = _project_bary_to_valid(res.x.reshape(-1, 2))
                    p = _barycentric_to_cartesian(bc[:, 0], bc[:, 1])
                    _update_best(p)
                    if _min_triangle_area(p) > _min_triangle_area(_barycentric_to_cartesian(bc_best[:, 0], bc_best[:, 1])):
                        bc_best = bc.copy()
                except Exception:
                    pass
    
    # ---- Phase C: Final ultra-fine polish with hard-min objective ----
    if best_pts is not None and _elapsed_ok(10):
        bc = _cartesian_to_barycentric(best_pts[:, 0], best_pts[:, 1])
        
        def final_nm(x):
            bc = _project_bary_to_valid(x.reshape(-1, 2))
            pts = _barycentric_to_cartesian(bc[:, 0], bc[:, 1])
            return -_min_triangle_area(pts)
        
        try:
            res = minimize(final_nm, bc.flatten(), method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-16})
            bc = _project_bary_to_valid(res.x.reshape(-1, 2))
            p = _barycentric_to_cartesian(bc[:, 0], bc[:, 1])
            _update_best(p)
        except Exception:
            pass
        
        # Final coordinate descent
        try:
            p = _coordinate_descent(best_pts, max_iter=100)
            _update_best(p)
        except Exception:
            pass
    
    if best_pts is None:
        pts = np.array([_barycentric_to_cartesian(*_random_bary(rng)) for _ in range(N)])
        return pts
    
    return _project_to_triangle(best_pts)


def _project_to_triangle(pts):
    """Vectorized projection into equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _coordinate_descent(pts, max_iter=100, tol=1e-10):
    """Coordinate descent: for each point, try small perturbations."""
    current_min = _min_triangle_area(pts)
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for it in range(max_iter):
        improved = False
        for idx in range(N):
            best_local = current_min
            best_pos = pts[idx].copy()
            
            for step in step_sizes:
                for dx in [-step, 0.0, step]:
                    for dy in [-step, 0.0, step]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        new_pts = pts.copy()
                        new_pts[idx] = best_pos + np.array([dx, dy])
                        new_pts = _project_to_triangle(new_pts)
                        new_min = _min_triangle_area(new_pts)
                        if new_min > best_local + tol:
                            best_local = new_min
                            best_pos = new_pts[idx].copy()
            
            if np.linalg.norm(best_pos - pts[idx]) > 1e-12:
                pts[idx] = best_pos
                current_min = _min_triangle_area(pts)
                improved = True
        
        if not improved:
            break
    
    return pts
