# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


_N = 13
# Pre-compute all 286 triangle index triples
_TRIS = np.array([(i, j, k) for i in range(_N) for j in range(i+1, _N) for k in range(j+1, _N)], dtype=np.int32)
_I_IDX = _TRIS[:, 0]
_J_IDX = _TRIS[:, 1]
_K_IDX = _TRIS[:, 2]


def _triangle_areas(pts):
    """Compute all 286 triangle areas vectorized."""
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return areas


def _score(pts):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    if len(pts) < 3:
        return 0.0
    areas = _triangle_areas(pts)
    a_min = np.min(areas)
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull


def _soft_min_triangle_area(pts, beta):
    """Soft-min of triangle areas using log-sum-exp for smoothness."""
    areas = _triangle_areas(pts)
    areas = np.maximum(areas, 1e-15)
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -log_sum / beta
    return soft_min


def _hull_area_safe(pts):
    """Compute hull area, with a small penalty if hull is degenerate."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10


def _objective(x, beta):
    """Objective to minimize: -A_min_soft / A_hull."""
    pts = np.column_stack([x[0:13], x[13:26]])
    soft_min = _soft_min_triangle_area(pts, beta)
    hull = _hull_area_safe(pts)
    if hull <= 0 or not np.isfinite(soft_min) or not np.isfinite(hull):
        return 1e10
    return -soft_min / hull


def _regular_polygon(n=13):
    """Create regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _hexagonal_lattice_init(n=13):
    """Create a hexagonal lattice-like arrangement."""
    pts = []
    # Central point
    pts.append([0.0, 0.0])
    # First ring (6 points)
    for k in range(6):
        angle = k * np.pi / 3
        pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    # Second ring (partial, to get to 13)
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        pts.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
    return np.array(pts[:n])


def _concentric_rings_init(n=13):
    """Two concentric rings: 4 inner, 9 outer."""
    inner_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False) + np.pi / 9
    inner_pts = np.column_stack([0.4 * np.cos(inner_angles), 0.4 * np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    return np.vstack([inner_pts, outer_pts])


def _delauanay_relax(pts, n_iter=500, learning_rate=0.01):
    """
    Physics-based relaxation using Delaunay triangulation.
    Move internal vertices to maximize minimum circumradius of adjacent triangles.
    """
    pts = pts.copy()
    lr = learning_rate
    
    for it in range(n_iter):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(pts)
        except Exception:
            break
        
        # Compute circumradii of all triangles in the Delaunay triangulation
        n_tris = len(tri.simplices)
        circumradii = np.zeros(n_tris)
        
        for idx in range(n_tris):
            t = tri.simplices[idx]
            a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
            # Circumradius = abc / (4 * area)
            area = abs(np.cross(b - a, c - a)) / 2.0
            if area < 1e-12:
                circumradii[idx] = 1e6
                continue
            side_ab = np.linalg.norm(b - a)
            side_bc = np.linalg.norm(c - b)
            side_ca = np.linalg.norm(a - c)
            circumradii[idx] = (side_ab * side_bc * side_ca) / (4.0 * area)
        
        # For each vertex, compute the gradient direction
        # We want to maximize the minimum circumradius
        # Use a soft-min approach: gradient of -log(sum(exp(-beta * r)))
        beta = 200
        neg_beta_r = -beta * circumradii
        max_val = np.max(neg_beta_r)
        weights = np.exp(neg_beta_r - max_val)
        weights /= np.sum(weights)
        
        # Compute force on each vertex
        forces = np.zeros_like(pts)
        for idx in range(n_tris):
            t = tri.simplices[idx]
            a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
            # Gradient of circumradius w.r.t. each vertex
            # For triangle (a,b,c), the circumcenter O satisfies:
            # |O-a|^2 = |O-b|^2 = |O-c|^2
            # The circumradius R = |O-a|
            # dR/da = (O-a)/|O-a| (direction from a to O, normalized)
            # Actually, moving a in the direction away from O increases R
            # Let's compute the circumcenter
            d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
            if abs(d) < 1e-12:
                continue
            ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) + (b[0]**2 + b[1]**2) * (c[1] - a[1]) + (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
            uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) + (b[0]**2 + b[1]**2) * (a[0] - c[0]) + (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
            O = np.array([ux, uy])
            
            # Force on each vertex: direction from O to vertex (to increase distance from O)
            for vi in t:
                v = pts[vi]
                direction = v - O
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    continue
                direction /= norm
                forces[vi] += weights[idx] * direction
        
        # Apply forces with learning rate and decay
        pts += lr * forces
        lr *= 0.995  # Slow decay
    
    return pts


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    time_limit = 600  # Safety margin from 630s

    n = 13
    best_pts = _regular_polygon(n)
    best_score = _score(best_pts)

    # Try multiple initializations
    initializations = []

    # 1. Regular polygon
    initializations.append(_regular_polygon(n))

    # 2. Hexagonal lattice
    initializations.append(_hexagonal_lattice_init(n))

    # 3. Concentric rings
    initializations.append(_concentric_rings_init(n))

    # 4. Two concentric rings (5 + 8)
    inner_angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    outer_angles = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    initializations.append(np.vstack([inner_pts, outer_pts]))

    # 5. Slightly perturbed regular polygon
    for seed in range(3):
        rng2 = np.random.default_rng(seed)
        perturbed = _regular_polygon(n)
        perturbed += rng2.normal(0, 0.15, (n, 2))
        initializations.append(perturbed)

    # Optimize each initialization
    beta_schedule = [50, 100, 200, 500, 1000]

    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 80:
            break

        # First, apply Delaunay-based relaxation
        relaxed_pts = _delauanay_relax(init_pts, n_iter=300, learning_rate=0.02)
        relaxed_score = _score(relaxed_pts)
        
        # Use the better of original and relaxed as starting point
        if relaxed_score > _score(init_pts):
            current_pts = relaxed_pts
        else:
            current_pts = init_pts.copy()

        # Gradient-based optimization with soft-min objective
        x0 = current_pts.flatten()
        current_x = x0.copy()
        
        for beta in beta_schedule:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 40:
                break

            # Nelder-Mead optimization (more robust than L-BFGS-B for this problem)
            try:
                nm_result = minimize(
                    lambda x: _objective(x, beta),
                    current_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                if nm_result.fun < _objective(current_x, beta) - 1e-12:
                    current_x = nm_result.x
            except Exception:
                pass

            # Additional Delaunay relaxation after gradient step
            current_pts = np.column_stack([current_x[0:13], current_x[13:26]])
            elapsed = time.time() - start_time
            if elapsed < time_limit - 30:
                relaxed_pts = _delauanay_relax(current_pts, n_iter=200, learning_rate=0.01)
                relaxed_score = _score(relaxed_pts)
                if relaxed_score > _score(current_pts):
                    current_x = relaxed_pts.flatten()

        # Evaluate final result
        final_pts = np.column_stack([current_x[0:13], current_x[13:26]])
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    # Final high-precision polish with exact objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 20 and best_score > 0:
        x_final = best_pts.flatten()

        def exact_obj(x):
            pts = np.column_stack([x[0:13], x[13:26]])
            s = _score(pts)
            return -s if s > 0 else 1e10

        # Nelder-Mead with tight tolerances
        try:
            nm_final = minimize(
                exact_obj,
                x_final,
                method='Nelder-Mead',
                options={
                    'maxiter': 8000,
                    'xatol': 1e-12,
                    'fatol': 1e-14
                }
            )
            if nm_final.fun < -best_score:
                best_pts = np.column_stack([nm_final.x[0:13], nm_final.x[13:26]])
                best_score = -nm_final.fun
        except Exception:
            pass

        # One more Delaunay relaxation pass
        elapsed = time.time() - start_time
        if elapsed < time_limit - 10:
            relaxed_pts = _delauanay_relax(best_pts, n_iter=300, learning_rate=0.005)
            relaxed_score = _score(relaxed_pts)
            if relaxed_score > best_score:
                best_pts = relaxed_pts
                best_score = relaxed_score

    # Validate result
    if best_score <= 0 or not np.all(np.isfinite(best_pts)):
        return _regular_polygon(n)

    return best_pts


# EVOLVE-BLOCK-END
