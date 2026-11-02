"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy:
  1. Support-function parameterization: 14 points as vertices of a convex polygon
     defined by support function values (radii) at 14 fixed non-uniform angles.
     This strictly enforces convexity by construction.
  2. Non-uniform angle sampling that clusters near diagonals to better resolve
     the geometry of the smallest triangles.
  3. Multi-start Nelder-Mead optimization on the 14 radial parameters.
  4. Cartesian refinement from best support-function results.
  5. Time-guarded to finish well within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

# Pre-compute all triangle index combinations
tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)

# Non-uniform angles: cluster near diagonals (45°, 135°, 225°, 315°)
# Start with uniform 14-gon angles, then perturb to cluster near diagonals
_base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
# Add perturbation that clusters angles near the four diagonal directions
_diagonal_angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
# Create a smooth perturbation that pulls angles toward nearest diagonal
def _angle_perturbation(angles):
    """Create a perturbation that clusters angles near diagonal directions."""
    pert = np.zeros_like(angles)
    for da in _diagonal_angles:
        # Gaussian bump centered at each diagonal
        diff = np.angle(np.exp(1j * (angles - da)))  # circular difference
        bump = np.exp(-diff**2 / (2 * 0.3**2))
        pert += bump
    return pert

# Apply clustering: mix uniform and clustered angles
_cluster_strength = 0.3
_angles_support = _base_angles + _cluster_strength * _angle_perturbation(_base_angles)
# Normalize to maintain ordering
_angles_support = np.sort(_angles_support)
# Re-center to start at 0
_angles_support = _angles_support - _angles_support[0]


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


def _support_function_to_points(radii):
    """
    Convert support function values (radii) at fixed angles to polygon vertices.
    
    For a convex polygon, the support function h(θ) gives the distance from the
    origin to the edge with normal direction θ. The vertices are found at the
    intersections of consecutive edges.
    
    Given radii r_i at angles θ_i, the edge i has:
      normal: n_i = (cos θ_i, sin θ_i)
      offset: r_i
    
    The vertex between edge i and edge i+1 is the intersection of:
      n_i · x = r_i
      n_{i+1} · x = r_{i+1}
    """
    angles = _angles_support
    n = len(radii)
    vertices = np.zeros((n, 2))
    
    for i in range(n):
        j = (i + 1) % n
        # Edge i: n_i · x = r_i
        # Edge j: n_j · x = r_j
        n_i = np.array([np.cos(angles[i]), np.sin(angles[i])])
        n_j = np.array([np.cos(angles[j]), np.sin(angles[j])])
        r_i = radii[i]
        r_j = radii[j]
        
        # Solve: [n_i; n_j] · x = [r_i; r_j]
        A = np.array([n_i, n_j])
        b = np.array([r_i, r_j])
        
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-15:
            # Parallel edges - shouldn't happen with distinct angles
            vertices[i] = np.array([r_i * n_i[0], r_i * n_i[1]])
        else:
            x = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
            y = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det
            vertices[i] = np.array([x, y])
    
    return vertices


def _objective_support(radii):
    """Objective for support function parameterization."""
    if np.any(~np.isfinite(radii)):
        return 1e10
    if np.any(radii <= 0):
        return 1e10
    pts = _support_function_to_points(radii)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_support_soft(radii, beta=80.0):
    """Soft objective for support function parameterization."""
    if np.any(~np.isfinite(radii)):
        return 1e10
    if np.any(radii <= 0):
        return 1e10
    pts = _support_function_to_points(radii)
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


# ---------- Support function parameterization ----------
def _generate_support_inits():
    """Generate diverse initial support function parameter sets (radii)."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 14-gon: all radii equal
    # For a regular polygon, h(θ) = r * cos(π/N) where r is circumradius
    # But for support function, if we want unit circumradius, h(θ) = cos(π/14)
    base_r = np.cos(np.pi / N)
    inits.append(np.full(N, base_r))
    
    # 2. Elliptical support functions
    for rx, ry in [(1.0, 0.8), (0.8, 1.0), (1.2, 0.7), (0.7, 1.2), (1.1, 0.9)]:
        # Support function of ellipse: h(θ) = sqrt(a²cos²θ + b²sin²θ)
        h = np.sqrt(rx**2 * np.cos(_angles_support)**2 + ry**2 * np.sin(_angles_support)**2)
        inits.append(h)
    
    # 3. Random perturbations of regular
    for seed in range(20):
        rs = np.random.default_rng(seed * 17 + 3)
        r = base_r * rs.uniform(0.8, 1.2, N)
        inits.append(r)
    
    # 4. Alternating large/small (diamond-like)
    for amp in [0.1, 0.2, 0.3]:
        r = base_r * (1 + amp * np.cos(2 * _angles_support))
        inits.append(r)
    
    # 5. Quadratic modulation
    for coeff in [0.05, 0.1, 0.15]:
        r = base_r * (1 + coeff * np.cos(4 * _angles_support))
        inits.append(r)
    
    # 6. Larger random variations
    for seed in range(15):
        rs = np.random.default_rng(seed * 29 + 11)
        r = rs.uniform(0.5, 1.5, N)
        inits.append(r)
    
    return inits


# ---------- Optimizer helpers ----------

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
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
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


def _make_cartesian_inits():
    """Generate diverse initial point configurations (Cartesian)."""
    inits = []
    rng = np.random.default_rng(42)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    for r1 in [0.3, 0.5, 0.6, 0.7, 0.85]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts)
    
    for r1, r2 in [(0.3, 0.6), (0.4, 0.75), (0.5, 0.85)]:
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        inits.append(pts)
    
    for seed in range(10):
        rs = np.random.default_rng(seed * 37 + 19)
        pts = rs.uniform(-1, 1, size=(N, 2))
        inits.append(pts)
    
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    for js in [0.05, 0.1, 0.2]:
        inits.append(base + rng.normal(0, js, size=(N, 2)))
    
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7), (0.8, 1.2)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]))
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
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
    
    # Phase 1: Support function multi-start Nelder-Mead (14D parameter space)
    support_inits = _generate_support_inits()
    
    for i, r0 in enumerate(support_inits):
        if time.time() > deadline - 80:
            break
        try:
            res = minimize(_objective_support, r0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_obj:
                pts = _support_function_to_points(res.x)
                update_best(pts)
        except Exception:
            pass
    
    # Phase 2: Cartesian local refinement from best found
    if best_pts is not None and time.time() < deadline - 50:
        try:
            res = minimize(_objective_hard, best_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B with soft objective for smooth refinement
    if best_pts is not None and time.time() < deadline - 30:
        try:
            res = minimize(_objective_soft, best_pts.flatten(), method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 4: Additional support function starts with different angle clustering
    if time.time() < deadline - 20:
        # Try with different clustering strengths
        for strength in [0.1, 0.2, 0.4, 0.5]:
            if time.time() > deadline - 20:
                break
            # Create new angles with different clustering
            angles_new = _base_angles + strength * _angle_perturbation(_base_angles)
            angles_new = np.sort(angles_new)
            angles_new = angles_new - angles_new[0]
            
            # Generate inits for these angles
            for r0 in [np.full(N, np.cos(np.pi / N)), 
                       np.cos(np.pi / N) * (1 + 0.1 * np.cos(2 * angles_new)),
                       np.cos(np.pi / N) * (1 + 0.2 * np.cos(4 * angles_new))]:
                if time.time() > deadline - 20:
                    break
                # Temporarily use new angles
                global _angles_support
                old_angles = _angles_support
                _angles_support = angles_new
                try:
                    res = minimize(_objective_support, r0, method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                    if res.fun < best_obj:
                        pts = _support_function_to_points(res.x)
                        update_best(pts)
                except Exception:
                    pass
                finally:
                    _angles_support = old_angles
    
    # Phase 5: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 10:
        try:
            res = minimize(_objective_hard, best_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        except Exception:
            pass
        
        if time.time() < deadline - 5:
            try:
                res = minimize(_objective_hard, best_pts.flatten(), method='Powell',
                              options={'maxiter': 2000, 'xtol': 1e-13, 'ftol': 1e-15})
                if res.fun < best_obj:
                    update_best(res.x.reshape(N, 2))
            except Exception:
                pass
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    best_pts = _normalize_unit_area(best_pts)
    
    return best_pts
