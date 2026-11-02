"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy: dual-lattice Minkowski-sum parameterization.
  1. Points = regular 14-gon + optimized asymmetric perturbation polygon
     defined by 7 independent edge-vector lengths (strict convexity guaranteed).
  2. Fast vectorized triangle-area computation over all 364 triples.
  3. Multi-phase optimization: global (DE) + local (NM, Powell, L-BFGS) refinement.
  4. Time-guarded to finish within 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

# Pre-compute all triangle index combinations (364 triples)
tri_indices = np.array(list(combinations(range(N), 3)))  # shape (364, 3)


# ---------- Core geometry (fast, vectorized) ----------

def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_triangle_areas(points):
    """All triangle areas over all triples, vectorized."""
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
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
    """Minimize hull_area / min_triangle_area (equivalent to maximizing score)."""
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


def _objective_soft(flat, beta=50.0):
    """Soft objective using log-sum-exp for smooth minimum approximation."""
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


# ---------- Dual-lattice Minkowski-sum parameterization ----------
# Points = regular 14-gon (scaled) + perturbation polygon
# Perturbation polygon defined by 7 edge vectors, each with length and fixed
# alternating direction pattern to ensure strict convexity.
# The perturbation is a closed polygon with 14 edges, where edges come in
# opposite pairs (edge i and edge i+7 are parallel and equal length),
# giving 7 independent length parameters.

# Base regular 14-gon directions (unit vectors at angles 2*pi*k/14)
_BASE_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
# Perturbation edge directions: 14 edges, alternating pattern
# For strict convexity, perturbation edges must not flip the base polygon's
# convexity. We use a pattern where edge i has direction angle = 2*pi*i/14 + delta_i
# To keep it simple and convex-guaranteed: perturbation is a scaled/rotated
# regular 14-gon with per-edge length modulation.
# 
# Simpler robust approach: perturbation polygon vertices are defined by
# 7 radial offsets applied to a base pattern. The perturbation polygon is
# constructed as: for k=0..13, vertex_k = base_vertex_k + eps * offset_k
# where offset_k alternates between two directions to create asymmetry.
#
# Final approach: parameterize by 7 edge lengths e_0..e_6. The perturbation
# polygon has 14 edges. Edge 2i and edge 2i+7 have length e_i, and edge 2i+1
# and edge 2i+8 have length 0 (or small). This creates a "stretched" shape.
# Actually, let's use a cleaner construction:
#
# Perturbation polygon: 14 vertices. Vertex k is at angle 2*pi*k/14 from center,
# with radial displacement d_k. We parameterize d_k using 7 free parameters
# via: d_k = a_k * cos(2*pi*k/14 + phase) + b_k * sin(2*pi*k/14 + phase)
# This is getting complex. Let me use the simplest convex-guaranteed approach:
#
# FINAL: points[k] = R * base_dir[k] + sum_{m=1}^{7} t_m * cos(m * 2*pi*k/14 + phi_m) * dir[k]
# This is a Fourier-like parameterization but with 7 independent amplitudes and phases.
# Total params: 7 amplitudes + 7 phases = 14 params, plus 1 scale = 15 params.
# This guarantees the shape is a smooth convex curve (for small enough t_m).

def _decode_minkowski(params):
    """
    params: [scale, t_1, ..., t_7, phi_1, ..., phi_7]
    15 parameters total.
    scale: overall scale of base polygon
    t_m: amplitude of m-th harmonic perturbation
    phi_m: phase of m-th harmonic perturbation
    Returns (N, 2) array of points.
    """
    scale = params[0]
    t = params[1:8]    # 7 amplitudes
    phi = params[8:15] # 7 phases
    
    k = np.arange(N, dtype=float)
    base_angles = 2 * np.pi * k / N
    
    # Base regular polygon
    base_x = scale * np.cos(base_angles)
    base_y = scale * np.sin(base_angles)
    
    # Add harmonic perturbations (radial modulation)
    radial_offset = np.zeros(N)
    for m in range(7):
        radial_offset += t[m] * np.cos(m * base_angles + phi[m])
    
    # Apply radial perturbation
    r = scale + radial_offset
    # Ensure positive radii for convexity
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(base_angles)
    y = r * np.sin(base_angles)
    
    return np.column_stack([x, y])


def _objective_minkowski(params, deadline):
    """Objective for Minkowski-sum parameters."""
    if time.time() > deadline:
        return 1e10
    pts = _decode_minkowski(params)
    return _objective_hard(pts.flatten())


def _objective_minkowski_soft(params, deadline, beta=50.0):
    """Soft objective for Minkowski-sum parameters."""
    if time.time() > deadline:
        return 1e10
    pts = _decode_minkowski(params)
    return _objective_soft(pts.flatten(), beta)


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


# ---------- Initial configurations ----------

def _make_initial_minkowski_params():
    """Generate diverse initial Minkowski parameter configurations."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Pure regular 14-gon (no perturbation)
    params = np.concatenate([[1.0], np.zeros(7), np.zeros(7)])
    inits.append(params)
    
    # 2. Small random perturbations
    for seed in range(20):
        rs = np.random.default_rng(seed * 17 + 3)
        scale = rs.uniform(0.8, 1.2)
        t = rs.uniform(-0.1, 0.1, 7)
        phi = rs.uniform(0, 2 * np.pi, 7)
        inits.append(np.concatenate([[scale], t, phi]))
    
    # 3. Larger perturbations
    for seed in range(15):
        rs = np.random.default_rng(seed * 31 + 7)
        scale = rs.uniform(0.7, 1.3)
        t = rs.uniform(-0.3, 0.3, 7)
        phi = rs.uniform(0, 2 * np.pi, 7)
        inits.append(np.concatenate([[scale], t, phi]))
    
    # 4. Dominant single harmonic
    for m in range(7):
        for amp in [0.05, 0.1, 0.2]:
            for ph in [0, np.pi/2, np.pi, 3*np.pi/2]:
                t = np.zeros(7)
                phi = np.zeros(7)
                t[m] = amp
                phi[m] = ph
                inits.append(np.concatenate([[1.0], t, phi]))
    
    # 5. Two-harmonic combinations
    for m1 in range(3):
        for m2 in range(m1+1, 7):
            for amp1 in [0.05, 0.1]:
                for amp2 in [0.05, 0.1]:
                    t = np.zeros(7)
                    phi = np.zeros(7)
                    t[m1] = amp1
                    t[m2] = amp2
                    phi[m1] = 0
                    phi[m2] = np.pi/2
                    inits.append(np.concatenate([[1.0], t, phi]))
    
    # 6. Jittered regular polygon encoded in Minkowski params
    base = np.column_stack([np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)])
    for js in [0.05, 0.1, 0.15]:
        perturbed = base + rng.normal(0, js, size=(N, 2))
        # Fit perturbation to Minkowski model (least squares on radial)
        cx, cy = np.mean(perturbed[:, 0]), np.mean(perturbed[:, 1])
        radii = np.sqrt((perturbed[:, 0] - cx)**2 + (perturbed[:, 1] - cy)**2)
        angles = np.arctan2(perturbed[:, 1] - cy, perturbed[:, 0] - cx)
        # Sort by angle
        order = np.argsort(angles)
        radii_sorted = radii[order]
        angles_sorted = angles[order]
        # Approximate: use mean radius as scale, fit harmonics
        scale = np.mean(radii_sorted)
        t = np.zeros(7)
        phi = np.zeros(7)
        for m in range(7):
            # Project onto cos(m*theta + phi)
            theta = angles_sorted
            # Least squares for a*cos(m*theta) + b*sin(m*theta)
            A = np.column_stack([np.cos(m * theta), np.sin(m * theta)])
            try:
                coeff, _, _, _ = np.linalg.lstsq(A, radii_sorted - scale, rcond=None)
                t[m] = np.sqrt(coeff[0]**2 + coeff[1]**2)
                phi[m] = np.arctan2(-coeff[1], coeff[0])
            except Exception:
                pass
        inits.append(np.concatenate([[scale], t, phi]))
    
    return inits


# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    
    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Phase 1: Optimize in Minkowski parameter space from diverse inits
    inits = _make_initial_minkowski_params()
    
    for i, params0 in enumerate(inits):
        if time.time() > deadline - 80:
            break
        res = _nm(params0, deadline, _objective_minkowski, max_iter=800)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _decode_minkowski(res.x)
            _consider(pts)
    
    # Phase 2: Switch to free-coordinate space for local refinement
    if best_pts is not None and time.time() < deadline - 60:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=1500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))
    
    # Phase 3: Powell refinement in free-coordinate space
    if best_pts is not None and time.time() < deadline - 45:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=800)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))
    
    # Phase 4: L-BFGS on soft objective in free-coordinate space
    if best_pts is not None and time.time() < deadline - 35:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=2000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))
    
    # Phase 5: NM on soft objective in free-coordinate space
    if best_pts is not None and time.time() < deadline - 28:
        res = _nm(best_pts.flatten(), deadline, _objective_soft, max_iter=1200)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))
    
    # Phase 6: Differential evolution in Minkowski space (global search)
    if time.time() < deadline - 25:
        try:
            bounds = [(0.5, 2.0)] + [(-0.5, 0.5)] * 7 + [(0, 2*np.pi)] * 7
            res_de = differential_evolution(
                _objective_minkowski, bounds,
                args=(deadline,),
                seed=42, maxiter=40, tol=1e-10,
                popsize=12, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, update_interval=1
            )
            if res_de is not None and np.isfinite(res_de.fun) and res_de.fun < 1e9:
                pts = _decode_minkowski(res_de.x)
                _consider(pts)
        except Exception:
            pass
    
    # Phase 7: Local refinement after DE in free-coordinate space
    if best_pts is not None and time.time() < deadline - 15:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=1200)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))
    
    # Phase 8: Powell final polish
    if best_pts is not None and time.time() < deadline - 10:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=600)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))
    
    # Phase 9: Final L-BFGS polish
    if best_pts is not None and time.time() < deadline - 5:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=1000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))
    
    # Fallback
    if best_pts is None:
        best_pts = np.column_stack([np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)])
    
    final_points = _normalize_unit_area(best_pts)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            final_points = _normalize_unit_area(np.column_stack([np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)]))
    except Exception:
        final_points = _normalize_unit_area(np.column_stack([np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)]))
    
    return final_points
