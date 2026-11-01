"""Radial-ring packing for 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _ring_structure():
    """Define hexagonal-like ring counts summing to 300."""
    counts = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]
    assert sum(counts) == 300
    return np.array(counts)


def _build_points(ring_radii, phases, ring_counts):
    """Build point array from ring radii, phases, and counts."""
    points_list = []
    for r, phase, count in zip(ring_radii, phases, ring_counts):
        if count == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + phase
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        points_list.append(np.column_stack([xs, ys]))
    if not points_list:
        return np.zeros((0, 2))
    return np.vstack(points_list)


def _compute_packing_radius(points):
    """Compute the maximum common radius for given centre positions."""
    if len(points) == 0:
        return 0.0
    if len(points) == 1:
        return 1.0 - np.hypot(*points[0])
    dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dists_boundary)
    dm = cdist(points, points)
    np.fill_diagonal(dm, np.inf)
    min_pair = np.min(dm) / 2.0
    return min(min_boundary, min_pair)


def _max_radius_analytic(ring_counts):
    """Binary search for max r where minimal ring radii fit in unit circle."""
    k = len(ring_counts)
    def feasible(r):
        if r <= 0:
            return True
        R = np.zeros(k)
        for i, n in enumerate(ring_counts):
            if n > 1:
                R_min_ring = r / np.sin(np.pi / n)
            else:
                R_min_ring = 0.0
            if i == 0:
                R[i] = R_min_ring
            else:
                R[i] = max(R_min_ring, R[i-1] + 2*r)
        return R[-1] + r <= 1.0
    lo, hi = 0.0, 0.1
    for _ in range(60):
        mid = (lo + hi) / 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _optimize_phases(ring_radii, ring_counts, n_starts=15):
    """Optimize angular phases to maximize packing radius."""
    k = len(ring_counts)
    def objective(phases):
        points = _build_points(ring_radii, phases, ring_counts)
        return -_compute_packing_radius(points)
    best_r = 0.0
    best_phases = None
    for seed in range(n_starts):
        rng = np.random.default_rng(seed)
        phases0 = np.zeros(k)
        for i in range(1, k):
            if ring_counts[i] > 0 and ring_counts[i-1] > 0:
                phases0[i] = np.pi / ring_counts[i-1] + rng.uniform(-0.3, 0.3)
            else:
                phases0[i] = rng.uniform(0, 2*np.pi)
        try:
            res = minimize(objective, phases0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            r_val = -res.fun
            if r_val > best_r:
                best_r = r_val
                best_phases = res.x.copy()
        except:
            pass
    if best_phases is None:
        best_phases = np.zeros(k)
        for i in range(1, k):
            if ring_counts[i] > 0 and ring_counts[i-1] > 0:
                best_phases[i] = np.pi / ring_counts[i-1]
        best_r = _compute_packing_radius(_build_points(ring_radii, best_phases, ring_counts))
    return best_r, best_phases


def _optimize_radii(ring_counts, phases, n_starts=8):
    """Optimize ring radii for fixed phases."""
    k = len(ring_counts)
    def objective(params):
        radii = np.maximum(params, 0.0)
        points = _build_points(radii, phases, ring_counts)
        return -_compute_packing_radius(points)
    best_r = 0.0
    best_radii = None
    for seed in range(n_starts):
        rng = np.random.default_rng(seed + 100)
        radii0 = np.linspace(0.05, 0.9, k)
        if ring_counts[0] == 1:
            radii0[0] = 0.0
        radii0 = np.maximum(radii0 + rng.uniform(-0.02, 0.02, size=k), 0.0)
        try:
            res = minimize(objective, radii0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            r_val = -res.fun
            if r_val > best_r:
                best_r = r_val
                best_radii = np.maximum(res.x, 0.0)
        except:
            pass
    if best_radii is None:
        best_radii = np.linspace(0.05, 0.9, k)
        if ring_counts[0] == 1:
            best_radii[0] = 0.0
        best_r = _compute_packing_radius(_build_points(best_radii, phases, ring_counts))
    return best_r, best_radii


def _fallback_hexagonal():
    """Fallback hexagonal lattice packing."""
    r = 0.053
    spacing = 2 * r
    pts = []
    for j in range(-20, 21):
        for i in range(-20, 21):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) + r <= 1.0:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        dists = np.hypot(pts[:, 0], pts[:, 1])
        order = np.argsort(dists, kind='stable')
        return pts[order[:N]]
    return np.zeros((N, 2))


def construct():
    """Construct the packing configuration."""
    ring_counts = _ring_structure()
    k = len(ring_counts)
    
    r_upper = _max_radius_analytic(ring_counts)
    best_r = 0.0
    best_points = None
    
    for frac in [0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96]:
        r_target = r_upper * frac
        
        radii_min = np.zeros(k)
        for i, n in enumerate(ring_counts):
            if n > 1:
                radii_min[i] = r_target / np.sin(np.pi / n)
            else:
                radii_min[i] = 0.0
            if i > 0:
                radii_min[i] = max(radii_min[i], radii_min[i-1] + 2*r_target)
        
        if radii_min[-1] + r_target > 1.0:
            continue
        
        r_ph, phases_opt = _optimize_phases(radii_min, ring_counts, n_starts=15)
        if r_ph > best_r:
            best_r = r_ph
            best_points = _build_points(radii_min, phases_opt, ring_counts)
        
        r_rad, radii_opt = _optimize_radii(ring_counts, phases_opt, n_starts=8)
        if r_rad > best_r:
            best_r = r_rad
            best_points = _build_points(radii_opt, phases_opt, ring_counts)
    
    if best_points is not None:
        # Final joint refinement
        ring_radii_est = np.zeros(k)
        phases_est = np.zeros(k)
        idx = 0
        for i, n in enumerate(ring_counts):
            if n == 0:
                continue
            ring_pts = best_points[idx:idx+n]
            ring_radii_est[i] = np.mean(np.hypot(ring_pts[:, 0], ring_pts[:, 1]))
            if n > 1:
                angles = np.arctan2(ring_pts[:, 1], ring_pts[:, 0])
                phases_est[i] = np.mean(angles)
            idx += n
        
        def full_obj(params):
            radii = np.maximum(params[:k], 0.0)
            phases = params[k:]
            points = _build_points(radii, phases, ring_counts)
            return -_compute_packing_radius(points)
        
        x0 = np.concatenate([ring_radii_est, phases_est])
        try:
            res = minimize(full_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            r_final = -res.fun
            if r_final > best_r:
                radii_final = np.maximum(res.x[:k], 0.0)
                phases_final = res.x[k:]
                best_r = r_final
                best_points = _build_points(radii_final, phases_final, ring_counts)
        except:
            pass
    
    if best_points is None or len(best_points) != N:
        return _fallback_hexagonal()
    
    r_check = _compute_packing_radius(best_points)
    if r_check < 0.05:
        return _fallback_hexagonal()
    
    return best_points

# EVOLVE-BLOCK-END
