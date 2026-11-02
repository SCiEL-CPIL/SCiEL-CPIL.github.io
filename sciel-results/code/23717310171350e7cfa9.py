"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Pre-compute all triangle index combinations for vectorized scoring
_tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[_tri_indices[:, 0]]
    pj = p[_tri_indices[:, 1]]
    pk = p[_tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(points):
    """Convex hull area using scipy."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _objective_hard(flat):
    """Objective to minimize: hull_area / min_triangle_area (inverse of score)."""
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
    """Soft objective using log-sum-exp on triangle areas for smoother gradient."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
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


# ---------- Ring parameterization (2 rings of 7) ----------
# params: [r1, r2, phi1, phi2, d1_0..d1_6, d2_0..d2_6]
#   r1, r2: radii of inner and outer ring
#   phi1, phi2: angular offsets (rotation) of each ring
#   d1_i, d2_i: angular deviations from uniform spacing
# Total: 18 parameters

def _decode_ring2(params):
    """Decode 18-dim ring params to (N,2) points."""
    r1 = params[0]
    r2 = params[1]
    phi1 = params[2]
    phi2 = params[3]
    d1 = params[4:11]
    d2 = params[11:18]
    
    base_angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    base_angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    
    angles1 = phi1 + base_angles1 + d1
    angles2 = phi2 + base_angles2 + d2
    
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    
    return np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])


def _objective_ring2(params):
    pts = _decode_ring2(params)
    return _objective_hard(pts.flatten())


def _objective_ring2_soft(params, beta=80.0):
    pts = _decode_ring2(params)
    return _objective_soft(pts.flatten(), beta)


# ---------- Fourier boundary parameterization ----------
def _fourier_boundary(coeffs, n_samples=300):
    """Generate points on a smooth convex boundary from Fourier coefficients."""
    K = (len(coeffs) - 1) // 2
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    
    r = np.full(n_samples, coeffs[0])
    for k in range(1, K + 1):
        ak = coeffs[2 * k - 1]
        bk = coeffs[2 * k]
        r += ak * np.cos(k * thetas) + bk * np.sin(k * thetas)
    
    r_min = r.min()
    if r_min <= 0.1:
        r = r + (0.1 - r_min + 1e-6)
    
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    return np.column_stack([x, y])


def _place_by_arc_length(boundary_pts, n_points, phase_shifts=None):
    """Place n_points at equal arc-length intervals along the boundary."""
    n = len(boundary_pts)
    if phase_shifts is None:
        phase_shifts = np.zeros(n_points)
    
    diffs = np.diff(boundary_pts, axis=0, append=boundary_pts[:1])
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_length = np.cumsum(seg_lengths)
    total_length = cum_length[-1]
    
    target_fracs = (np.arange(n_points) + phase_shifts) / n_points
    target_fracs = target_fracs % 1.0
    
    points = np.zeros((n_points, 2))
    for i in range(n_points):
        target_len = target_fracs[i] * total_length
        idx = np.searchsorted(cum_length, target_len)
        if idx == 0:
            points[i] = boundary_pts[0]
        elif idx >= n:
            points[i] = boundary_pts[-1]
        else:
            seg_start = cum_length[idx - 1]
            seg_end = cum_length[idx]
            t = (target_len - seg_start) / (seg_end - seg_start + 1e-15)
            points[i] = (1 - t) * boundary_pts[idx - 1] + t * boundary_pts[idx]
    
    return points


# ---------- Optimizer helpers with time guards ----------

def _nm(flat0, deadline, obj, max_iter=2000, xatol=1e-9, fatol=1e-11):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True})
        return res
    except Exception:
        return None


def _powell(flat0, deadline, obj, max_iter=500, xtol=1e-10, ftol=1e-12):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': xtol, 'ftol': ftol})
        return res
    except Exception:
        return None


def _lbfgs(flat0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(obj, flat0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None


def construct():
    """Construct 14 points maximizing min triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_points = None
    best_score = 0.0
    rng = np.random.RandomState(42)
    
    # ========== Phase 1: Ring-structured search ==========
    # Generate diverse initial ring parameter vectors
    ring_inits = []
    
    # Regular two-ring configurations
    for r1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        for phi2 in [0.0, np.pi / 7, np.pi / 14, 2 * np.pi / 7]:
            params = np.zeros(18)
            params[0] = r1
            params[1] = 1.0
            params[3] = phi2
            ring_inits.append(params)
    
    # Jittered ring params
    for js in [0.05, 0.1, 0.15, 0.2]:
        params = np.zeros(18)
        params[0] = 0.6
        params[1] = 1.0
        params[3] = np.pi / 7
        params[4:18] = rng.normal(0, js, size=14)
        ring_inits.append(params)
    
    # Random ring params
    for seed in range(15):
        rs = np.random.RandomState(seed * 13 + 7)
        params = np.zeros(18)
        params[0] = rs.uniform(0.2, 0.9)
        params[1] = rs.uniform(0.8, 1.3)
        params[2] = rs.uniform(-0.3, 0.3)
        params[3] = rs.uniform(0.0, 2 * np.pi)
        params[4:18] = rs.normal(0, 0.12, size=14)
        ring_inits.append(params)
    
    # Evaluate all ring inits
    for params in ring_inits:
        if time.time() > deadline - 60:
            break
        pts = _decode_ring2(params)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Optimize top ring configurations with Nelder-Mead in ring space
    if best_points is not None and time.time() < deadline - 50:
        # Sort ring inits by score and take top ones
        scored_inits = []
        for params in ring_inits[:30]:  # limit to top 30
            pts = _decode_ring2(params)
            s = _score(pts)
            scored_inits.append((s, params))
        scored_inits.sort(key=lambda x: -x[0])
        
        for _, params in scored_inits[:8]:
            if time.time() > deadline - 45:
                break
            res = _nm(params, deadline, _objective_ring2, max_iter=1500)
            if res is not None and res.fun < 1e9:
                opt_pts = _decode_ring2(res.x)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
    
    # ========== Phase 2: Cartesian refinement ==========
    if best_points is not None:
        # Nelder-Mead in Cartesian space
        if time.time() < deadline - 30:
            flat0 = best_points.flatten()
            res = _nm(flat0, deadline, _objective_hard, max_iter=2000)
            if res is not None and res.fun < 1e9:
                opt_pts = res.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
        
        # Powell refinement
        if time.time() < deadline - 20:
            flat0 = best_points.flatten()
            res = _powell(flat0, deadline, _objective_hard, max_iter=400)
            if res is not None and res.fun < 1e9:
                opt_pts = res.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
        
        # L-BFGS-B with soft objective
        if time.time() < deadline - 15:
            flat0 = best_points.flatten()
            res = _lbfgs(flat0, deadline, lambda x: _objective_soft(x, beta=60.0), max_iter=800)
            if res is not None and res.fun < 1e9:
                opt_pts = res.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
    
    # ========== Phase 3: Differential Evolution in ring space ==========
    if time.time() < deadline - 35:
        bounds = [(0.1, 1.2), (0.7, 1.5), (-0.5, 0.5), (-0.5, 0.5)] + \
                 [(-0.4, 0.4)] * 7 + [(-0.4, 0.4)] * 7
        
        try:
            de_result = differential_evolution(
                _objective_ring2, bounds,
                seed=42, maxiter=30, popsize=15,
                tol=1e-8, mutation=(0.3, 1.2), recombination=0.9,
                polish=True,
                update_interval=5,
                init='sobol'
            )
            if de_result.fun < 1e9:
                opt_pts = _decode_ring2(de_result.x)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
        except Exception:
            pass
    
    # ========== Phase 4: Fourier boundary exploration ==========
    if time.time() < deadline - 20:
        n_samples = 300
        for n_harm in [1, 2, 3]:
            for trial in range(6):
                if time.time() > deadline - 15:
                    break
                
                a0 = 1.0 + rng.uniform(-0.1, 0.1)
                coeffs = np.zeros(2 * (n_harm + 1))
                coeffs[0] = a0
                for k in range(1, n_harm + 1):
                    scale = 0.2 / k
                    coeffs[2 * k - 1] = rng.uniform(-scale, scale)
                    coeffs[2 * k] = rng.uniform(-scale, scale)
                
                boundary = _fourier_boundary(coeffs, n_samples)
                
                # Check convexity
                hull_area = _hull_area(boundary)
                ref_hull = _hull_area(np.column_stack(
                    [np.cos(np.linspace(0, 2 * np.pi, n_samples, endpoint=False)),
                     np.sin(np.linspace(0, 2 * np.pi, n_samples, endpoint=False))]))
                if hull_area < 0.85 * ref_hull:
                    continue
                
                initial_shifts = np.zeros(N)
                pts = _place_by_arc_length(boundary, N, initial_shifts)
                s = _score(pts)
                
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                
                # Optimize in Cartesian from this starting point
                if time.time() < deadline - 10:
                    flat0 = pts.flatten()
                    res = _nm(flat0, deadline, _objective_hard, max_iter=800)
                    if res is not None and res.fun < 1e9:
                        opt_pts = res.x.reshape(N, 2)
                        s2 = _score(opt_pts)
                        if s2 > best_score:
                            best_score = s2
                            best_points = opt_pts.copy()
    
    # ========== Phase 5: Multiple restarts from perturbed best ==========
    if best_points is not None and time.time() < deadline - 10:
        for trial in range(6):
            if time.time() > deadline - 5:
                break
            
            perturb = rng.normal(0, 0.06, (N, 2))
            pts = best_points + perturb
            
            def obj(x):
                return _objective_hard(x)
            
            res = _nm(pts.flatten(), deadline, obj, max_iter=1000)
            if res is not None and res.fun < 1e9:
                opt_pts = res.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
    
    # ========== Phase 6: Final fine-tuning ==========
    if best_points is not None and time.time() < deadline - 3:
        # Small perturbations
        for trial in range(50):
            if time.time() > deadline - 1:
                break
            perturbed = best_points + rng.normal(0, 0.01, (N, 2))
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_points = perturbed.copy()
        
        # One final Powell polish
        if time.time() < deadline - 2:
            res = _powell(best_points.flatten(), deadline, _objective_hard, max_iter=200)
            if res is not None and res.fun < 1e9:
                opt_pts = res.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
    
    return best_points
