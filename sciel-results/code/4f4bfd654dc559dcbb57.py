import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """Compute the minimum triangle area divided by convex hull area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_min_score(points, temperature=0.01):
    """Smooth surrogate using log-sum-exp for the minimum triangle area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    max_area = np.max(areas)
    if max_area < 1e-12:
        return 0.0
    shifted = areas - max_area
    soft_min = max_area - temperature * np.log(np.sum(np.exp(shifted / temperature)))

    return soft_min / hull_area

def _soft_min_value(areas, temp):
    """Compute soft min of areas using log-sum-exp."""
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(shifted / temp)))
    return max_a - temp * log_sum

def _smooth_objective(x, temp=0.05):
    """Objective function using soft-min for smoothness."""
    p = x.reshape(N, 2)
    if p.shape != (N, 2):
        return 1e6
    
    try:
        hull = ConvexHull(p)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e6
    except Exception:
        return 1e6

    p_i = p[TRIPLE_INDICES[:, 0]]
    p_j = p[TRIPLE_INDICES[:, 1]]
    p_k = p[TRIPLE_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area_soft = _soft_min_value(areas, temp)
    
    return -min_area_soft / hull_area

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    """Generate points on two concentric circles."""
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out

def _generate_three_ring(r1, r2, phi1, phi2, n0=7, n1=5, n2=2):
    """Generate points on three concentric circles with 7+5+2=14 points."""
    angles0 = np.linspace(0, 2*np.pi, n0, endpoint=False)
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + phi1
    angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + phi2
    
    pts0 = np.column_stack((np.cos(angles0), np.sin(angles0)))
    pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    
    return np.vstack((pts0, pts1, pts2))

def _generate_heptagon_pairs(offsets, affine_params):
    """
    Generate 14 points as 7 pairs on a regular heptagon.
    Each pair is at a heptagon vertex, separated radially by the offset.
    """
    n_pairs = 7
    angles = 2 * np.pi * np.arange(n_pairs) / n_pairs
    base_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    pts = np.zeros((14, 2))
    for i in range(n_pairs):
        r_offset = offsets[i] / 2.0
        pts[2*i] = base_pts[i] - r_offset * base_pts[i]
        pts[2*i+1] = base_pts[i] + r_offset * base_pts[i]
    
    scale, rotation, shear = affine_params
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation), np.cos(rotation)]])
    S = np.array([[1, shear], [0, 1]])
    A = scale * S @ R
    pts = pts @ A.T
    
    return pts

def _coordinate_descent_polish(points, time_limit=2.0):
    """Iteratively adjust each point to improve the minimum triangle area."""
    start_time = time.time()
    pts = points.copy()
    score = _calculate_score(pts)
    
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step in step_sizes:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                directions = [
                    [step, 0], [-step, 0], [0, step], [0, -step],
                    [step, step], [step, -step], [-step, step], [-step, -step]
                ]
                for dx, dy in directions:
                    trial = pts.copy()
                    trial[i] += [dx, dy]
                    trial_score = _calculate_score(trial)
                    if trial_score > score:
                        pts = trial
                        score = trial_score
                        improved = True
                        break
        if not improved:
            break
    
    return pts

def _polish_points(points, time_limit=5.0):
    """High-precision polish using smooth surrogate and L-BFGS-B."""
    start_time = time.time()
    
    best_pts = points.copy()
    best_score = _calculate_score(points)
    
    # Phase 1: L-BFGS-B with coarse-to-fine adaptive temperature
    temp_schedule = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for temp in temp_schedule:
        if time.time() - start_time > time_limit * 0.6:
            break
        
        def neg_smooth_score(x, t=temp):
            return _smooth_objective(x, t)

        try:
            res = minimize(neg_smooth_score, best_pts.flatten(), method='L-BFGS-B',
                          options={'maxiter':200, 'ftol':1e-14, 'gtol':1e-12, 'maxls':100})
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
        except Exception:
            pass
    
    # Phase 2: Multi-restart local polish from perturbed points
    if time.time() - start_time < time_limit * 0.85:
        rng = np.random.RandomState(42)
        for n_restarts in range(3):
            if time.time() - start_time > time_limit * 0.85:
                break
            
            noise_scale = 0.01 * (n_restarts + 1)
            perturbed = best_pts + rng.randn(N, 2) * noise_scale
            
            def neg_smooth_score_pert(x, t=0.01):
                return _smooth_objective(x, t)
            
            try:
                res = minimize(neg_smooth_score_pert, perturbed.flatten(), method='L-BFGS-B',
                              options={'maxiter':150, 'ftol':1e-14, 'gtol':1e-12, 'maxls':100})
                new_pts = res.x.reshape(N, 2)
                new_score = _calculate_score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
            except Exception:
                pass
    
    # Phase 3: Final coordinate descent polish on exact score
    if time.time() - start_time < time_limit * 0.95:
        remaining = time_limit - (time.time() - start_time)
        best_pts = _coordinate_descent_polish(best_pts, max(1.0, remaining))
        best_score = _calculate_score(best_pts)
    
    return best_pts, best_score

def _optimize_two_ring(n_out, n_in, time_limit=8.0):
    """Optimize two-ring configuration."""
    start_time = time.time()
    
    def objective(params):
        r_in, phi_rel = params
        if r_in < 0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        score = _calculate_score(pts)
        return -score

    best_score = 0
    best_params = None
    
    for r0 in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        for phi0 in [0, np.pi/n_in if n_in > 0 else 0, np.pi/(2*n_in) if n_in > 0 else 0, np.pi/(n_out if n_out > 0 else 1)]:
            x0 = [r0, phi0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                             options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':300})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
            if time.time() - start_time > time_limit:
                return best_score, best_params

    return best_score, best_params

def _optimize_three_ring(time_limit=10.0):
    """Optimize three-ring configuration: 7 outer, 5 middle, 2 inner."""
    start_time = time.time()
    
    def objective(params):
        r1, r2, phi1, phi2 = params
        if r1 < 0 or r1 > 1.5 or r2 < 0 or r2 > 1.5:
            return 1e6
        pts = _generate_three_ring(r1, r2, phi1, phi2)
        score = _calculate_score(pts)
        return -score

    best_score = 0
    best_params = None
    
    rng = np.random.RandomState(99)
    for restart in range(6):
        if time.time() - start_time > time_limit:
            break
        
        if restart == 0:
            x0 = [0.6, 0.3, 0.0, 0.0]
        elif restart == 1:
            x0 = [0.7, 0.4, np.pi/5, np.pi/2]
        elif restart == 2:
            x0 = [0.5, 0.25, np.pi/10, 0.0]
        else:
            x0 = [rng.uniform(0.3, 1.0), rng.uniform(0.1, 0.8), 
                  rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':400})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Also try L-BFGS-B with smooth objective
    if best_params is not None and time.time() - start_time < time_limit * 0.7:
        def smooth_obj(params):
            r1, r2, phi1, phi2 = params
            if r1 < 0 or r1 > 1.5 or r2 < 0 or r2 > 1.5:
                return 1e6
            pts = _generate_three_ring(r1, r2, phi1, phi2)
            return -_soft_min_score(pts, 0.02)
        
        try:
            res = minimize(smooth_obj, best_params, method='L-BFGS-B',
                          options={'maxiter':200, 'ftol':1e-14, 'gtol':1e-12, 'maxls':100})
            new_score = _calculate_score(_generate_three_ring(*res.x))
            if new_score > best_score:
                best_score = new_score
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None:
        pts = _generate_three_ring(*best_params)
        final_score = _calculate_score(pts)
        if final_score > best_score:
            best_score = final_score
    
    return best_score, best_params

def _heptagon_objective(params):
    """Objective for heptagon pair optimization."""
    offsets = params[:7]
    affine_params = params[7:10]
    offsets_clipped = np.clip(offsets, -0.5, 0.5)
    affine_clipped = np.clip(affine_params, [-2, -2, -2], [2, 2, 2])
    pts = _generate_heptagon_pairs(offsets_clipped, affine_clipped)
    score = _calculate_score(pts)
    return -score

def _heptagon_smooth_objective(params, temp=0.03):
    """Smooth objective for heptagon pair optimization."""
    offsets = params[:7]
    affine_params = params[7:10]
    offsets_clipped = np.clip(offsets, -0.5, 0.5)
    affine_clipped = np.clip(affine_params, [-2, -2, -2], [2, 2, 2])
    pts = _generate_heptagon_pairs(offsets_clipped, affine_clipped)
    score = _soft_min_score(pts, temp)
    return -score

def _optimize_heptagon_pairs(time_limit=15.0):
    """Optimize the 10-parameter heptagon pair representation."""
    start_time = time.time()
    best_score = 0
    best_params = None
    
    rng = np.random.RandomState(123)
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if restart == 0:
            offsets = np.zeros(7)
        elif restart == 1:
            offsets = np.ones(7) * 0.1
        elif restart == 2:
            offsets = np.ones(7) * 0.2
        else:
            offsets = rng.uniform(-0.3, 0.3, size=7)
        
        if restart == 0:
            affine = [1.0, 0.0, 0.0]
        else:
            affine = [1.0 + rng.uniform(-0.2, 0.2), 
                      rng.uniform(-0.3, 0.3), 
                      rng.uniform(-0.3, 0.3)]
        
        x0 = np.concatenate([offsets, affine])
        
        try:
            res = minimize(_heptagon_objective, x0, method='Nelder-Mead',
                          options={'xatol':1e-6, 'fatol':1e-6, 'maxiter':500})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        if time.time() - start_time < time_limit * 0.8:
            try:
                res2 = minimize(_heptagon_smooth_objective, x0, method='L-BFGS-B',
                               options={'maxiter':300, 'ftol':1e-12, 'gtol':1e-10, 'maxls':50})
                if -res2.fun > best_score:
                    best_score = -res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass
    
    if best_params is not None:
        offsets = np.clip(best_params[:7], -0.5, 0.5)
        affine = np.clip(best_params[7:10], [-2, -2, -2], [2, 2, 2])
        pts = _generate_heptagon_pairs(offsets, affine)
        final_score = _calculate_score(pts)
        if final_score > best_score:
            best_score = final_score
    
    return best_score, best_params

def _polish_heptagon(params, time_limit=5.0):
    """Polish the heptagon pair parameters with high precision."""
    start_time = time.time()
    best_params = params.copy()
    best_score = _calculate_score(_generate_heptagon_pairs(
        np.clip(params[:7], -0.5, 0.5),
        np.clip(params[7:10], [-2, -2, -2], [2, 2, 2])
    ))
    
    temps = [0.05, 0.03, 0.01, 0.005, 0.002]
    
    for temp in temps:
        if time.time() - start_time > time_limit * 0.7:
            break
        
        def neg_smooth(x, t=temp):
            return _heptagon_smooth_objective(x, t)
        
        try:
            res = minimize(neg_smooth, best_params, method='L-BFGS-B',
                          options={'maxiter':200, 'ftol':1e-14, 'gtol':1e-12, 'maxls':100})
            new_params = res.x
            new_score = _calculate_score(_generate_heptagon_pairs(
                np.clip(new_params[:7], -0.5, 0.5),
                np.clip(new_params[7:10], [-2, -2, -2], [2, 2, 2])
            ))
            if new_score > best_score:
                best_params = new_params
                best_score = new_score
        except Exception:
            pass
    
    if time.time() - start_time < time_limit * 0.9:
        remaining = time_limit - (time.time() - start_time)
        best_params = _coordinate_descent_params(best_params, max(1.0, remaining))
        final_pts = _generate_heptagon_pairs(
            np.clip(best_params[:7], -0.5, 0.5),
            np.clip(best_params[7:10], [-2, -2, -2], [2, 2, 2])
        )
        best_score = _calculate_score(final_pts)
    
    return best_params, best_score

def _coordinate_descent_params(params, time_limit=2.0):
    """Coordinate descent on the 10 heptagon parameters."""
    start_time = time.time()
    best_params = params.copy()
    
    def get_score(p):
        pts = _generate_heptagon_pairs(
            np.clip(p[:7], -0.5, 0.5),
            np.clip(p[7:10], [-2, -2, -2], [2, 2, 2])
        )
        return _calculate_score(pts)
    
    best_score = get_score(best_params)
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step in step_sizes:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(10):
                for direction in [-1, 1]:
                    trial = best_params.copy()
                    trial[i] += direction * step
                    trial_score = get_score(trial)
                    if trial_score > best_score:
                        best_params = trial
                        best_score = trial_score
                        improved = True
                        break
        if not improved:
            break
    
    return best_params

def _generate_four_ring(r1, r2, r3, phi1, phi2, phi3):
    """Generate points on four concentric circles: 6+4+3+1=14 points."""
    angles0 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles1 = np.linspace(0, 2*np.pi, 4, endpoint=False) + phi1
    angles2 = np.linspace(0, 2*np.pi, 3, endpoint=False) + phi2
    angles3 = np.array([phi3])
    
    pts0 = np.column_stack((np.cos(angles0), np.sin(angles0)))
    pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    pts3 = np.array([[r3 * np.cos(phi3), r3 * np.sin(phi3)]])
    
    return np.vstack((pts0, pts1, pts2, pts3))

def _optimize_four_ring(time_limit=8.0):
    """Optimize four-ring configuration: 6 outer, 4 middle, 3 inner, 1 center."""
    start_time = time.time()
    
    def objective(params):
        r1, r2, r3, phi1, phi2, phi3 = params
        if r1 < 0 or r1 > 1.5 or r2 < 0 or r2 > 1.5 or r3 < 0 or r3 > 1.5:
            return 1e6
        pts = _generate_four_ring(r1, r2, r3, phi1, phi2, phi3)
        score = _calculate_score(pts)
        return -score

    best_score = 0
    best_params = None
    
    rng = np.random.RandomState(77)
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
        
        if restart == 0:
            x0 = [0.7, 0.4, 0.15, 0.0, np.pi/4, 0.0]
        elif restart == 1:
            x0 = [0.65, 0.35, 0.1, np.pi/8, np.pi/6, np.pi/3]
        else:
            x0 = [rng.uniform(0.4, 1.0), rng.uniform(0.2, 0.7), rng.uniform(0.0, 0.5),
                  rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':350})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None:
        pts = _generate_four_ring(*best_params)
        final_score = _calculate_score(pts)
        if final_score > best_score:
            best_score = final_score
    
    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Strategy 1: Heptagon pair optimization (10 parameters) - primary approach
    hept_time = min(35.0, deadline - start_time - 40)
    hept_score, hept_params = _optimize_heptagon_pairs(time_limit=hept_time)
    
    if hept_params is not None:
        hept_pts = _generate_heptagon_pairs(
            np.clip(hept_params[:7], -0.5, 0.5),
            np.clip(hept_params[7:10], [-2, -2, -2], [2, 2, 2])
        )
        hept_final_score = _calculate_score(hept_pts)
        try_update(hept_final_score, hept_pts)
        
        if time.time() < deadline - 25:
            remaining = deadline - time.time() - 15
            polished_params, polished_score = _polish_heptagon(hept_params, time_limit=min(18.0, remaining))
            polished_pts = _generate_heptagon_pairs(
                np.clip(polished_params[:7], -0.5, 0.5),
                np.clip(polished_params[7:10], [-2, -2, -2], [2, 2, 2])
            )
            polished_final = _calculate_score(polished_pts)
            try_update(polished_final, polished_pts)
    
    # Strategy 2: Two-ring optimization
    if time.time() < deadline - 30:
        splits = [(14, 0), (13, 1), (12, 2), (11, 3), (10, 4), (9, 5), (8, 6), (7, 7)]
        
        for n_out, n_in in splits:
            if time.time() > deadline - 25:
                break
            
            if n_in == 0:
                pts = _generate_two_ring(n_out, 0, 0, 0)
                polished_pts, polished_score = _polish_points(pts, time_limit=4.0)
                try_update(polished_score, polished_pts)
                continue
            
            remaining_for_ring = deadline - time.time() - 20
            ring_time = max(3.0, min(6.0, remaining_for_ring / 3.0))
            score, params = _optimize_two_ring(n_out, n_in, time_limit=ring_time)
            if params is not None:
                pts = _generate_two_ring(n_out, n_in, params[0], params[1])
                remaining = deadline - time.time() - 15
                polish_time = max(2.0, min(5.0, remaining))
                polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
                try_update(polished_score, polished_pts)
    
    # Strategy 3: Three-ring optimization
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 15
        three_score, three_params = _optimize_three_ring(time_limit=min(8.0, remaining))
        if three_params is not None:
            pts = _generate_three_ring(*three_params)
            final_score = _calculate_score(pts)
            try_update(final_score, pts)
            
            if time.time() < deadline - 10:
                remaining = deadline - time.time() - 8
                polished_pts, polished_score = _polish_points(pts, time_limit=min(6.0, remaining))
                try_update(polished_score, polished_pts)
    
    # Strategy 4: Four-ring optimization
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 12
        four_score, four_params = _optimize_four_ring(time_limit=min(6.0, remaining))
        if four_params is not None:
            pts = _generate_four_ring(*four_params)
            final_score = _calculate_score(pts)
            try_update(final_score, pts)
            
            if time.time() < deadline - 8:
                remaining = deadline - time.time() - 6
                polished_pts, polished_score = _polish_points(pts, time_limit=min(5.0, remaining))
                try_update(polished_score, polished_pts)
    
    # Final intensive polish on best points
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        final_pts, final_score = _polish_points(best_points, time_limit=max(5.0, remaining))
        try_update(final_score, final_pts)
    
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)
        remaining = deadline - time.time() - 3
        polished_pts, polished_score = _polish_points(best_points, time_limit=max(5.0, remaining))
        best_points = polished_pts
        best_score = polished_score
    
    return best_points
