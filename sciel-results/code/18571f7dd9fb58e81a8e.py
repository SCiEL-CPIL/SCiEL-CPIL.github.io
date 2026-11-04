import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)
])

def _compute_areas(points):
    """Compute all triangle areas efficiently using precomputed cross products."""
    p = points  # (N, 2)
    C = p[:, 0:1] * p[:, 1:2].T - p[:, 1:2] * p[:, 0:1].T
    i_idx = TRIPLE_INDICES[:, 0]
    j_idx = TRIPLE_INDICES[:, 1]
    k_idx = TRIPLE_INDICES[:, 2]
    vals = C[j_idx, k_idx] + C[i_idx, j_idx] - C[i_idx, k_idx]
    return 0.5 * np.abs(vals)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, temperature=0.015):
    """Smooth surrogate: soft-minimum of triangle areas / hull area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    s = np.sum(np.exp(shifted / temperature))
    soft_min = m - temperature * np.log(s)
    return soft_min / hull_area

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + phi_rel
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    return pts_out

def _generate_three_ring(n1, n2, n3, r2, r3, phi2, phi3):
    """Three concentric rings: outer (unit), middle (r2), inner (r3)."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    p1 = np.column_stack((np.cos(a1), np.sin(a1)))
    a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phi2
    p2 = np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2)))
    a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + phi3
    p3 = np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
    return np.vstack((p1, p2, p3))

def _optimize_two_ring(n_out, n_in, time_limit=8.0):
    start_time = time.time()
    best_score = 0.0
    best_params = None

    def objective(params):
        r_in, phi_rel = params
        if r_in < 0.0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        return -_calculate_score(pts)

    r_starts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    phi_starts = [0.0]
    if n_in > 0:
        phi_starts += [np.pi / n_in, np.pi / (2.0 * n_in), 2.0 * np.pi / n_in,
                       np.pi / (3.0 * n_in), np.pi / (4.0 * n_in)]
    if n_out > 0:
        phi_starts += [np.pi / n_out, np.pi / (2.0 * n_out)]

    for r0 in r_starts:
        for phi0 in phi_starts:
            if time.time() - start_time > time_limit:
                return best_score, best_params
            try:
                res = minimize(objective, [r0, phi0], method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500, 'disp': False})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    return best_score, best_params

def _optimize_three_ring(n1, n2, n3, time_limit=8.0):
    start_time = time.time()
    best_score = 0.0
    best_params = None

    def objective(params):
        r2, r3, phi2, phi3 = params
        if r2 < 0.0 or r2 > 1.5 or r3 < 0.0 or r3 > 1.5:
            return 1e6
        pts = _generate_three_ring(n1, n2, n3, r2, r3, phi2, phi3)
        return -_calculate_score(pts)

    r_starts = [0.3, 0.5, 0.7, 0.9]
    r3_starts = [0.2, 0.4, 0.6, 0.8]
    phi_starts = [0.0, np.pi / n2 if n2 > 0 else 0.0, np.pi / n3 if n3 > 0 else 0.0]

    count = 0
    for r2 in r_starts:
        for r3 in r3_starts:
            for phi2 in phi_starts:
                for phi3 in phi_starts:
                    if time.time() - start_time > time_limit or count >= 20:
                        return best_score, best_params
                    count += 1
                    try:
                        res = minimize(objective, [r2, r3, phi2, phi3], method='Nelder-Mead',
                                       options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 400, 'disp': False})
                        if -res.fun > best_score:
                            best_score = -res.fun
                            best_params = res.x.copy()
                    except Exception:
                        pass
    return best_score, best_params

def _coordinate_descent_polish(points, time_limit=5.0):
    """Optimize each point individually while holding others fixed."""
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    for iteration in range(8):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            def obj_2d(x):
                p[i] = x
                return -_calculate_score(p)
            try:
                res = minimize(obj_2d, p[i], method='Nelder-Mead',
                               options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 800, 'disp': False})
                new_s = _calculate_score(p)
                if -res.fun > s + 1e-12:
                    p[i] = res.x
                    s = -res.fun
                    improved = True
                else:
                    p[i] = points[i]
            except Exception:
                pass
        if not improved:
            break
    return s, p

def _soft_min_polish(points, time_limit=10.0):
    """Polish using soft-min surrogate with L-BFGS-B for smooth gradient exploitation."""
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)
    
    # Two-stage: broad exploration then sharp refinement
    temps = [0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.003, 0.0015, 0.001]
    for t in temps:
        if time.time() - start_time > time_limit:
            break
        
        def obj_soft(x):
            pts = x.reshape(N, 2)
            return -_soft_min_score(pts, temperature=t)
        
        def obj_soft_grad(x):
            pts = x.reshape(N, 2)
            val = -_soft_min_score(pts, temperature=t)
            # Numerical gradient for L-BFGS-B
            g = np.zeros_like(x)
            eps = 1e-6
            for i in range(len(x)):
                x_p = x.copy()
                x_m = x.copy()
                x_p[i] += eps
                x_m[i] -= eps
                g[i] = (obj_soft(x_p) - obj_soft(x_m)) / (2 * eps)
            return val, g
        
        try:
            res = minimize(obj_soft_grad, p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
            new_p = res.x.reshape(N, 2)
            new_s = _calculate_score(new_p)
            if new_s > s:
                s = new_s
                p = new_p
        except Exception:
            try:
                res = minimize(obj_soft, p.flatten(), method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 400, 'disp': False})
                new_p = res.x.reshape(N, 2)
                new_s = _calculate_score(new_p)
                if new_s > s:
                    s = new_s
                    p = new_p
            except Exception:
                pass
    
    return s, p

def _generate_regular_polygon(r=1.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack((x, y))

def _generate_random_points(seed):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_points = None
    
    # 1. Start with Regular 14-gon
    reg_points = _generate_regular_polygon(1.0)
    score = _calculate_score(reg_points)
    if score > best_score:
        best_score = score
        best_points = reg_points.copy()
    
    # 2. Try Two-Ring configurations with parameter optimization (reduced time)
    ring_configs = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
    ]
    
    for n_out, n_in in ring_configs:
        if time.time() > deadline - 55:
            break
        score, params = _optimize_two_ring(n_out, n_in, time_limit=3.0)
        if score > best_score:
            best_score = score
            if params is not None:
                best_points = _generate_two_ring(n_out, n_in, params[0], params[1])
    
    # 3. Try Three-Ring configurations (reduced time)
    three_ring_configs = [
        (5, 5, 4),
        (6, 4, 4),
        (7, 4, 3),
    ]
    
    for n1, n2, n3 in three_ring_configs:
        if time.time() > deadline - 45:
            break
        score, params = _optimize_three_ring(n1, n2, n3, time_limit=2.5)
        if score > best_score:
            best_score = score
            if params is not None:
                best_points = _generate_three_ring(n1, n2, n3, params[0], params[1], params[2], params[3])
    
    # 4. Random Starts (fewer, just for diversity)
    for seed in range(1, 8):
        if time.time() > deadline - 40:
            break
        start_pts = _generate_random_points(seed)
        score = _calculate_score(start_pts)
        if score > best_score:
            best_score = score
            best_points = start_pts.copy()
    
    # 5. Initial coordinate descent polish
    if best_points is not None and time.time() < deadline - 30:
        score, pts = _coordinate_descent_polish(best_points, time_limit=8.0)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # 6. Main soft-min polish with L-BFGS-B (the key improvement)
    if best_points is not None and time.time() < deadline - 12:
        score, pts = _soft_min_polish(best_points, time_limit=25.0)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # 7. Second round of coordinate descent after soft-min
    if best_points is not None and time.time() < deadline - 6:
        score, pts = _coordinate_descent_polish(best_points, time_limit=6.0)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # 8. Final soft-min polish with tighter temperatures
    if best_points is not None and time.time() < deadline - 2:
        score, pts = _soft_min_polish(best_points, time_limit=4.0)
        if score > best_score:
            best_score = score
            best_points = pts
    
    if best_points is None:
        best_points = reg_points
        
    return best_points
