"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(pts):
    """Compute minimum triangle area for n points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1.0

def _score(pts):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-15:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a

def _all_triangle_areas(pts):
    """Compute all triangle areas as a flat array (vectorized)."""
    n = len(pts)
    areas = np.zeros((n * (n-1) * (n-2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                areas[idx] = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                idx += 1
    return areas

def _soft_min_score(pts, beta):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    soft_min(x) = -log(sum(exp(-x/beta))) * beta
    As beta -> 0, this approaches min(x).
    """
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    scaled = -areas / beta
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum_exp * beta
    return soft_min

def _radial_to_pts(angles, radii, center):
    """Convert sorted angles and radii to 2D points."""
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.column_stack((x, y))

def _pts_to_radial(pts, center):
    """Convert 2D points to angles and radii relative to center."""
    dx = pts[:, 0] - center[0]
    dy = pts[:, 1] - center[1]
    angles = np.arctan2(dy, dx)
    # Normalize angles to [0, 2*pi)
    angles = angles % (2 * np.pi)
    # Sort by angle
    order = np.argsort(angles)
    angles_sorted = angles[order]
    radii_sorted = np.sqrt(dx[order]**2 + dy[order]**2)
    return angles_sorted, radii_sorted

def _radial_objective(params, beta, center):
    """
    Objective in radial parameterization.
    params = [angles (N), radii (N)]
    We maximize min triangle area / hull area.
    """
    angles = params[:N]
    radii = params[N:]
    pts = _radial_to_pts(angles, radii, center)
    score = _soft_min_score(pts, beta)
    return -score

def _radial_grad(params, beta, center):
    """
    Gradient of negative soft-min score with respect to radial params.
    """
    angles = params[:N]
    radii = params[N:]
    pts = _radial_to_pts(angles, radii, center)
    n = len(pts)
    
    # Compute all triangle areas and gradients w.r.t. point coords
    areas = np.zeros(1)
    grads = []
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                area = 0.5 * abs(cross)
                areas = np.append(areas, area)
                
                sign = 1.0 if cross >= 0 else -1.0
                
                g_ax = -0.5 * sign * (cy - ay)
                g_ay = 0.5 * sign * (cx - ax)
                g_bx = 0.5 * sign * (cy - ay)
                g_by = -0.5 * sign * (cx - ax)
                g_cx = 0.5 * sign * (by - ay)
                g_cy = -0.5 * sign * (bx - ax)
                
                grad_vec = np.zeros(2 * n)
                grad_vec[2*i] = g_ax
                grad_vec[2*i+1] = g_ay
                grad_vec[2*j] = g_bx
                grad_vec[2*j+1] = g_by
                grad_vec[2*k] = g_cx
                grad_vec[2*k+1] = g_cy
                grads.append(grad_vec)
    
    areas = areas[1:]
    grads = np.array(grads)
    
    # Soft-min weights
    scaled = -areas / beta
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # Total gradient w.r.t. point coordinates
    grad_pts = np.sum(weights[:, np.newaxis] * grads, axis=0)
    
    # Chain rule: d(pts)/d(angles) and d(pts)/d(radii)
    # pts[i] = center + radii[i] * [cos(angles[i]), sin(angles[i])]
    # d(pts[i])/d(angles[i]) = radii[i] * [-sin(angles[i]), cos(angles[i])]
    # d(pts[i])/d(radii[i]) = [cos(angles[i]), sin(angles[i])]
    
    grad_angles = np.zeros(N)
    grad_radii = np.zeros(N)
    
    for i in range(N):
        gx = grad_pts[2*i]
        gy = grad_pts[2*i+1]
        ca = np.cos(angles[i])
        sa = np.sin(angles[i])
        r_i = radii[i]
        
        # d(pts[i])/d(angles[i]) = r_i * [-sa, ca]
        grad_angles[i] = gx * (-r_i * sa) + gy * (r_i * ca)
        
        # d(pts[i])/d(radii[i]) = [ca, sa]
        grad_radii[i] = gx * ca + gy * sa
    
    return np.concatenate([grad_angles, grad_radii])

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def _center_ring(n, inner_radius=0.15, outer_radius=0.5, center=(0.5, 0.5)):
    """Generate points: one at center, rest on a ring."""
    center_pt = np.array([center])
    ring_n = n - 1
    angles = np.linspace(0, 2 * np.pi, ring_n, endpoint=False)
    radius = (inner_radius + outer_radius) / 2
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    ring_pts = np.column_stack((x, y))
    return np.vstack([center_pt, ring_pts])

def _halton_sequence(n, base):
    """Generate Halton sequence in [0, 1)."""
    result = []
    f = 1.0
    r = 0.0
    for i in range(1, n + 1):
        f = 1.0 / base
        a = i
        while a > 0:
            r += f * (a % base)
            a //= base
            f /= base
        result.append(r)
    return result

def _halton_points(n, center=(0.5, 0.5), radius=0.45):
    """Generate points using Halton sequence mapped to a disk."""
    x_seq = _halton_sequence(n, 2)
    y_seq = _halton_sequence(n, 3)
    theta = 2 * np.pi * np.array(x_seq)
    r = radius * np.sqrt(np.array(y_seq))
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    center = (0.5, 0.5)
    
    # Initial guesses
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    starts = [
        _regular_polygon(N, radius=0.5, center=center),
        _regular_polygon(N, radius=0.45, center=center),
        _regular_polygon(N, radius=0.55, center=center),
        _center_ring(N, inner_radius=0.1, outer_radius=0.5, center=center),
        _center_ring(N, inner_radius=0.2, outer_radius=0.5, center=center),
        _halton_points(N, center=center),
    ]
    
    for start in starts:
        if time.time() > deadline - 10:
            break
        s = _score(start)
        if s > best_score:
            best_score = s
            best_pts = start.copy()
    
    # Convert best points to radial parameters
    angles0, radii0 = _pts_to_radial(best_pts, center)
    x0 = np.concatenate([angles0, radii0])
    
    # Radial parameterization optimization
    # Bounds: angles in [0, 2*pi), radii in [0.01, 0.5]
    angle_bounds = [(0.0, 2 * np.pi)] * N
    radius_bounds = [(0.01, 0.5)] * N
    bounds = angle_bounds + radius_bounds
    
    # Annealing schedule for beta
    beta_schedule = [50, 25, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for beta in beta_schedule:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _radial_objective,
                x0,
                args=(beta, center),
                jac=_radial_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
            )
            
            new_angles = result.x[:N]
            new_radii = result.x[N:]
            new_pts = _radial_to_pts(new_angles, new_radii, center)
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
            
            x0 = result.x.copy()
            
        except Exception:
            pass
    
    # Final polish with very small beta
    if time.time() < deadline - 10:
        for beta in [0.0002, 0.0001, 0.00005]:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(
                    _radial_objective,
                    x0,
                    args=(beta, center),
                    jac=_radial_grad,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
                )
                
                new_angles = result.x[:N]
                new_radii = result.x[N:]
                new_pts = _radial_to_pts(new_angles, new_radii, center)
                new_score = _score(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                
                x0 = result.x.copy()
            except Exception:
                pass
    
    # Try a few radial restarts if time remains
    if time.time() < deadline - 30:
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            # Random radial initialization
            rand_angles = np.sort(rng.uniform(0, 2 * np.pi, N))
            rand_radii = rng.uniform(0.3, 0.5, N)
            x0 = np.concatenate([rand_angles, rand_radii])
            
            for beta in [10, 1, 0.1, 0.01]:
                if time.time() > deadline - 15:
                    break
                try:
                    result = minimize(
                        _radial_objective,
                        x0,
                        args=(beta, center),
                        jac=_radial_grad,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 30}
                    )
                    
                    new_angles = result.x[:N]
                    new_radii = result.x[N:]
                    new_pts = _radial_to_pts(new_angles, new_radii, center)
                    new_score = _score(new_pts)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
                    
                    x0 = result.x.copy()
                except Exception:
                    break
    
    return best_pts
# EVOLVE-BLOCK-END
