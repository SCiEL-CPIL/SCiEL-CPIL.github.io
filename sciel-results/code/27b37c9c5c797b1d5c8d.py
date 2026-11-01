import numpy as np
import scipy.spatial as spat
import scipy.optimize as opt
import time
import itertools

N = 14

def calc_score(points):
    """Compute the exact score: min triangle area / convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, k + 1):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area / hull_area


def _min_triangle_area_and_indices(points):
    """Return (min_area, list_of_active_triples) where active means area within 1e-9 of min."""
    n = len(points)
    min_area = float('inf')
    all_areas = []
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                all_areas.append(area)
                triples.append((i, j, k))
                if area < min_area:
                    min_area = area
    # Collect active triples (within tolerance of min)
    tol = 1e-9 * max(1.0, min_area)
    active = []
    for idx, area in enumerate(all_areas):
        if area <= min_area + tol:
            active.append(triples[idx])
    return min_area, active


def _grad_min_area_from_active(points, active_triples):
    """
    Compute gradient of the minimum area with respect to point coordinates.
    Uses only the active (minimum-area) triples.
    Returns gradient vector of shape (2*N,).
    """
    n = len(points)
    grad = np.zeros(2 * n)
    for (i, j, k) in active_triples:
        pi = points[i]
        pj = points[j]
        pk = points[k]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        s = np.sign(cross)
        if s == 0:
            s = 1.0

        # dA/dpi, dA/dpj, dA/dpk for A = 0.5 * |cross|
        dA_dpi_x = 0.5 * s * (pj[1] - pk[1])
        dA_dpi_y = 0.5 * s * (pj[0] - pk[0])
        dA_dpj_x = 0.5 * s * (pk[1] - pi[1])
        dA_dpj_y = 0.5 * s * (-(pk[0] - pi[0]))
        dA_dpk_x = 0.5 * s * (-(pj[1] - pi[1]))
        dA_dpk_y = 0.5 * s * (pj[0] - pi[0])

        grad[2 * i + 0] += dA_dpi_x
        grad[2 * i + 1] += dA_dpi_y
        grad[2 * j + 0] += dA_dpj_x
        grad[2 * j + 1] += dA_dpj_y
        grad[2 * k + 0] += dA_dpk_x
        grad[2 * k + 1] += dA_dpk_y

    return grad


def _two_ring_points(params):
    """
    Construct 14 points from a two-ring parametric model.
    params = [r_inner, angle_offset_inner, angle_offset_outer, aspect]
    Outer ring: 7 points on ellipse (cos, aspect*sin) rotated by angle_offset_outer
    Inner ring: 7 points on circle of radius r_inner rotated by angle_offset_inner
    Points are centered at origin.
    """
    r_inner = params[0]
    ang_off_inner = params[1]
    ang_off_outer = params[2]
    aspect = params[3]

    n_outer = 7
    n_inner = 7

    # Outer ring
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + ang_off_outer
    outer_x = np.cos(angles_outer)
    outer_y = aspect * np.sin(angles_outer)
    outer_pts = np.column_stack((outer_x, outer_y))

    # Inner ring
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + ang_off_inner
    inner_x = r_inner * np.cos(angles_inner)
    inner_y = r_inner * np.sin(angles_inner)
    inner_pts = np.column_stack((inner_x, inner_y))

    points = np.vstack([outer_pts, inner_pts])
    return points


def _neg_score_two_ring(params):
    """Negative score for the two-ring parametric model."""
    points = _two_ring_points(params)
    return -calc_score(points)


def _local_gradient_ascent(points, max_iters=500, step_size=1e-3):
    """
    Local gradient ascent on the exact min triangle area (not normalized).
    Uses active triples for gradient. Normalizes by hull area at the end.
    """
    current = points.copy()
    best_score = calc_score(current)
    
    for it in range(max_iters):
        min_area, active = _min_triangle_area_and_indices(current)
        if len(active) == 0:
            break
        
        grad = _grad_min_area_from_active(current, active)
        
        # Normalize gradient by current hull area to approximate score gradient
        try:
            hull = spat.ConvexHull(current)
            ha = hull.volume
        except Exception:
            ha = 1.0
        if ha < 1e-10:
            break
        
        # Gradient ascent step
        current = current + step_size * grad / ha
        
        # Keep points in a reasonable range
        current -= current.mean(axis=0)
        max_norm = np.max(np.sqrt(np.sum(current**2, axis=1)))
        if max_norm > 10:
            current /= max_norm * 0.5
        
        score = calc_score(current)
        if score > best_score:
            best_score = score
        else:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-8:
                break
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    best_points = None
    best_score = 0.0

    # Phase 1: Optimize the two-ring parametric model
    # Initial parameters: [r_inner, ang_off_inner, ang_off_outer, aspect]
    initial_params_list = [
        [0.5, 0.0, 0.0, 1.0],
        [0.5, np.pi/7, 0.0, 1.0],
        [0.5, 0.0, np.pi/7, 1.0],
        [0.5, np.pi/7, np.pi/7, 1.0],
        [0.6, 0.0, 0.0, 1.0],
        [0.4, 0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0, 0.8],
        [0.5, 0.0, 0.0, 1.2],
        [0.5, np.pi/7, 0.0, 0.9],
        [0.5, 0.0, np.pi/7, 1.1],
    ]

    for params0 in initial_params_list:
        if time.time() > deadline:
            break
        
        try:
            res = opt.minimize(
                _neg_score_two_ring,
                params0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                         'adaptive': True, 'maxfev': 3000}
            )
            if time.time() > deadline:
                break
            
            opt_params = res.x
            # Ensure r_inner is positive
            opt_params[0] = max(0.1, opt_params[0])
            opt_params[3] = max(0.3, min(3.0, opt_params[3]))
            
            points = _two_ring_points(opt_params)
            score = calc_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            pass

    # Phase 2: Local gradient ascent from the best two-ring configuration
    if best_points is not None and time.time() < deadline - 30:
        try:
            refined = _local_gradient_ascent(best_points, max_iters=300, step_size=1e-3)
            score = calc_score(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()
        except Exception:
            pass

    # Phase 3: Try a few random perturbations and local refine
    if best_points is not None:
        for trial in range(5):
            if time.time() > deadline - 20:
                break
            perturbed = best_points + np.random.randn(N, 2) * 0.05
            try:
                refined = _local_gradient_ascent(perturbed, max_iters=200, step_size=1e-3)
                score = calc_score(refined)
                if score > best_score:
                    best_score = score
                    best_points = refined.copy()
            except Exception:
                pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
