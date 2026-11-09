# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # 20s safety margin before 600s kill

def _triangle_areas(points):
    """Compute all C(n,3) triangle areas for n points. Returns array of areas."""
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _min_triangle_area(points):
    """Minimum triangle area."""
    return _triangle_areas(points).min()

def _hull_area(points):
    """Area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return 0.0

def _score(points):
    """A_min / A_hull."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or a_min <= 0:
        return 0.0
    return a_min / a_hull

def _soft_min_objective(flat, beta):
    """
    Negative soft-min of triangle areas (to maximize min area).
    Uses log-sum-exp for numerical stability.
    Also penalizes if hull area is too small or points are degenerate.
    """
    points = flat.reshape(-1, 2)
    areas = _triangle_areas(points)
    
    # Soft-min via log-sum-exp: -log(sum(exp(-beta * a_i))) / beta approximates min(a_i)
    # For numerical stability:
    min_a = areas.min()
    if min_a < 1e-15:
        return 1e6  # Large penalty for near-zero area
    
    # log-sum-exp trick
    shifted = -beta * (areas - min_a)  # shift by min for stability
    max_shifted = shifted.max()
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = min_a - log_sum / beta
    
    # Hull area
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 1e6
    
    if a_hull < 1e-10:
        return 1e6
    
    # We want to maximize soft_min / a_hull, so minimize negative
    return -soft_min / a_hull

def _soft_min_objective_grad(flat, beta):
    """Gradient of the soft-min objective (approximate via finite differences for L-BFGS-B)."""
    # Use analytical gradient where possible, but for simplicity use numerical
    # Actually, let's provide an analytical gradient
    points = flat.reshape(-1, 2)
    n = len(points)
    areas = _triangle_areas(points)
    
    min_a = areas.min()
    if min_a < 1e-15:
        return np.ones_like(flat)
    
    shifted = -beta * (areas - min_a)
    max_shifted = shifted.max()
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    
    # d(soft_min)/d(a_i) = -exp_shifted[i] / sum_exp
    d_softmin_da = -exp_shifted / sum_exp
    
    # Gradient of each area w.r.t. point coordinates
    idx = np.array(list(itertools.combinations(range(n), 3)))
    grad_flat = np.zeros_like(flat)
    
    for k in range(len(idx)):
        w = d_softmin_da[k]
        if abs(w) < 1e-15:
            continue
        i, j, l = idx[k]
        pi = points[i]
        pj = points[j]
        pl = points[l]
        
        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # = 0.5 * |(xj-xi)(yl-yi) - (yj-yi)(xl-xi)|
        # For gradient, we need sign. Let's use the cross product form.
        v1 = pj - pi
        v2 = pl - pi
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sign = np.sign(cross)
        if sign == 0:
            sign = 1.0
        
        # d(area)/d(xi) = -0.5 * sign * (yl - yj)
        # d(area)/d(yi) = 0.5 * sign * (xl - xj)
        # d(area)/d(xj) = 0.5 * sign * (yl - yi)
        # d(area)/d(yj) = -0.5 * sign * (xl - xi)
        # d(area)/d(xl) = -0.5 * sign * (yj - yi)
        # d(area)/d(yl) = 0.5 * sign * (xj - xi)
        
        dxi = -0.5 * sign * (pl[1] - pj[1])
        dyi = 0.5 * sign * (pl[0] - pj[0])
        dxj = 0.5 * sign * (pl[1] - pi[1])
        dyj = -0.5 * sign * (pl[0] - pi[0])
        dxl = -0.5 * sign * (pj[1] - pi[1])
        dyl = 0.5 * sign * (pj[0] - pi[0])
        
        grad_flat[2*i] += w * dxi
        grad_flat[2*i+1] += w * dyi
        grad_flat[2*j] += w * dxj
        grad_flat[2*j+1] += w * dyj
        grad_flat[2*l] += w * dxl
        grad_flat[2*l+1] += w * dyl
    
    # Now we need d(objective)/d(flat) where objective = -soft_min / a_hull
    # = (-d(soft_min)/d(flat) * a_hull + soft_min * d(a_hull)/d(flat)) / a_hull^2
    # This is complex. For L-BFGS-B, let's just use numerical gradient or 
    # simplify by fixing the hull normalization.
    
    # Simpler approach: just return the gradient of -soft_min and let the 
    # optimizer handle the scaling. Actually, let's just use a simpler objective
    # that maximizes min area directly (since hull area changes slowly).
    
    return -grad_flat  # We want to maximize soft_min, so minimize -soft_min

def _optimize_from_start(points, beta_schedule, max_time=None):
    """Optimize starting from given points with a beta schedule."""
    flat = points.flatten()
    n = len(points)
    
    best_flat = flat.copy()
    best_score = _score(points)
    
    current = flat.copy()
    
    for beta in beta_schedule:
        if max_time is not None and time.time() > max_time:
            break
        
        # L-BFGS-B optimization
        try:
            result = minimize(
                _soft_min_objective,
                current,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success or result.fun < 0:
                candidate = result.x.reshape(n, 2)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_flat = result.x.copy()
                current = result.x.copy()
        except Exception:
            pass
        
        # Powell polish
        try:
            result2 = minimize(
                lambda f: -_score(f.reshape(n, 2)),
                current,
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-14, 'ftol': 1e-16}
            )
            if result2.success:
                candidate = result2.x.reshape(n, 2)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_flat = result2.x.copy()
                current = result2.x.copy()
        except Exception:
            pass
    
    return best_flat.reshape(n, 2), best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = _DEADLINE
    
    # Start with a reasonable initial configuration
    # Try a regular polygon + center + some interior points
    best_points = None
    best_score = 0.0
    
    # Initial configurations to try
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Regular 12-gon + center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    poly12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([poly12, center]))
    
    # Regular 11-gon + 2 interior points
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    poly11 = np.column_stack([np.cos(angles11), np.sin(angles11)])
    interior = np.array([[0.3, 0.3], [-0.3, -0.3]])
    inits.append(np.vstack([poly11, interior]))
    
    # Random initializations
    rng = np.random.default_rng(seed=42)
    for seed in [42, 123, 456, 789, 101]:
        rng = np.random.default_rng(seed=seed)
        pts = rng.random((n, 2))
        # Scale to unit square
        pts = pts - pts.min(axis=0)
        pts = pts / (pts.max(axis=0) + 1e-10)
        inits.append(pts)
    
    # Beta schedule: start smooth, get sharper
    beta_schedule = [20, 50, 100, 200, 500, 1000]
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30:  # Need 30s for final polish
            break
        
        # Evaluate initial score
        s_init = _score(init_pts)
        if s_init > best_score:
            best_score = s_init
            best_points = init_pts.copy()
        
        # Optimize
        try:
            opt_pts, opt_score = _optimize_from_start(
                init_pts, 
                beta_schedule,
                max_time=deadline - 30
            )
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts
        except Exception:
            pass
    
    # Final high-precision polish on the best point
    if best_points is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                lambda f: -_score(f.reshape(n, 2)),
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-16, 'ftol': 1e-18}
            )
            if result.success:
                final_pts = result.x.reshape(n, 2)
                final_score = _score(final_pts)
                if final_score > best_score:
                    best_points = final_pts
                    best_score = final_score
        except Exception:
            pass
    
    # Ensure valid output
    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Verify it's valid
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

# EVOLVE-BLOCK-END
