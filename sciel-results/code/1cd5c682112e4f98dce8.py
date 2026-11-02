"""Combined program for Heilbronn convex regions n=14.
Strategy: 
1. Structural change: Parameterize points by angles on the unit circle to enforce convex position.
2. Optimization: Nelder-Mead on angles to find optimal angular distribution.
3. Expansion: Allow radial degrees of freedom to refine the solution.
4. Refinement: Coordinate descent and Powell for high-precision polishing.
"""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# ---------- Core Geometry ----------

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(points_flat):
    """Minimize hull_area / min_triangle_area (scale-invariant)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_soft(points_flat, beta=150.0):
    """Smooth soft-min objective: hull_area / soft_min(triangle areas)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

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

# ---------- Angular Parameterization Helpers ----------

def _angles_to_points(angles):
    """Convert angles to points on unit circle."""
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _objective_angular(angles_flat):
    """Objective for angular optimization. Points are on unit circle."""
    angles = angles_flat
    # Ensure angles are sorted and within [0, 2pi) relative to first point?
    # Actually, for a closed loop, we can just use the angles as is, 
    # but we must ensure they represent a convex polygon.
    # Sorting them guarantees convex position if we connect in sorted order.
    sorted_angles = np.sort(angles)
    points = _angles_to_points(sorted_angles)
    
    # If points are collinear or degenerate, return high cost
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _coordinate_descent(pts, deadline, max_rounds=30):
    """Cyclic coordinate descent on the hard objective."""
    pts = pts.copy()
    n = len(pts)
    step0 = 0.02
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                for _ in range(20):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig
                    if s_plus > s_minus:
                        pts[i, coord] = orig + step
                    elif s_minus > s_plus:
                        pts[i, coord] = orig - step
                    else:
                        break
                    improved = True
                    step *= 0.5
                    if step < 1e-8:
                        break
        if not improved:
            break
    return pts

# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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

    # --- Phase 1: Angular Optimization ---
    # Search for optimal angular distribution on the unit circle.
    # This enforces convex position and reduces dimensionality.
    
    inits_angular = []
    rng = np.random.default_rng(42)
    
    # Regular 14-gon angles
    reg_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits_angular.append(reg_angles.copy())
    
    # Jittered regular polygons
    for js in [0.01, 0.05, 0.1, 0.2]:
        inits_angular.append(reg_angles + rng.normal(0, js, N))
        
    # Random sorted angles
    for seed in range(10):
        rs = np.random.default_rng(seed)
        a = rs.uniform(0, 2*np.pi, N)
        inits_angular.append(np.sort(a))
        
    # Two-cluster angles (simulating inner/outer structure in angle space)
    for shift in [0.1, 0.2, 0.5]:
        a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + shift
        inits_angular.append(np.sort(np.concatenate([a1, a2])))

    for x0 in inits_angular:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(_objective_angular, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < 1e9:
                pts = _angles_to_points(np.sort(res.x))
                _consider(pts)
        except Exception:
            pass

    # --- Phase 2: Radial Expansion & Cartesian Refinement ---
    # If we have a good angular start, allow radial movement.
    if best_pts is not None:
        # Try to optimize radii while keeping angles fixed initially?
        # Or just switch to full Cartesian NM starting from the angular solution.
        
        # 2a. Nelder-Mead in Cartesian space starting from angular solution
        if time.time() < deadline - 30:
            try:
                res = minimize(_objective_hard, best_pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < 1e9:
                    _consider(res.x.reshape(N, 2))
            except Exception:
                pass

        # 2b. Coordinate Descent for fine-tuning
        if time.time() < deadline - 20:
            refined = _coordinate_descent(best_pts, deadline)
            _consider(refined)
            
        # 2c. Powell for high precision
        if time.time() < deadline - 15:
            try:
                res = minimize(_objective_hard, best_pts.flatten(), method='Powell',
                               options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-14})
                if res.fun < 1e9:
                    _consider(res.x.reshape(N, 2))
            except Exception:
                pass

    # --- Phase 3: Alternative Starts in Cartesian Space ---
    # In case the angular search got stuck in a local minimum, try some direct Cartesian starts.
    
    cart_inits = []
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cart_inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # Jittered
    for js in [0.1, 0.2]:
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        pts += rng.normal(0, js, size=(N, 2))
        cart_inits.append(pts.flatten())
        
    # Random in square
    for seed in range(5):
        rs = np.random.default_rng(seed * 100)
        pts = rs.uniform(-1, 1, size=(N, 2))
        cart_inits.append(pts.flatten())

    for x0 in cart_inits:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(_objective_hard, x0, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.fun < 1e9:
                _consider(res.x.reshape(N, 2))
        except Exception:
            pass

    # --- Final Polish ---
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res = minimize(_objective_soft, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < 1e9:
                _consider(res.x.reshape(N, 2))
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    final_points = _normalize_unit_area(best_pts)

    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
# EVOLVE-BLOCK-END
