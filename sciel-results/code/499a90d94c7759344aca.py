import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
from itertools import combinations

N = 14

# Pre-compute all triangle index combinations for vectorization
tri_indices = np.array(list(combinations(range(N), 3)))

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    Vectorized for speed.
    """
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    # Cross product of (pj-pi) and (pk-pi)
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score is min_triangle_area / convex_hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _decode_bspline(ctrl_params):
    """
    Decode 14-dim control point parameters to 14 points on a B-spline curve.
    ctrl_params: shape (14,) = [x0,y0, x1,y1, ..., x6,y6]
    """
    ctrl = ctrl_params.reshape(7, 2)
    
    # Check for degeneracy
    if np.allclose(ctrl, ctrl[0]):
        return np.zeros((N, 2))
        
    try:
        # Create a closed cubic B-spline
        # For periodic boundary conditions, we need at least 4 points, but 7 is fine.
        # We repeat the first point to close the curve for interpolation logic if needed, 
        # but CubicSpline with bc_type='periodic' handles it.
        
        t = np.arange(7)
        
        # Create cubic splines for x and y
        cs_x = CubicSpline(t, ctrl[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, ctrl[:, 1], bc_type='periodic')
        
        # Sample many points along the curve to compute arc length
        # Use a fine grid
        t_fine = np.linspace(0, 7, 2000)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        
        # Compute arc length
        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        
        # Total arc length
        total_s = s[-1]
        
        if total_s < 1e-6:
             return np.zeros((N, 2))

        # Sample 14 points at equal arc-length intervals
        s_targets = np.linspace(0, total_s, N, endpoint=False)
        
        # Interpolate to get points at target arc lengths
        x_pts = np.interp(s_targets, s, x_fine)
        y_pts = np.interp(s_targets, s, y_fine)
        
        pts = np.column_stack([x_pts, y_pts])
        
        # Center the points to remove translation invariance
        pts -= pts.mean(axis=0)
        
        # Normalize scale? The score is scale invariant. 
        # However, for numerical stability in the optimizer, it's good to keep them in a reasonable range.
        # We can normalize so that the max norm is 1.
        norm = np.max(np.linalg.norm(pts, axis=1))
        if norm > 1e-6:
            pts = pts / norm
            
        return pts
        
    except Exception:
        # Fallback: Regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        return pts

def _objective_bspline(ctrl_params):
    """
    Objective function for B-spline control points.
    Minimizes negative score.
    """
    pts = _decode_bspline(ctrl_params)
    score = _score(pts)
    return -score

def _generate_initial_configs():
    """
    Generate diverse initial control point configurations.
    """
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular heptagon
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(ctrl.flatten().copy())
    
    # 2. Elliptical shapes
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.5, 0.6), (0.6, 1.5)]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
        
    # 3. Perturbed regular heptagon
    for js in [0.05, 0.1, 0.2, 0.3]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
        ctrl += rng.normal(0, js, size=(7, 2))
        inits.append(ctrl.flatten().copy())
        
    # 4. Random convex-ish polygons
    for seed in range(20):
        rs = np.random.default_rng(seed * 13 + 7)
        # Generate angles in sorted order
        angles = np.sort(rs.uniform(0, 2 * np.pi, 7))
        # Ensure they are spread out
        # Add random radii
        radii = rs.uniform(0.5, 1.5, 7)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
        
    # 5. "Square-like" control points (7 points on a square perimeter)
    # This helps find solutions that look like squares/rectangles
    pts_sq = []
    # 2 on each side of a unit square
    # Side 1: (0,0) to (1,0) -> (0.25, 0), (0.75, 0)
    # Side 2: (1,0) to (1,1) -> (1, 0.25), (1, 0.75)
    # Side 3: (1,1) to (0,1) -> (0.75, 1), (0.25, 1)
    # Side 4: (0,1) to (0,0) -> (0, 0.75), (0, 0.25)
    # We need 7 points. Let's use 2 on 3 sides and 1 on the 4th? Or just random points on a square.
    for seed in range(5):
        rs = np.random.default_rng(seed * 31 + 11)
        # Sample 7 points on the perimeter of a unit square
        pts = []
        for _ in range(7):
            side = rs.integers(0, 4)
            t = rs.uniform(0, 1)
            if side == 0: pts.append([t, 0])
            elif side == 1: pts.append([1, t])
            elif side == 2: pts.append([1-t, 1])
            else: pts.append([0, 1-t])
        pts = np.array(pts)
        # Center
        pts -= pts.mean(axis=0)
        # Scale to unit norm
        norm = np.max(np.linalg.norm(pts, axis=1))
        if norm > 0: pts = pts / norm
        inits.append(pts.flatten().copy())
        
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    def update_best(pts, score):
        nonlocal best_score, best_params, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            # We don't strictly need to store params if we only return points, 
            # but for debugging or further refinement, it's good.
            
    # Phase 1: Multi-start Nelder-Mead on B-spline control points
    inits = _generate_initial_configs()
    
    # Evaluate initial scores
    for init in inits:
        if time.time() > deadline:
            break
        pts = _decode_bspline(init)
        score = _score(pts)
        update_best(pts, score)
        
    # Run NM on each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                _objective_bspline, 
                init, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            if res.fun < 0:
                score = -res.fun
                pts = _decode_bspline(res.x)
                # Verify score
                actual_score = _score(pts)
                if actual_score > best_score:
                    best_score = actual_score
                    best_points = pts.copy()
                    best_params = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Differential Evolution for global exploration
    if time.time() < deadline - 30:
        # Bounds for control points: [-2, 2] is usually sufficient
        bounds = [(-2.0, 2.0)] * 14
        
        try:
            result = differential_evolution(
                _objective_bspline, 
                bounds, 
                maxiter=30, 
                popsize=10, 
                tol=1e-4, 
                seed=42,
                polish=False,
                update_interval=1
            )
            
            if result.fun < 0:
                score = -result.fun
                pts = _decode_bspline(result.x)
                actual_score = _score(pts)
                if actual_score > best_score:
                    best_score = actual_score
                    best_points = pts.copy()
                    best_params = result.x.copy()
                    
                    # Polish the DE result
                    if time.time() < deadline - 10:
                        res_polish = minimize(
                            _objective_bspline, 
                            result.x, 
                            method='Nelder-Mead', 
                            options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5}
                        )
                        if res_polish.fun < 0:
                            score2 = -res_polish.fun
                            pts2 = _decode_bspline(res_polish.x)
                            actual_score2 = _score(pts2)
                            if actual_score2 > best_score:
                                best_score = actual_score2
                                best_points = pts2.copy()
                                best_params = res_polish.x.copy()
        except Exception:
            pass

    # Phase 3: Cartesian Refinement of the Best Points
    # If we have a good set of points, try to optimize their Cartesian coordinates directly.
    # This allows escaping the B-spline manifold.
    if best_points is not None and time.time() < deadline - 5:
        x0 = best_points.flatten().copy()
        
        # Objective for Cartesian points
        def _objective_cart(flat):
            pts = flat.reshape(N, 2)
            # Ensure points are not degenerate
            if np.any(~np.isfinite(pts)):
                return 1e10
            score = _score(pts)
            return -score
        
        try:
            # Use Powell or Nelder-Mead
            # Nelder-Mead is robust for non-smooth functions
            res_cart = minimize(
                _objective_cart, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            if res_cart.fun < 0:
                score_cart = -res_cart.fun
                pts_cart = res_cart.x.reshape(N, 2)
                # Check if it's still a valid convex configuration (it should be, but hull area might change)
                actual_score_cart = _score(pts_cart)
                if actual_score_cart > best_score:
                    best_score = actual_score_cart
                    best_points = pts_cart.copy()
        except Exception:
            pass

    # Fallback: Regular 14-gon
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points

    return best_points

# EVOLVE-BLOCK-END
