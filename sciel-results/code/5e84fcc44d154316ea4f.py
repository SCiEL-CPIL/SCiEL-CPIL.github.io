"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)

def _all_areas(points):
    """Compute all triangle areas. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |det(p2-p1, p3-p1)|
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * cross

def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples."""
    return np.min(_all_areas(points))

def _params_to_points_symmetric(params):
    """
    Map symmetric parameters to 11 points.
    Structure:
    - Point 0: on the symmetry axis (x=0.5), parameterized by y only.
    - Points 1-10: 5 pairs of mirrored points.
      params[1+2i], params[2+2i] = x, y for left point of pair i.
      Right point is (1-x, y).
    
    params: array of shape (11,)
    """
    pts = np.zeros((N, 2))
    # Axis point
    y0 = params[0]
    # Ensure y0 is valid for x=0.5
    # Max y at x=0.5 is H
    y0 = np.clip(y0, 0.0, H)
    pts[0] = [0.5, y0]
    
    for i in range(5):
        x = params[1 + 2*i]
        y = params[2 + 2*i]
        
        # Clip x to [0, 0.5] to maintain symmetry and inside triangle (left half)
        x = np.clip(x, 0.0, 0.5)
        
        # Clip y to be inside the triangle for the given x
        # Left boundary: y <= SQRT3 * x
        # Right boundary: y <= SQRT3 * (1-x) -> for x<=0.5, 1-x >= 0.5, so right boundary is usually not the constraint if x is small.
        # Bottom boundary: y >= 0
        # Top boundary: y <= H
        
        # The triangle is defined by:
        # y >= 0
        # y <= SQRT3 * x
        # y <= SQRT3 * (1 - x)
        
        max_y_left = SQRT3 * x
        max_y_right = SQRT3 * (1 - x)
        max_y = min(max_y_left, max_y_right, H)
        
        y = np.clip(y, 0.0, max_y)
        
        pts[1 + 2*i] = [x, y]
        pts[2 + 2*i] = [1.0 - x, y]
        
    return pts

def _points_to_params_symmetric(points):
    """Extract symmetric parameters from a full point set for initialization."""
    pts = points.copy()
    params = np.zeros(11)
    
    # Find the point closest to the axis x=0.5 to be the axis point
    dists = np.abs(pts[:, 0] - 0.5)
    axis_idx = np.argmin(dists)
    params[0] = pts[axis_idx, 1]
    
    # For the remaining 10 points, sort by x to pair them
    remaining_indices = [i for i in range(N) if i != axis_idx]
    remaining_points = pts[remaining_indices]
    
    # Sort remaining points by x coordinate
    sort_idx = np.argsort(remaining_points[:, 0])
    sorted_remaining = remaining_points[sort_idx]
    
    # Pair them: (0, 1), (2, 3), etc.
    # Ideally, they should be symmetric. We take the average or just the left one.
    # Since we are just initializing, we take the left point of each pair.
    for i in range(5):
        left_pt = sorted_remaining[2*i]
        right_pt = sorted_remaining[2*i + 1]
        
        # The left point should have x <= 0.5
        if left_pt[0] > 0.5:
             # Swap if necessary, though sorting by x should handle this mostly
             left_pt, right_pt = right_pt, left_pt
             
        x_val = left_pt[0]
        y_val = (left_pt[1] + right_pt[1]) / 2.0 # Average y for robustness
        
        params[1 + 2*i] = x_val
        params[2 + 2*i] = y_val
        
    return params

def _inside_penalty(points):
    """Calculate penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    # Distances to boundaries (negative means outside)
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 * (1 - x) - y
    
    # Sum of squared violations
    violations = np.maximum(0, -d1)**2 + np.maximum(0, -d2)**2 + np.maximum(0, -d3)**2
    return np.sum(violations)

def _objective_symmetric(params):
    """
    Objective in symmetric parameter space: -min_area + penalty.
    """
    pts = _params_to_points_symmetric(params)
    pen = _inside_penalty(pts)
    if pen > 1e-4:
        return 1e6 + pen
    area = _min_area(pts)
    return -area + 10.0 * pen

def _soft_min_area(points, beta=50.0):
    """Smooth approximation of min area using log-sum-exp."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    min_val = np.min(areas)
    return min_val - np.log(np.sum(np.exp(-beta * (areas - min_val)))) / beta

def _objective_soft_symmetric(params, beta=50.0):
    """
    Objective in symmetric parameter space using soft min: -soft_min_area + penalty.
    """
    pts = _params_to_points_symmetric(params)
    pen = _inside_penalty(pts)
    if pen > 1e-4:
        return 1e6 + pen
    area = _soft_min_area(pts, beta=beta)
    return -area + 10.0 * pen

def _init_random_symmetric(rng):
    """Generate random symmetric parameters."""
    params = np.zeros(11)
    # Axis point
    params[0] = rng.random() * H
    
    for i in range(5):
        # Choose x in [0, 0.5]
        x = rng.random() * 0.5
        # Choose y valid for this x
        max_y = min(SQRT3 * x, SQRT3 * (1 - x), H)
        y = rng.random() * max_y
        params[1 + 2*i] = x
        params[2 + 2*i] = y
    return params

def _init_grid_symmetric():
    """Generate a grid of initial symmetric parameters for diversity."""
    params = np.zeros(11)
    params[0] = H / 2.0
    heights = np.linspace(0.1, 0.9, 5) * H
    for i, h in enumerate(heights):
        x_min = h / SQRT3
        # Place points roughly evenly spaced
        x = 0.1 + i * 0.1
        if x < x_min:
            x = x_min
        if x > 0.5:
            x = 0.5
        params[1 + 2*i] = x
        params[2 + 2*i] = h
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin for 150s limit
    
    # Bounds for symmetric parameters
    # params[0]: y of axis point [0, H]
    # params[1+2i]: x of left point [0, 0.5]
    # params[2+2i]: y of left point [0, H]
    bounds_sym = [(0, H)]
    for i in range(5):
        bounds_sym.append((0, 0.5))
        bounds_sym.append((0, H))
    
    best_params_sym = None
    best_score = -np.inf
    
    def update_best_sym(params_sym):
        nonlocal best_params_sym, best_score
        if params_sym is not None:
            pts = _params_to_points_symmetric(params_sym)
            # Re-check validity strictly
            if _inside_penalty(pts) > 1e-6:
                return
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_params_sym = params_sym.copy()

    def update_best_full(points):
        nonlocal best_params_sym, best_score
        if points is not None:
            # Clamp to ensure validity
            pts = points.copy()
            for i in range(N):
                x, y = pts[i]
                x = np.clip(x, 0.0, 1.0)
                y = np.clip(y, 0.0, H)
                max_y_left = SQRT3 * x
                if y > max_y_left:
                    y = max_y_left
                max_y_right = SQRT3 * (1 - x)
                if y > max_y_right:
                    y = max_y_right
                pts[i] = [x, y]
            
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_params_sym = _points_to_params_symmetric(pts)

    rng = np.random.default_rng(42)
    
    # Phase 1: Global Optimization with Differential Evolution using Soft Min
    # Soft min provides smoother gradients for DE
    if time.time() < deadline - 40:
        try:
            res_de = differential_evolution(
                lambda p: _objective_soft_symmetric(p, beta=30.0),
                bounds_sym,
                maxiter=60,
                popsize=35,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            update_best_sym(res_de.x)
        except Exception:
            pass
    
    # Phase 2: Local Polish with Nelder-Mead from best DE result using Exact Min
    if best_params_sym is not None and time.time() < deadline - 20:
        try:
            res_nm = minimize(
                _objective_symmetric,
                best_params_sym,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            update_best_sym(res_nm.x)
        except Exception:
            pass

    # Phase 3: Multi-start Local Optimization
    # Try several random starts and grid starts to escape local minima
    n_starts = 15
    starts = []
    
    # Add grid start
    starts.append(_init_grid_symmetric())
    
    # Add random starts
    for _ in range(n_starts - 1):
        starts.append(_init_random_symmetric(rng))
        
    # If we have a best from DE, include it
    if best_params_sym is not None:
        starts.append(best_params_sym)
    
    for s in starts:
        if time.time() > deadline - 15:
            break
        try:
            # Quick local optimization
            res_r = minimize(
                _objective_symmetric,
                s,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            update_best_sym(res_r.x)
        except Exception:
            pass
            
    # Phase 4: Final Polish on the absolute best
    if best_params_sym is not None and time.time() < deadline - 5:
        try:
            # One more high-precision polish
            res_final = minimize(
                _objective_symmetric,
                best_params_sym,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            update_best_sym(res_final.x)
        except Exception:
            pass

    # Fallback if no valid solution found
    if best_params_sym is None:
        best_params_sym = _init_grid_symmetric()
    
    points = _params_to_points_symmetric(best_params_sym)
    return points
