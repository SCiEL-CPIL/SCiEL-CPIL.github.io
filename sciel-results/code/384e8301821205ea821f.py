"""Optimized packing of 11 unit regular hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11

# Precompute unit hexagon vertices (flat-top orientation, side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
_UNIT_HEX_VERTS = np.array([
    [np.cos(np.radians(a)), np.sin(np.radians(a))]
    for a in [0, 60, 120, 180, 240, 300]
])

def _get_hex_vertices(cx, cy, angle_deg):
    """Get vertices of a unit hexagon centered at (cx, cy) rotated by angle_deg."""
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return cx + R @ _UNIT_HEX_VERTS.T

def _point_in_hexagon(px, py, cx, cy, angle_deg):
    """Check if point (px, py) is inside or on boundary of hexagon."""
    # Transform point to hexagon's local frame
    theta = np.radians(-angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    local = R @ np.array([px - cx, py - cy])
    
    # In local frame, unit hexagon (flat-top) has vertices at angles 0,60,...
    # A point is inside if it satisfies the 6 half-plane constraints
    # For flat-top hexagon with vertices at (1,0), (0.5, sqrt(3)/2), etc.
    # The half-planes are:
    # x <= 1, -x <= 1, x/2 + sqrt(3)*y/2 <= 1, -x/2 - sqrt(3)*y/2 <= 1,
    # x/2 - sqrt(3)*y/2 <= 1, -x/2 + sqrt(3)*y/2 <= 1
    x, y = local
    s3 = np.sqrt(3)
    return (x <= 1 + 1e-9 and -x <= 1 + 1e-9 and
            x/2 + s3*y/2 <= 1 + 1e-9 and -x/2 - s3*y/2 <= 1 + 1e-9 and
            x/2 - s3*y/2 <= 1 + 1e-9 and -x/2 + s3*y/2 <= 1 + 1e-9)

def _hexagon_overlap_area(cx1, cy1, a1, cx2, cy2, a2):
    """Compute approximate overlap area between two unit hexagons using sampling."""
    # Quick rejection: if centers are far apart, no overlap
    dx = cx1 - cx2
    dy = cy1 - cy2
    if dx*dx + dy*dy > 4.0:  # max distance for overlap is 2 (sum of circumradii)
        return 0.0
    
    # Use a grid of sample points in the intersection region
    # Sample points in a circle around the midpoint
    mx = (cx1 + cx2) / 2
    my = (cy1 + cy2) / 2
    r = 1.5  # radius of sampling region
    
    # Create grid
    n = 20
    xs = np.linspace(mx - r, mx + r, n)
    ys = np.linspace(my - r, my + r, n)
    X, Y = np.meshgrid(xs, ys)
    
    count = 0
    total = 0
    for i in range(n):
        for j in range(n):
            px, py = X[i, j], Y[i, j]
            if _point_in_hexagon(px, py, cx1, cy1, a1) and _point_in_hexagon(px, py, cx2, cy2, a2):
                count += 1
            total += 1
    
    area = (2*r) ** 2
    return count / total * area

def _check_no_overlap(inner, tol=0.01):
    """Check that no two inner hexagons overlap. Returns True if valid."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            # Quick distance check
            dx = x1 - x2
            dy = y1 - y2
            if dx*dx + dy*dy > 4.0:
                continue
            overlap = _hexagon_overlap_area(x1, y1, a1, x2, y2, a2)
            if overlap > tol:
                return False
    return True

def _outer_hex_contains(cx, cy, angle_deg, outer_cx, outer_cy, outer_side, outer_angle_deg):
    """Check if a unit hexagon at (cx,cy) with rotation angle_deg is inside the outer hexagon."""
    # Get vertices of the inner hexagon
    verts = _get_hex_vertices(cx, cy, angle_deg)
    
    # Check if all vertices are inside the outer hexagon
    for vx, vy in verts:
        if not _point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_angle_deg):
            return False
    return True

def _compute_min_outer_side(inner, outer_cx, outer_cy, outer_angle_deg):
    """Compute the minimum side length of outer hexagon needed to contain all inner hexagons."""
    # The outer hexagon is centered at (outer_cx, outer_cy) with rotation outer_angle_deg
    # We need to find the minimum side length s such that all inner hexagons fit.
    
    # For a regular hexagon with side length s, the distance from center to a side (apothem) is s * sqrt(3)/2
    # The distance from center to a vertex is s
    
    # For each inner hexagon vertex, compute the required side length
    # A point (px, py) is inside a hexagon centered at (cx, cy) with rotation angle a
    # if in the local frame, the point satisfies the 6 half-plane constraints.
    
    # The maximum "radius" in each of the 6 directions determines the required side length.
    
    theta = np.radians(outer_angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    max_required = 0.0
    for ix, iy, ia in inner:
        verts = _get_hex_vertices(ix, iy, ia)
        for vx, vy in verts:
            # Transform to outer hexagon's local frame
            local = R @ np.array([vx - outer_cx, vy - outer_cy])
            x, y = local
            s3 = np.sqrt(3)
            # The 6 constraints give the required side length:
            # x <= s, -x <= s, x/2 + s3*y/2 <= s, etc.
            # So required s >= x, s >= -x, s >= x/2 + s3*y/2, etc.
            req1 = x
            req2 = -x
            req3 = x/2 + s3*y/2
            req4 = -x/2 - s3*y/2
            req5 = x/2 - s3*y/2
            req6 = -x/2 + s3*y/2
            req = max(req1, req2, req3, req4, req5, req6)
            if req > max_required:
                max_required = req
    
    return max(0.0, max_required)

def _objective(params, outer_cx, outer_cy, outer_angle_deg):
    """Objective function: return the outer side length, or a large penalty if invalid."""
    # params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11]
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append([x, y, a])
    
    # Check for overlaps
    if not _check_no_overlap(inner, tol=0.02):
        return 1e6  # Large penalty
    
    # Compute minimum outer side length
    side = _compute_min_outer_side(inner, outer_cx, outer_cy, outer_angle_deg)
    return side

def construct():
    """Construct the packing. Returns (inner, center, side, angle)."""
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Initial guess: arrange hexagons in a honeycomb pattern
    # Unit hexagons in honeycomb: spacing is sqrt(3) in one direction, 1.5 in the other
    # For flat-top hexagons, neighbors are at distance sqrt(3) horizontally and 1.5 vertically
    
    # Start with a simple linear arrangement and optimize
    initial_inner = []
    for i in range(N):
        x = 2.0 * i
        y = 0.0
        a = 0.0
        initial_inner.append([x, y, a])
    
    # Convert to flat params
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    
    # Outer hexagon parameters (we'll optimize these too, but start with reasonable values)
    outer_cx = 0.0
    outer_cy = 0.0
    outer_angle_deg = 0.0
    
    # Define bounds for differential evolution
    # Positions: roughly -5 to 5, angles: 0 to 360
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((0.0, 360.0)) # angle
    
    # Run differential evolution with time limit
    best_result = None
    best_side = 1e6
    
    # First, do a quick local optimization from the initial guess
    try:
        res = minimize(_objective, initial_params, args=(outer_cx, outer_cy, outer_angle_deg),
                       method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_side:
            best_side = res.fun
            best_result = res.x
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
    except:
        pass
    
    # Now try differential evolution
    try:
        # Use a smaller population and fewer iterations to stay within time
        def objective_wrapper(params):
            return _objective(params, outer_cx, outer_cy, outer_angle_deg)
        
        res_de = differential_evolution(objective_wrapper, bounds, maxiter=50, popsize=15,
                                         tol=1e-4, seed=42, polish=False,
                                         init='sobol')
        if res_de.fun < best_side:
            best_side = res_de.fun
            best_result = res_de.x
    except:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        # Not enough time for more optimization
        pass
    else:
        # Do a final local optimization from the best result
        if best_result is not None:
            try:
                res_final = minimize(_objective, best_result, args=(outer_cx, outer_cy, outer_angle_deg),
                                     method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
                if res_final.fun < best_side:
                    best_side = res_final.fun
                    best_result = res_final.x
            except:
                pass
    
    # Construct the final inner list
    if best_result is None:
        inner = initial_inner
    else:
        inner = []
        for i in range(N):
            x = best_result[3*i]
            y = best_result[3*i+1]
            a = best_result[3*i+2]
            inner.append([x, y, a])
    
    # Compute the actual outer side length
    side = _compute_min_outer_side(inner, outer_cx, outer_cy, outer_angle_deg)
    
    # Return the result
    return inner, (outer_cx, outer_cy), side, outer_angle_deg

# EVOLVE-BLOCK-END
