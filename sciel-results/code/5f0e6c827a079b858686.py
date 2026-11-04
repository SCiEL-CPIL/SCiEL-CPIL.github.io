"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

DEADLINE = 140.0

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    # Regular hexagon vertices: side length s, centered at (cx,cy), rotated by a
    angles = a + np.pi/6 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hex_contains_point(v, px, py):
    """Check if point (px,py) is inside or on hexagon with vertices v (CCW)."""
    for i in range(6):
        ax, ay = v[i]
        bx, by = v[(i+1) % 6]
        # cross product to check if point is to the left of edge (ax,ay)->(bx,by)
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross < -1e-12:
            return False
    return True

def _hex_distance(v1, v2):
    """Minimum distance between two convex hexagons. 0 if they overlap."""
    # First check if any vertex of one is inside the other
    for i in range(6):
        if _hex_contains_point(v2, v1[i][0], v1[i][1]):
            return 0.0
    for i in range(6):
        if _hex_contains_point(v1, v2[i][0], v2[i][1]):
            return 0.0
    # Otherwise, compute min edge-to-edge distance
    min_dist = float('inf')
    for i in range(6):
        a1 = v1[i]; b1 = v1[(i+1) % 6]
        for j in range(6):
            a2 = v2[j]; b2 = v2[(j+1) % 6]
            d = _seg_seg_dist(a1, b1, a2, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _seg_seg_dist(a1, b1, a2, b2):
    """Minimum distance between two line segments."""
    # If segments intersect, distance is 0
    if _segs_intersect(a1, b1, a2, b2):
        return 0.0
    # Check endpoints
    d = min(_pt_seg_dist(a1, a2, b2), _pt_seg_dist(b1, a2, b2),
            _pt_seg_dist(a2, a1, b1), _pt_seg_dist(b2, a1, b1))
    return d

def _pt_seg_dist(p, a, b):
    """Distance from point p to segment ab."""
    ax, ay = a; bx, by = b; px, py = p
    dx = bx - ax; dy = by - ay
    if dx == 0 and dy == 0:
        return np.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    cx = ax + t * dx; cy = ay + t * dy
    return np.hypot(px - cx, py - cy)

def _segs_intersect(a1, b1, a2, b2):
    """Check if two segments intersect."""
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    o1 = orient(a1, b1, a2)
    o2 = orient(a1, b1, b2)
    o3 = orient(a2, b2, a1)
    o4 = orient(a2, b2, b1)
    if o1 == 0 and _on_segment(a1, b1, a2): return True
    if o2 == 0 and _on_segment(a1, b1, b2): return True
    if o3 == 0 and _on_segment(a2, b2, a1): return True
    if o4 == 0 and _on_segment(a2, b2, b1): return True
    return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))

def _on_segment(a, b, c):
    """Check if c is on segment ab."""
    if c[0] < min(a[0], b[0]) or c[0] > max(a[0], b[0]) or c[1] < min(a[1], b[1]) or c[1] > max(a[1], b[1]):
        return False
    return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) < 1e-10

def _outer_hex_contains_inner(cx, cy, side, inner_vertices):
    """Check if all vertices of inner hexagon are inside or on outer hexagon."""
    for i in range(6):
        px, py = inner_vertices[i]
        if not _hex_contains_point(inner_vertices * 0 + np.array([[0,0]]*6), px, py):  # placeholder
            pass
    # Actually check against outer
    return True  # Will be replaced

def _is_valid(inner, outer_center, outer_side, outer_angle):
    """Check all constraints: non-overlap and containment."""
    ocx, ocy = outer_center
    outer_v = _hex_vertices(ocx, ocy, outer_side, outer_angle)
    inner_vs = []
    for x, y, a in inner:
        v = _hex_vertices(x, y, 1.0, a)
        inner_vs.append(v)
        # Containment check
        for j in range(6):
            if not _hex_contains_point(outer_v, v[j][0], v[j][1]):
                return False
    # Non-overlap check
    for i in range(N):
        for j in range(i+1, N):
            d = _hex_distance(inner_vs[i], inner_vs[j])
            if d < 1e-9:
                return False
    return True

def _compute_objective(params, outer_center, outer_side, outer_angle):
    """Compute penalty: sum of violations."""
    ocx, ocy = outer_center
    outer_v = _hex_vertices(ocx, ocy, outer_side, outer_angle)
    # Parse params: [x0,y0,a0, x1,y1,a1, ...]
    inner_vs = []
    for i in range(N):
        x = params[3*i]; y = params[3*i+1]; a = params[3*i+2]
        v = _hex_vertices(x, y, 1.0, a)
        inner_vs.append(v)
    penalty = 0.0
    # Containment penalty
    for i in range(N):
        for j in range(6):
            px, py = inner_vs[i][j]
            if not _hex_contains_point(outer_v, px, py):
                # Compute distance from point to hexagon
                min_d = float('inf')
                for k in range(6):
                    d = _pt_seg_dist((px,py), outer_v[k], outer_v[(k+1)%6])
                    min_d = min(min_d, d)
                penalty += min_d * min_d
    # Non-overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            d = _hex_distance(inner_vs[i], inner_vs[j])
            if d < 0.5:  # Only penalize if close
                penalty += (0.5 - d) * (0.5 - d)
    return penalty

def construct():
    start_time = time.time()
    # Initial guess: honeycomb-like arrangement
    # Try to place hexagons in a compact cluster
    best_params = None
    best_outer_side = float('inf')
    best_valid = False
    
    # Start with a reasonable initial configuration
    # Place hexagons in a hexagonal grid pattern
    def _initial_config():
        params = []
        # Center hexagon
        params.extend([0.0, 0.0, 0.0])
        # 6 surrounding
        for k in range(6):
            a = np.radians(60*k)
            x = 2.0 * np.cos(a)
            y = 2.0 * np.sin(a)
            params.extend([x, y, 0.0])
        # 5 more
        for k in range(5):
            a = np.radians(60*k + 30)
            x = 3.464 * np.cos(a)
            y = 3.464 * np.sin(a)
            params.extend([x, y, 0.0])
        return np.array(params)
    
    params = _initial_config()
    
    # First, find a valid configuration with a reasonable outer side
    # Try different outer side lengths
    for outer_side in [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]:
        if time.time() - start_time > DEADLINE:
            break
        for outer_angle in [0.0, 30.0]:
            if time.time() - start_time > DEADLINE:
                break
            # Try to optimize
            result = minimize(
                _compute_objective, params,
                args=((0.0, 0.0), outer_side, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if result.fun < 1e-8:
                # Check validity
                inner = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
                if _is_valid(inner, (0.0, 0.0), outer_side, outer_angle):
                    if outer_side < best_outer_side:
                        best_outer_side = outer_side
                        best_params = result.x.copy()
                        best_valid = True
                        break
        if best_valid:
            break
    
    if not best_valid:
        # Fallback: return the initial configuration with a large outer side
        inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    # Polish: refine with tighter tolerances
    if time.time() - start_time < DEADLINE - 5:
        result = minimize(
            _compute_objective, best_params,
            args=((0.0, 0.0), best_outer_side, 0.0),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        inner = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
        if _is_valid(inner, (0.0, 0.0), best_outer_side, 0.0):
            best_params = result.x.copy()
    
    inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
    return inner, (0.0, 0.0), best_outer_side, 0.0
# EVOLVE-BLOCK-END

