"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.deg2rad(angle_deg)
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300
    angles = a + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _point_in_hexagon(point, center, side, angle_deg):
    """Check if a point is inside a regular hexagon."""
    cx, cy = center
    px, py = point
    # Translate to center
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    a = -np.deg2rad(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    # For a regular hexagon with side length s, the inscribed circle radius is s*sqrt(3)/2
    # and circumscribed circle radius is s
    # The hexagon (flat-top) in local coords has constraints:
    # |x| <= s
    # |x*sqrt(3)/2 + y/2| <= s  (for each of 6 sides)
    # Actually, for a regular hexagon with "radius" (center to vertex) = s:
    # The 6 side constraints are:
    # x*cos(k*60) + y*sin(k*60) <= s for k=0..5
    # Let's use that
    for k in range(6):
        ka = np.deg2rad(k * 60)
        val = rx * np.cos(ka) + ry * np.sin(ka)
        if val > side + 1e-9:
            return False
    return True

def _hex_hex_overlap(p1, s1, a1, p2, s2, a2):
    """Check if two hexagons overlap. Return True if they overlap (interior intersection)."""
    # For two convex polygons, they overlap if:
    # 1. One vertex of either is inside the other, OR
    # 2. An edge of one intersects an edge of the other
    v1 = _hex_vertices(p1, s1, a1)
    v2 = _hex_vertices(p2, s2, a2)
    
    # Check if any vertex of v1 is inside v2
    for i in range(6):
        if _point_in_hexagon(v1[i], p2, s2, a2):
            return True
    # Check if any vertex of v2 is inside v1
    for i in range(6):
        if _point_in_hexagon(v2[i], p1, s1, a1):
            return True
    
    # Check edge intersections
    for i in range(6):
        e1 = v1[i], v1[(i+1)%6]
        for j in range(6):
            e2 = v2[j], v2[(j+1)%6]
            if _segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                # Check if it's a proper intersection (not just touching at a point)
                # For our purposes, any intersection means overlap
                return True
    return False

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def orient(a, b, c):
        val = (b[1]-a[1])*(c[0]-a[0]) - (b[0]-a[0])*(c[1]-a[1])
        if val > 1e-10: return 1
        if val < -1e-10: return -1
        return 0
    
    def on_segment(a, b, c):
        # Check if point b lies on segment a-c
        if (min(a[0], c[0]) - 1e-10 <= b[0] <= max(a[0], c[0]) + 1e-10 and
            min(a[1], c[1]) - 1e-10 <= b[1] <= max(a[1], c[1]) + 1e-10):
            return True
        return False
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    
    if o1 == 0 and on_segment(p1, p3, p2): return True
    if o2 == 0 and on_segment(p1, p4, p2): return True
    if o3 == 0 and on_segment(p3, p1, p4): return True
    if o4 == 0 and on_segment(p3, p2, p4): return True
    
    return False

def _is_valid_configuration(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are inside the outer and non-overlapping."""
    oc = np.array(outer_center)
    
    # Check each inner hex is inside outer
    for i in range(len(inner)):
        cx, cy, ang = inner[i]
        ic = np.array([cx, cy])
        # Check all 6 vertices of inner hex are inside outer hex
        verts = _hex_vertices(ic, 1.0, ang)
        for v in verts:
            if not _point_in_hexagon(v, outer_center, outer_side, outer_angle):
                return False
    
    # Check pairwise non-overlap
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            if _hex_hex_overlap(inner[i][:2], 1.0, inner[i][2], 
                                inner[j][:2], 1.0, inner[j][2]):
                return False
    
    return True

def _compute_min_outer_side(inner, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Compute the minimum side length of an outer hexagon (fixed center and angle) 
    that contains all inner hexagons."""
    oc = np.array(outer_center)
    
    # For each inner hex, compute the maximum distance from outer center in each of the 6 normal directions
    # The outer hexagon constraint: for each direction k (0..5), the projection of all points 
    # onto that direction must be <= outer_side
    # Actually, the hexagon is defined by: x*cos(k*60) + y*sin(k*60) <= s for k=0..5
    # And also the opposite: -x*cos(k*60) - y*sin(k*60) <= s, which is the same as k+3
    
    # So we need: for all 6 directions, max over all vertices of (v - oc) dot dir_k <= s
    # and (oc - v) dot dir_k <= s, i.e., |dot| <= s
    
    max_proj = 0.0
    for i in range(len(inner)):
        cx, cy, ang = inner[i]
        ic = np.array([cx, cy])
        verts = _hex_vertices(ic, 1.0, ang)
        for v in verts:
            d = v - oc
            for k in range(6):
                ka = np.deg2rad(k * 60)
                proj = abs(d[0]*np.cos(ka) + d[1]*np.sin(ka))
                if proj > max_proj:
                    max_proj = proj
    
    return max_proj

def _objective(params, outer_center=None, outer_angle=0.0):
    """Objective: minimize outer side length given inner hex positions."""
    n = N
    # params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    inner = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append([x, y, a])
    
    if outer_center is None:
        outer_center = (0.0, 0.0)
    
    side = _compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Add penalty for overlaps to guide the optimizer
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            # Simple distance-based penalty: if centers are too close, they likely overlap
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Two unit hexagons overlap if distance < ~2 (roughly)
            # More precisely, the minimum distance between non-overlapping hexagons depends on orientation
            # Use a soft penalty
            if dist < 2.2:
                penalty += (2.2 - dist) ** 2
    
    return side + 0.1 * penalty

def _gradient_penalty(params, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Compute gradient of the penalty term for overlapping hexagons."""
    n = N
    inner = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append([x, y, a])
    
    grad = np.zeros_like(params)
    for i in range(n):
        for j in range(i+1, n):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 2.2 and dist > 1e-6:
                # d/dx_i of (2.2 - dist)^2 = -2*(2.2-dist)*dx/dist
                factor = -2.0 * (2.2 - dist) * 0.1 / dist
                grad[3*i] += factor * dx
                grad[3*i+1] += factor * dy
                grad[3*j] -= factor * dx
                grad[3*j+1] -= factor * dy
    
    return grad

def _initial_layout():
    """Create an initial layout with hexagons in a honeycomb-like arrangement."""
    inner = []
    # Place in a roughly circular/hexagonal pattern
    # 12 hexagons: maybe 2 rings? Center + 6 + 5? Or 3 rows of 4?
    # Let's try a compact arrangement
    # Row-based: 4 rows
    # Actually, let's use a hexagonal lattice pattern
    # For 12 hexagons, we can do:
    #   .  .  .  .
    #  .  .  .  .  .
    #   .  .  .  .
    # But that's 13. Let's do:
    #  .  .  .  .
    #   .  .  .  .
    #  .  .  .  .
    # That's 12.
    
    # Hex lattice spacing: for unit hexagons, center-to-center distance for adjacent
    # in a honeycomb is sqrt(3) for flat-top orientation
    dx = np.sqrt(3)  # horizontal spacing
    dy = 1.5  # vertical spacing between rows (for pointy-top)
    # Actually for flat-top hexagons, the vertical distance between rows is 1.5 * side * 2/sqrt(3)?
    # Let's just use approximate values and let the optimizer fix it
    
    positions = []
    # 4 rows: 3, 4, 4, 1? No, let's do 3,4,3,2 or similar
    # Simpler: just place them in a grid-like pattern
    for i in range(12):
        row = i // 4
        col = i % 4
        x = col * dx + (row % 2) * dx/2
        y = row * 1.5
        positions.append([x, y, 0.0])
    
    # Center them
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    return positions

def _optimize_single(outer_center=(0.0, 0.0), outer_angle=0.0, seed=0):
    """Run optimization for a single configuration."""
    np.random.seed(seed)
    
    inner = _initial_layout()
    
    # Add some randomness
    for i in range(N):
        inner[i][0] += np.random.uniform(-0.5, 0.5)
        inner[i][1] += np.random.uniform(-0.5, 0.5)
        inner[i][2] += np.random.uniform(-15, 15)
    
    x0 = np.array([v for p in inner for v in p])
    
    best_side = _compute_min_outer_side(inner, outer_center, outer_angle)
    best_params = x0.copy()
    
    # Use Nelder-Mead (derivative-free, robust)
    try:
        result = minimize(
            _objective, x0,
            args=(outer_center, outer_angle),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
        )
        if result.fun < best_side:
            # Verify and get actual side
            inner_res = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
            side_res = _compute_min_outer_side(inner_res, outer_center, outer_angle)
            if side_res < best_side:
                best_side = side_res
                best_params = result.x.copy()
    except Exception:
        pass
    
    # Also try BFGS with finite differences
    try:
        result2 = minimize(
            _objective, x0,
            args=(outer_center, outer_angle),
            method='L-BFGS-B',
            options={'maxiter': 500}
        )
        if result2.fun < best_side:
            inner_res = [[result2.x[3*i], result2.x[3*i+1], result2.x[3*i+2]] for i in range(N)]
            side_res = _compute_min_outer_side(inner_res, outer_center, outer_angle)
            if side_res < best_side:
                best_side = side_res
                best_params = result2.x.copy()
    except Exception:
        pass
    
    return best_side, best_params

def _refine_layout(inner, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Refine a layout using local optimization."""
    x0 = np.array([v for p in inner for v in p])
    best_side = _compute_min_outer_side(inner, outer_center, outer_angle)
    best_params = x0.copy()
    
    try:
        result = minimize(
            _objective, x0,
            args=(outer_center, outer_angle),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
        )
        inner_res = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
        side_res = _compute_min_outer_side(inner_res, outer_center, outer_angle)
        if side_res < best_side:
            best_side = side_res
            best_params = result.x.copy()
    except Exception:
        pass
    
    return best_side, best_params

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    best_side = float('inf')
    best_inner = None
    
    # Try different outer angles
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            
            side, params = _optimize_single(
                outer_center=(0.0, 0.0),
                outer_angle=outer_angle,
                seed=seed
            )
            
            if side < best_side:
                best_side = side
                best_inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
        
        if time.time() > deadline - 10:
            break
    
    # If no valid result, use a fallback
    if best_inner is None:
        inner = _initial_layout()
        best_side = _compute_min_outer_side(inner, (0.0, 0.0), 0.0)
        best_inner = inner
    
    # Final refinement
    if time.time() < deadline - 5:
        side, params = _refine_layout(best_inner, (0.0, 0.0), 0.0)
        if side < best_side:
            best_side = side
            best_inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
    
    return best_inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
