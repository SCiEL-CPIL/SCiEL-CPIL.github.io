"""
Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.
Structural change: use smooth SDF-based objective with L-BFGS-B optimization,
shared rotation, and final SAT verification with local polish.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - SAFETY_MARGIN - _elapsed()

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _project_on_axis(verts, axis):
    """Project vertices onto axis, return (min, max)."""
    dots = [v[0]*axis[0] + v[1]*axis[1] for v in verts]
    return min(dots), max(dots)

def _sat_overlap(verts1, verts2):
    """
    Separating Axis Theorem: check if two convex polygons overlap.
    Returns True if they overlap (i.e., no separating axis found).
    """
    edges = []
    for verts in (verts1, verts2):
        for i in range(len(verts)):
            j = (i + 1) % len(verts)
            dx = verts[j][0] - verts[i][0]
            dy = verts[j][1] - verts[i][1]
            nx, ny = -dy, dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            edges.append((nx/norm, ny/norm))
    
    for axis in edges:
        min1, max1 = _project_on_axis(verts1, axis)
        min2, max2 = _project_on_axis(verts2, axis)
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
    return True

def _hexagon_overlap(verts1, verts2):
    """Check if two hexagons overlap using SAT."""
    return _sat_overlap(verts1, verts2)

def _outer_hex_side(inner_hexes, outer_center, outer_angle):
    """
    Compute the minimum outer side length such that all inner hexagons fit inside.
    inner_hexes: list of (cx, cy, angle_deg)
    outer_center: (cx, cy)
    outer_angle: rotation of outer hexagon in degrees
    """
    cx, cy = outer_center
    angle_rad = math.radians(outer_angle)
    
    max_dist = 0.0
    for (ix, iy, iang) in inner_hexes:
        verts = _hex_vertices(ix, iy, 1.0, iang)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0 + math.pi / 2.0
            nx, ny = math.cos(theta), math.sin(theta)
            for (px, py) in verts:
                d = (px - cx) * nx + (py - cy) * ny
                if d > max_dist:
                    max_dist = d
    
    return max_dist / (math.sqrt(3) / 2.0)

def _sdf_point_to_hex(px, py, cx, cy, angle_deg):
    """
    Signed distance from point (px,py) to regular hexagon centered at (cx,cy) with side 1.
    Positive if outside, negative if inside.
    """
    angle_rad = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate into hexagon's local frame
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # In local frame, hexagon has vertices at angles 0, 60, ..., 300 degrees
    # Apothem = sqrt(3)/2
    apothem = math.sqrt(3) / 2.0
    
    # The hexagon in local coords: |x| <= 1 and |x/2 + sqrt(3)*y/2| <= sqrt(3)/2 etc.
    # Actually for a flat-top hexagon (vertex at angle 0):
    # The 6 half-planes are defined by normals at 30, 90, 150, 210, 270, 330 degrees
    # For pointy-top (vertex at 90): normals at 0, 60, 120, 180, 240, 300
    
    # Let's use the standard: hexagon with vertices at angles 0,60,...,300 from center
    # The 6 outward normals are at 30, 90, 150, 210, 270, 330 degrees
    # A point is inside if for all 6 normals: dot(point, normal) <= apothem
    
    # For a hexagon with vertex at angle 0 (pointing right), the normals are at 30+60k degrees
    # Let me reconsider: if vertices are at angle a + 60k, then edges are between consecutive vertices
    # Edge between vertex at angle a and a+60 has normal at a+30
    
    # For our hexagon with angle_deg rotation, vertices at angle_deg + 60k
    # Normals at angle_deg + 30 + 60k
    
    # Compute the maximum of (dot(p_local, normal_k) - apothem) over all 6 normals
    # If max < 0, point is inside. The SDF is approximately the max of these values
    # (exact SDF requires more care but this is a good approximation for smooth optimization)
    
    max_val = 0.0
    for k in range(6):
        theta = angle_rad + math.pi/6.0 + k * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        val = lx * nx + ly * ny - apothem
        if val > max_val:
            max_val = val
    
    # This gives a convex function that is 0 on the boundary and positive outside
    # For the smooth penalty, we use max(0, val)
    return max_val

def _sdf_hex_hex(cx1, cy1, a1, cx2, cy2, a2):
    """
    Approximate SDF between two unit hexagons.
    Uses the fact that if we consider the Minkowski difference, two hexagons don't overlap
    if the center-to-center distance in the appropriate metric exceeds a threshold.
    
    For a simpler smooth approach: compute the minimum of _sdf_point_to_hex over
    sample points on one hexagon relative to the other. This is expensive but smooth-ish.
    
    Alternative: use the distance between centers projected onto the relative orientation.
    For two hexagons with the same orientation, they don't overlap iff the distance
    between centers, measured in the hexagonal norm, exceeds 2*apothem in the appropriate direction.
    
    For simplicity and smoothness, we'll use a sum of SDFs sampled at vertices of one hexagon
    relative to the other.
    """
    # Sample vertices of hex1, compute SDF to hex2 for each
    verts1 = _hex_vertices(cx1, cy1, 1.0, a1)
    total = 0.0
    for (vx, vy) in verts1:
        d = _sdf_point_to_hex(vx, vy, cx2, cy2, a2)
        total += max(0.0, d)
    # Also sample vertices of hex2 relative to hex1
    verts2 = _hex_vertices(cx2, cy2, 1.0, a2)
    for (vx, vy) in verts2:
        d = _sdf_point_to_hex(vx, vy, cx1, cy1, a1)
        total += max(0.0, d)
    return total

def _smooth_objective(params, outer_center, outer_angle, shared_angle):
    """
    Smooth objective: outer side length + penalty for overlaps.
    params: flat array of length 2*N: [x0, y0, x1, y1, ...]
    All hexagons share the same rotation angle (shared_angle).
    """
    inner = []
    for i in range(N):
        x = params[2*i]
        y = params[2*i+1]
        inner.append((x, y, shared_angle))
    
    # Compute outer side needed
    side = _outer_hex_side(inner, outer_center, outer_angle)
    
    # Overlap penalty using SDF
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sdf_hex_hex(inner[i][0], inner[i][1], inner[i][2],
                            inner[j][0], inner[j][1], inner[j][2])
            penalty += d * d  # Quadratic penalty for smoothness
    
    return side + 10.0 * penalty

def _is_valid(inner, outer_center, outer_angle):
    """Check if configuration is valid: no overlaps, all inside outer hexagon."""
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if _hexagon_overlap(v1, v2):
                return False
    return True

def _generate_initial_configs():
    """Generate diverse initial configurations for the 12 inner hexagons."""
    configs = []
    
    # Config 1: Ring arrangement
    ring = []
    for i in range(12):
        angle = i * 30.0
        r = 3.0
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        ring.append((x, y))
    configs.append(ring)
    
    # Config 2: Two rows of 6
    row = []
    for i in range(6):
        x = (i - 2.5) * 2.1
        row.append((x, -1.2))
    for i in range(6):
        x = (i - 2.5) * 2.1
        row.append((x, 1.2))
    configs.append(row)
    
    # Config 3: Hexagonal close-packed
    hcp = []
    positions = [(0,0), (2.1,0), (-2.1,0), (1.05, 1.82), (-1.05, 1.82),
                 (1.05, -1.82), (-1.05, -1.82), (3.15, 0.91), (-3.15, 0.91),
                 (3.15, -0.91), (-3.15, -0.91), (0, 3.64)]
    for (x, y) in positions:
        hcp.append((x, y))
    configs.append(hcp)
    
    # Config 4: 3x4 grid
    grid = []
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * 2.1
            y = (r - 1.0) * 1.85
            if r % 2 == 1:
                x += 1.05
            grid.append((x, y))
    configs.append(grid)
    
    # Config 5: Random
    np.random.seed(42)
    rand = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        rand.append((x, y))
    configs.append(rand)
    
    # Config 6: Smaller ring
    ring2 = []
    for i in range(12):
        angle = i * 30.0
        r = 2.5
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        ring2.append((x, y))
    configs.append(ring2)
    
    return configs

def _local_polish(inner, outer_center, outer_angle, max_iters=50):
    """
    Local polish: use small perturbations to reduce the outer side length
    while maintaining validity.
    """
    best_side = _outer_hex_side(inner, outer_center, outer_angle)
    current = [list(h) for h in inner]
    
    step = 0.1
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        improved = False
        for i in range(N):
            for dim in range(2):  # x and y only
                for direction in [-1, 1]:
                    trial = [list(h) for h in current]
                    trial[i][dim] += direction * step
                    
                    # Check validity
                    if _is_valid(trial, outer_center, outer_angle):
                        trial_side = _outer_hex_side(trial, outer_center, outer_angle)
                        if trial_side < best_side - 1e-8:
                            current = trial
                            best_side = trial_side
                            improved = True
        
        if not improved:
            step *= 0.7
            if step < 0.01:
                break
    
    return current, best_side

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_inner = None
    best_side = float('inf')
    best_oa = 0.0
    
    outer_center = (0.0, 0.0)
    
    configs = _generate_initial_configs()
    
    # Try different outer angles and shared inner angles
    outer_angles = [0.0, 30.0]
    shared_angles = [0.0, 30.0]
    
    for oa in outer_angles:
        if _time_left() < 15.0:
            break
        
        for sa in shared_angles:
            if _time_left() < 15.0:
                break
            
            for config in configs:
                if _time_left() < 10.0:
                    break
                
                # Flatten config to params (positions only, angle is shared)
                params = np.array([v for item in config for v in item])
                
                def obj(p):
                    return _smooth_objective(p, outer_center, oa, sa)
                
                # Run L-BFGS-B
                try:
                    result = minimize(
                        obj, params, method='L-BFGS-B',
                        options={'maxiter': 100, 'maxfun': 500,
                                'ftol': 1e-10, 'gtol': 1e-8}
                    )
                except Exception:
                    continue
                
                if _time_left() < 5.0:
                    continue
                
                # Extract inner from result
                inner = []
                for i in range(N):
                    x = result.x[2*i]
                    y = result.x[2*i+1]
                    inner.append((x, y, sa))
                
                # Check validity
                if not _is_valid(inner, outer_center, oa):
                    continue
                
                side = _outer_hex_side(inner, outer_center, oa)
                
                if side < best_side:
                    best_side = side
                    best_inner = [list(h) for h in inner]
                    best_oa = oa
                
                # Local polish
                if _time_left() > 10.0:
                    polished, polished_side = _local_polish(inner, outer_center, oa, max_iters=30)
                    if polished_side < best_side:
                        best_side = polished_side
                        best_inner = polished
                        best_oa = oa
    
    # Final verification and best outer angle selection
    if best_inner is not None:
        # Try to find best outer angle for the final configuration
        final_best_side = float('inf')
        final_best_oa = best_oa
        for oa in [0.0, 30.0, 15.0, 45.0, 60.0]:
            side = _outer_hex_side(best_inner, (0.0, 0.0), oa)
            if side < final_best_side:
                final_best_side = side
                final_best_oa = oa
        
        # Verify no overlaps with final configuration
        has_overlap = False
        for i in range(N):
            for j in range(i+1, N):
                v1 = _hex_vertices(best_inner[i][0], best_inner[i][1], 1.0, best_inner[i][2])
                v2 = _hex_vertices(best_inner[j][0], best_inner[j][1], 1.0, best_inner[j][2])
                if _hexagon_overlap(v1, v2):
                    has_overlap = True
        
        if not has_overlap:
            return best_inner, (0.0, 0.0), final_best_side, final_best_oa
    
    # Fallback: safe default
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
