import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 140.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Get 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top orientation at 0)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    angles = base_angles + angle_rad
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hexagon_support(cx, cy, side, angle_deg, direction):
    """Compute support function of hexagon in a given direction (for containment check)."""
    angle_rad = np.radians(angle_deg)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    angles = base_angles + angle_rad
    vertices = np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    # Support in direction (dx, dy)
    dx, dy = direction
    dots = vertices[:, 0] * dx + vertices[:, 1] * dy
    return np.max(dots)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point is inside a regular hexagon."""
    # Transform to hexagon's local frame
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    dx = px - cx
    dy = py - cy
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    
    # In local frame, hexagon has vertices at (side*cos(k*60), side*sin(k*60)) for k=0..5
    # The hexagon is the intersection of 6 half-planes
    # For flat-top hex with side s, the constraints are:
    # |x| <= s (for the two vertical edges) -- wait, let me think more carefully
    
    # Actually for a regular hexagon with "radius" (center to vertex) = side (since it's regular),
    # the vertices are at distance `side` from center.
    # The inradius (center to edge midpoint) = side * sqrt(3)/2
    
    # The 6 edges are at angles 30, 90, 150, 210, 270, 330 from center (for flat-top)
    # Each edge is at distance inradius = side * sqrt(3)/2 from center
    # The normal to each edge points outward
    
    inradius = side * np.sqrt(3) / 2
    # Edge normals at angles 30, 90, 150, 210, 270, 330
    edge_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    for ea in edge_angles:
        nx = np.cos(ea)
        ny = np.sin(ea)
        if lx * nx + ly * ny > inradius:
            return False
    return True

def _hexagon_overlap_penalty(v1, v2, v3, v4, smooth_eps=0.01):
    """
    Smooth penalty for overlap between two hexagons.
    v1, v2: centers and sides of hex1 and hex2 (4 params each: cx, cy, side, angle)
    Returns a smooth measure of overlap (0 if disjoint, positive if overlapping).
    """
    cx1, cy1, s1, a1 = v1
    cx2, cy2, s2, a2 = v2
    
    # Get vertices
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    
    # Check if any vertex of hex1 is inside hex2, and vice versa
    # Also check edge intersections for more robust overlap detection
    
    # Simple approach: check if centers are too close relative to sizes
    # For same-oriented hexagons, minimum center distance for no overlap is 2*side
    # For different orientations, it's more complex
    
    # Use a smooth approximation: the minimum distance between the two hexagons
    # If negative, they overlap
    
    # Compute all pairwise distances between vertices and edges - too expensive
    # Instead, use a simpler proxy: check if any vertex is inside the other hexagon
    
    penalty = 0.0
    for vx, vy in verts1:
        if _point_in_hexagon(vx, vy, cx2, cy2, s2, a2):
            # Compute how deep inside (distance to boundary)
            # Approximate by distance to nearest edge
            angle_rad = np.radians(a2)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            dx = vx - cx2
            dy = vy - cy2
            lx = cos_a * dx + sin_a * dy
            ly = -sin_a * dx + cos_a * dy
            inradius = s2 * np.sqrt(3) / 2
            edge_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
            max_proj = -np.inf
            for ea in edge_angles:
                nx = np.cos(ea)
                ny = np.sin(ea)
                proj = lx * nx + ly * ny
                max_proj = max(max_proj, proj)
            depth = max_proj - inradius
            penalty += max(0, depth) ** 2
    
    for vx, vy in verts2:
        if _point_in_hexagon(vx, vy, cx1, cy1, s1, a1):
            angle_rad = np.radians(a1)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            dx = vx - cx1
            dy = vy - cy1
            lx = cos_a * dx + sin_a * dy
            ly = -sin_a * dx + cos_a * dy
            inradius = s1 * np.sqrt(3) / 2
            edge_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
            max_proj = -np.inf
            for ea in edge_angles:
                nx = np.cos(ea)
                ny = np.sin(ea)
                proj = lx * nx + ly * ny
                max_proj = max(max_proj, proj)
            depth = max_proj - inradius
            penalty += max(0, depth) ** 2
    
    # Also check center-to-center distance as a baseline
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    min_dist = 2.0  # For unit hexagons, minimum center distance for no overlap
    if dist < min_dist:
        penalty += (min_dist - dist) ** 2
    
    return penalty

def _containment_penalty(cx, cy, s, a, ocx, ocy, os, oa, smooth_eps=0.01):
    """
    Smooth penalty for inner hexagon not being contained in outer hexagon.
    """
    verts = _hex_vertices(cx, cy, s, a)
    penalty = 0.0
    for vx, vy in verts:
        if not _point_in_hexagon(vx, vy, ocx, ocy, os, oa):
            # Compute how far outside
            angle_rad = np.radians(oa)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            dx = vx - ocx
            dy = vy - ocy
            lx = cos_a * dx + sin_a * dy
            ly = -sin_a * dx + cos_a * dy
            inradius = os * np.sqrt(3) / 2
            edge_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
            max_proj = -np.inf
            for ea in edge_angles:
                nx = np.cos(ea)
                ny = np.sin(ea)
                proj = lx * nx + ly * ny
                max_proj = max(max_proj, proj)
            excess = max_proj - inradius
            if excess > 0:
                penalty += excess ** 2
    return penalty

def _objective(params):
    """
    params: [outer_cx, outer_cy, outer_side, outer_angle, 
             inner1_cx, inner1_cy, inner1_angle,
             inner2_cx, inner2_cy, inner2_angle,
             ...]
    Total: 4 + 3*N = 4 + 33 = 37 params
    """
    ocx, ocy, os, oa = params[0:4]
    
    total_penalty = 0.0
    total_penalty += os ** 2  # Objective: minimize outer side (squared for smoothness)
    
    inner_hexes = []
    for i in range(N):
        base = 4 + i * 3
        icx, icy, ia = params[base:base+3]
        inner_hexes.append((icx, icy, 1.0, ia))
    
    # Overlap penalties between all pairs
    for i in range(N):
        for j in range(i+1, N):
            total_penalty += _hexagon_overlap_penalty(inner_hexes[i], inner_hexes[j])
    
    # Containment penalties
    for i in range(N):
        total_penalty += _containment_penalty(inner_hexes[i][0], inner_hexes[i][1], 
                                              inner_hexes[i][2], inner_hexes[i][3],
                                              ocx, ocy, os, oa)
    
    return total_penalty

def _extract_result(params):
    """Extract the result from parameters."""
    ocx, ocy, os, oa = params[0:4]
    inner = []
    for i in range(N):
        base = 4 + i * 3
        icx, icy, ia = params[base:base+3]
        inner.append([icx, icy, ia])
    return inner, (ocx, ocy), os, oa

def _validate(inner, center, side, angle):
    """Check if the packing is valid."""
    ocx, ocy = center
    os = side
    oa = angle
    
    # Check containment
    for icx, icy, ia in inner:
        verts = _hex_vertices(icx, icy, 1.0, ia)
        for vx, vy in verts:
            if not _point_in_hexagon(vx, vy, ocx, ocy, os, oa):
                return False
    
    # Check no overlaps
    for i in range(N):
        for j in range(i+1, N):
            v1 = (inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = (inner[j][0], inner[j][1], 1.0, inner[j][2])
            # Check if any vertex of one is inside the other
            verts1 = _hex_vertices(v1[0], v1[1], v1[2], v1[3])
            verts2 = _hex_vertices(v2[0], v2[1], v2[2], v2[3])
            for vx, vy in verts1:
                if _point_in_hexagon(vx, vy, v2[0], v2[1], v2[2], v2[3]):
                    return False
            for vx, vy in verts2:
                if _point_in_hexagon(vx, vy, v1[0], v1[1], v1[2], v1[3]):
                    return False
    
    return True

def _initial_guess_hexagonal():
    """Create an initial guess with hexagonal arrangement."""
    # Central hexagon at origin
    # 6 hexagons around it at distance 2
    # 4 more hexagons further out
    
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # Ring of 6 at distance 2
    for k in range(6):
        angle = k * 60.0
        x = 2.0 * np.cos(np.radians(angle))
        y = 2.0 * np.sin(np.radians(angle))
        inner.append([x, y, 0.0])
    
    # 4 more hexagons - place them at positions that extend the pattern
    # For a hexagonal lattice, the next ring would be at distance 4
    # Place 4 of them
    for k in range(4):
        angle = k * 90.0 + 30.0
        x = 4.0 * np.cos(np.radians(angle))
        y = 4.0 * np.sin(np.radians(angle))
        inner.append([x, y, 0.0])
    
    # Outer hexagon needs to contain all these
    # The farthest points are at distance ~4 from center
    # A hexagon with side s has inradius s*sqrt(3)/2 and circumradius s
    # Need circumradius >= 4 + 1 (for the inner hexagon extent)
    # So s >= 5 approximately
    return inner, (0.0, 0.0), 6.0, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Get initial guess
    inner, center, side, angle = _initial_guess_hexagonal()
    
    # Convert to params
    params = np.zeros(4 + 3 * N)
    params[0:4] = [center[0], center[1], side, angle]
    for i in range(N):
        base = 4 + i * 3
        params[base] = inner[i][0]
        params[base+1] = inner[i][1]
        params[base+2] = inner[i][2]
    
    best_params = params.copy()
    best_score = side  # We track the outer side length
    
    # Run optimization with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb initial guess for diversity
        if restart > 0:
            np.random.seed(restart * 12345)
            perturbed = best_params.copy()
            # Add small random perturbations to inner positions
            for i in range(N):
                base = 4 + i * 3
                perturbed[base] += np.random.uniform(-0.5, 0.5)
                perturbed[base+1] += np.random.uniform(-0.5, 0.5)
                perturbed[base+2] += np.random.uniform(-10, 10)
            # Also perturb outer
            perturbed[0] += np.random.uniform(-1, 1)
            perturbed[1] += np.random.uniform(-1, 1)
            perturbed[2] = max(3.0, perturbed[2] + np.random.uniform(-1, 1))
            current_params = perturbed
        else:
            current_params = params.copy()
        
        # Run L-BFGS-B
        try:
            result = minimize(_objective, current_params, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.success or result.fun < _objective(best_params):
                if result.fun < _objective(best_params):
                    best_params = result.x.copy()
                    _, _, best_side, _ = _extract_result(best_params)
                    best_score = best_side
        except Exception:
            pass
        
        if time.time() > deadline - 10.0:
            break
    
    # Extract best result
    inner, center, side, angle = _extract_result(best_params)
    
    # Validate and if invalid, try to fix
    if not _validate(inner, center, side, angle):
        # Try a simpler, known-valid configuration
        # Place hexagons in a line with proper spacing
        inner = []
        for k in range(N):
            x = 2.0 * (k - N // 2)
            inner.append([x, 0.0, 0.0])
        # Outer hexagon large enough
        total_width = 2.0 * (N - 1) + 2.0  # spacing + hexagon width
        side = total_width / np.sqrt(3) + 1.0
        return inner, (0.0, 0.0), side, 0.0
    
    return inner, center, side, angle

