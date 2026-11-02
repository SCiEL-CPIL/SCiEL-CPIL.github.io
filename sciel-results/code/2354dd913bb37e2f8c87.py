"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with rotation.
    Returns squared distance from center in normalized hex coordinates."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = -math.radians(angle_deg)
    lx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    ly = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    
    # For a unit regular hexagon with flat-top orientation (vertex at angle 0),
    # the hexagon can be described as: max(|lx|, |ly|, |lx/2 + ly*sqrt(3)/2|) <= 1
    # But our hex has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The inradius is sqrt(3)/2, circumradius is 1
    
    # Check using the six half-planes
    # Edge normals for hexagon with vertex at angle 0:
    # The edges are perpendicular to directions at 30, 90, 150, 210, 270, 330 degrees
    # Inradius = sqrt(3)/2
    inradius = math.sqrt(3) / 2
    for k in range(6):
        a = math.radians(30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        dist = lx * nx + ly * ny
        if dist > inradius:
            return None  # Outside
    return lx * lx + ly * ly

def _hex_overlap_dist_sq(c1x, c1y, a1, c2x, c2y, a2):
    """Check if two unit hexagons overlap. Returns None if no overlap, else positive value."""
    # Quick rejection: if center distance > 2.0 (max possible distance for overlap), no overlap
    dx = c2x - c1x
    dy = c2y - c1y
    d2 = dx * dx + dy * dy
    if d2 > 4.0:  # (2*1)^2, max circumradius sum
        return None
    
    # Check if any vertex of hex1 is inside hex2, or vice versa
    verts1 = _get_hex_vertices(c1x, c1y, a1)
    verts2 = _get_hex_vertices(c2x, c2y, a2)
    
    for vx, vy in verts1:
        if _point_in_hex_dist_sq(vx, vy, c2x, c2y, a2) is not None:
            return 1.0  # Overlap detected
    for vx, vy in verts2:
        if _point_in_hex_dist_sq(vx, vy, c1x, c1y, a1) is not None:
            return 1.0  # Overlap detected
    
    return None

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different angles for the outer hexagon
    for angle_deg in [0.0, 30.0]:
        # For each angle, we need to find the minimum enclosing hexagon
        # A regular hexagon with side s and angle theta contains a point (px, py) if:
        # For each of 6 directions, the projection is within s * sqrt(3)/2
        
        # The hexagon can be described as intersection of 6 half-planes
        # Normal directions are at angle + 30 + 60*k for k=0..5
        # Inradius = s * sqrt(3)/2
        
        # To minimize s, we can use the fact that the optimal enclosing hexagon
        # will have at least 3 points on its boundary
        
        # Simple approach: try centers at various positions
        # The center should be near the centroid of the inner hexagons
        cx_vals = [h[0] for h in inner_hexes]
        cy_vals = [h[1] for h in inner_hexes]
        cx_mean = sum(cx_vals) / len(cx_vals)
        cy_mean = sum(cy_vals) / len(cy_vals)
        
        # Try a grid of centers around the mean
        for cx in [cx_mean + dx for dx in [-2.0, -1.0, 0.0, 1.0, 2.0]]:
            for cy in [cy_mean + dy for dy in [-2.0, -1.0, 0.0, 1.0, 2.0]]:
                # For this center and angle, compute required side length
                required_s = 0.0
                inradius_factor = math.sqrt(3) / 2
                
                for k in range(6):
                    a = math.radians(angle_deg + 30 + 60 * k)
                    nx = math.cos(a)
                    ny = math.sin(a)
                    
                    # For each inner hexagon, find max projection
                    max_proj = 0.0
                    for ix, iy, iangle in inner_hexes:
                        # Project all vertices of inner hexagon onto normal direction
                        verts = _get_hex_vertices(ix, iy, iangle)
                        for vx, vy in verts:
                            proj = (vx - cx) * nx + (vy - cy) * ny
                            if proj > max_proj:
                                max_proj = proj
                        # Also check the -normal direction
                        min_proj = 0.0
                        for vx, vy in verts:
                            proj = (vx - cx) * nx + (vy - cy) * ny
                            if proj < min_proj:
                                min_proj = proj
                        # The extent in this direction is max(max_proj, -min_proj)
                        extent = max(max_proj, -min_proj)
                        if extent > max_proj:
                            max_proj = extent
                    
                    # Required inradius for this direction
                    req_inradius = max_proj
                    req_side = req_inradius / inradius_factor
                    if req_side > required_s:
                        required_s = req_side
                
                if required_s < best_side:
                    best_side = required_s
                    best_center = (cx, cy)
                    best_angle = angle_deg
    
    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Place hexagons in a hexagonal lattice
    # Lattice vectors: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    sqrt3 = math.sqrt(3)
    v1x, v1y = sqrt3, 0.0
    v2x, v2y = sqrt3 / 2, 1.5
    
    # Generate candidate positions in a hexagonal cluster
    # For 11 hexagons, use a compact arrangement
    positions = []
    
    # Try different cluster shapes
    # Shape 1: 3-3-3-1-1 (rows of 3,3,3,1,1)
    for j in range(-2, 3):
        for i in range(-1, 2):
            if len(positions) >= N:
                break
            x = i * v1x + j * v2x
            y = j * v2y
            positions.append((x, y))
        if len(positions) >= N:
            break
    
    # Ensure we have exactly N positions
    while len(positions) < N:
        # Add more positions in a spiral pattern
        added = False
        for r in range(1, 10):
            for k in range(6):
                a = math.radians(60 * k)
                x = r * sqrt3 * math.cos(a)
                y = r * 1.5 * math.sin(a)
                # Check if this position is already used
                dup = False
                for px, py in positions:
                    if abs(px - x) < 0.01 and abs(py - y) < 0.01:
                        dup = True
                        break
                if not dup and len(positions) < N:
                    positions.append((x, y))
                    added = True
                    break
            if added:
                break
        if not added:
            break
    
    # Create inner hexagons with all angles = 0
    inner = [[pos[0], pos[1], 0.0] for pos in positions[:N]]
    
    # Check for overlaps and adjust if needed
    # For now, assume the lattice spacing prevents overlap
    # Center distance in lattice: sqrt(3) ≈ 1.732, which is exactly the edge-to-edge distance
    # for unit hexagons, so they should be just touching
    
    # Compute minimum enclosing hexagon
    if time.time() > _DEADLINE - 10.0:
        # Quick estimate
        xs = [h[0] for h in inner]
        ys = [h[1] for h in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        # Rough side length estimate
        max_dist = 0
        for h in inner:
            d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
            if d > max_dist:
                max_dist = d
        side = max_dist / (math.sqrt(3) / 2) + 0.1
        return inner, (cx, cy), side, 0.0
    
    cx, cy, side, angle = _min_enclosing_hex_side(inner)
    
    # Verify no overlaps
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            if _hex_overlap_dist_sq(inner[i][0], inner[i][1], inner[i][2],
                                     inner[j][0], inner[j][1], inner[j][2]) is not None:
                # Overlap detected, fall back to safer spacing
                inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
                return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
