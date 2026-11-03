"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 12

def _hex_vertices(cx, cy, side, rot_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length 'side',
    rotated by rot_deg degrees. Flat-top orientation when rot=0."""
    verts = []
    for k in range(6):
        angle = math.radians(60.0 * k + rot_deg)
        x = cx + side * math.cos(angle)
        y = cy + side * math.sin(angle)
        verts.append((x, y))
    return verts

def _get_all_vertices(positions):
    """Get all vertices of all inner hexagons."""
    all_verts = []
    for (cx, cy, rot) in positions:
        verts = _hex_vertices(cx, cy, 1.0, rot)
        all_verts.extend(verts)
    return all_verts

def _min_side_for_config(positions, outer_rot):
    """
    Calculate the minimum side length of a regular hexagon with given rotation
    that contains all inner hexagons.
    We assume the center of the outer hexagon is the center of the bounding box
    of the inner hexagons' vertices.
    """
    all_verts = _get_all_vertices(positions)
    if not all_verts:
        return 0.0
    
    # Calculate bounding box center
    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    
    # Normals for the outer hexagon sides
    # Flat-top hexagon (rot=0) has normals at 90, 150, 210, 270, 330, 30?
    # Vertices at 0, 60, 120, 180, 240, 300.
    # Side midpoints (normals) are at 30, 90, 150, 210, 270, 330.
    # General rotation: add outer_rot.
    
    max_range = 0.0
    for k in range(6):
        angle = math.radians(60.0 * k + 30.0 + outer_rot)
        nx = math.cos(angle)
        ny = math.sin(angle)
        
        projections = []
        for (px, py) in all_verts:
            # Project relative to center
            proj = (px - cx) * nx + (py - cy) * ny
            projections.append(proj)
        
        p_min = min(projections)
        p_max = max(projections)
        range_val = p_max - p_min
        
        # The width in this direction is range_val.
        # The outer hexagon must have a width >= range_val in this normal direction.
        # The width of a regular hexagon of side S in the normal direction is 2 * apothem = S * sqrt(3).
        # So S * sqrt(3) >= range_val  =>  S >= range_val / sqrt(3)
        
        required_side = range_val / math.sqrt(3)
        if required_side > max_range:
            max_range = required_side
            
    return max_range

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    s3 = math.sqrt(3)
    configs = []
    
    # Separation factor to ensure disjointness (avoid touching)
    sep = 1.005
    
    # Config A: 4x3 honeycomb (flat-top)
    # Centers:
    # Row 0 (y=0): x = 0, 3, 6, 9
    # Row 1 (y=s3/2): x = 1.5, 4.5, 7.5, 10.5
    # Row 2 (y=s3): x = 0, 3, 6, 9
    config_a = []
    for i in range(4):
        config_a.append([3.0 * i * sep, 0.0, 0.0])
    for i in range(4):
        config_a.append([3.0 * i * sep + 1.5 * sep, (s3 / 2.0) * sep, 0.0])
    for i in range(4):
        config_a.append([3.0 * i * sep, s3 * sep, 0.0])
    configs.append(config_a)
    
    # Config B: 3x4 honeycomb (transposed)
    # This is effectively rotating the arrangement or just swapping x/y logic
    # Let's generate a 3-wide, 4-high honeycomb
    config_b = []
    for i in range(3): # rows
        for j in range(4): # cols
            x = 3.0 * j * sep
            y = s3 * i * sep
            if i % 2 == 1:
                x += 1.5 * sep
            config_b.append([x, y, 0.0])
    configs.append(config_b)
    
    # Config C: 6x2 honeycomb
    config_c = []
    for i in range(6):
        for j in range(2):
            x = 3.0 * i * sep
            y = s3 * j * sep
            if j % 2 == 1:
                x += 1.5 * sep
            config_c.append([x, y, 0.0])
    configs.append(config_c)
    
    # Config D: 2x6 honeycomb
    config_d = []
    for i in range(2):
        for j in range(6):
            x = 3.0 * j * sep
            y = s3 * i * sep
            if i % 2 == 1:
                x += 1.5 * sep
            config_d.append([x, y, 0.0])
    configs.append(config_d)
    
    best_side = float('inf')
    best_config = None
    best_outer_rot = 0.0
    
    for config in configs:
        if time.time() > deadline - 10.0:
            break
        
        # Center the configuration
        all_verts = _get_all_vertices(config)
        xs = [v[0] for v in all_verts]
        ys = [v[1] for v in all_verts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        
        centered = [[p[0] - cx, p[1] - cy, p[2]] for p in config]
        
        # Try different outer rotations (0, 30, 15, 45, etc.)
        for rot in [0.0, 30.0, 15.0, 45.0]:
            if time.time() > deadline - 10.0:
                break
            
            side = _min_side_for_config(centered, rot)
            if side < best_side:
                best_side = side
                best_config = centered
                best_outer_rot = rot
    
    if best_config is not None and best_side < float('inf'):
        inner = [[p[0], p[1], p[2]] for p in best_config]
        return inner, (0.0, 0.0), best_side, best_outer_rot
    
    # Fallback
    inner = [[3.0 * i * sep, 0.0, 0.0] for i in range(N)]
    return inner, (0.0, 0.0), 50.0, 0.0

# EVOLVE-BLOCK-END
