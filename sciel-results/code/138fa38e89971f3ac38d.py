"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg=0.0):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) with given rotation."""
    rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = rad + i * math.pi / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_outer_side_flex(inner_positions, inner_angles):
    """
    Given inner hexagon centers and angles, find minimum outer hexagon side length
    (with center at origin and variable rotation) that contains all inner hexagons.
    We optimize over outer rotation.
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy), ang in zip(inner_positions, inner_angles):
        all_verts.extend(_hex_vertices(cx, cy, ang))
    
    if not all_verts:
        return 0.0
    
    # The outer hexagon with side s, center at origin, rotation theta has
    # support function: for direction u, the max dot product of vertices with u
    # must be <= s * cos(pi/6) = s * sqrt(3)/2 ... 
    # Actually for a regular hexagon with side s, the apothem (inradius) is s*sqrt(3)/2
    # and circumradius is s.
    # A point p is inside if for all 6 directions (normals to edges), dot(p, n_i) <= apothem
    # where n_i are unit normals at angles theta + i*pi/3
    # apothem = s * sqrt(3) / 2
    # So s >= max_i dot(p, n_i) / (sqrt(3)/2) for all p
    # s >= 2/sqrt(3) * max_i dot(p, n_i)
    
    best_s = float('inf')
    # Try many outer rotations
    for theta_deg in range(0, 60, 1):
        theta = math.radians(theta_deg)
        max_proj = 0.0
        for i in range(6):
            a = theta + i * math.pi / 3
            nx, ny = math.cos(a), math.sin(a)
            for (px, py) in all_verts:
                d = abs(px * nx + py * ny)
                if d > max_proj:
                    max_proj = d
        s = 2.0 * max_proj / math.sqrt(3.0)
        if s < best_s:
            best_s = s
    
    return best_s

def _cluster_positions(rows):
    """
    Generate positions for a hex cluster.
    rows: list of counts per row, e.g., [3,4,4] for 11 hexagons
    Using flat-top hex lattice with spacing sqrt(3) horizontally, 
    vertical spacing 1.5, horizontal offset sqrt(3)/2 for odd rows.
    """
    positions = []
    total = sum(rows)
    # Center the cluster
    # Row r has rows[r] hexagons
    # x spacing: sqrt(3), y spacing: 1.5
    # Odd rows offset by sqrt(3)/2 in x
    sqrt3 = math.sqrt(3.0)
    all_pos = []
    for r, count in enumerate(rows):
        y = r * 1.5
        x_offset = (sqrt3 / 2.0) if (r % 2 == 1) else 0.0
        for k in range(count):
            x = x_offset + (k - (count - 1) / 2.0) * sqrt3
            all_pos.append((x, y))
    
    # Center the cluster at origin
    cx = sum(p[0] for p in all_pos) / len(all_pos)
    cy = sum(p[1] for p in all_pos) / len(all_pos)
    return [(p[0] - cx, p[1] - cy) for p in all_pos]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try various cluster configurations for 11 hexagons
    # Each is a list of row counts
    configs = [
        [11],           # single row
        [6, 5],         # 2 rows
        [5, 6],         
        [4, 4, 3],      # 3 rows
        [3, 4, 4],      
        [4, 3, 4],
        [3, 3, 3, 2],   # 4 rows
        [2, 3, 3, 3],
        [3, 2, 3, 3],
        [2, 2, 3, 2, 2],# 5 rows
        [2, 3, 2, 3, 1],
        [1, 2, 3, 3, 2],
        [2, 3, 3, 3],
        [3, 3, 3, 2],
        [1, 3, 4, 3],
        [3, 4, 3, 1],
        [2, 4, 3, 2],
        [3, 4, 4],
        [4, 4, 3],
        [1, 4, 5, 1],
        [2, 5, 3, 1],
    ]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_rotation = 0.0
    
    for config in configs:
        if time.time() > deadline - 5.0:
            break
        if sum(config) != N:
            continue
        
        positions = _cluster_positions(config)
        angles = [0.0] * N
        
        side = _min_outer_side_flex(positions, angles)
        if side < best_side:
            best_side = side
            best_positions = positions
            best_angles = angles
    
    # Also try with 30-degree rotated inner hexagons (pointy-top)
    for config in configs:
        if time.time() > deadline - 5.0:
            break
        if sum(config) != N:
            continue
        
        positions = _cluster_positions(config)
        angles = [30.0] * N
        
        side = _min_outer_side_flex(positions, angles)
        if side < best_side:
            best_side = side
            best_positions = positions
            best_angles = angles
    
    if best_positions is None:
        # Fallback: linear arrangement
        best_positions = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        best_angles = [0.0] * N
        best_side = 40.0
    
    # Build inner list
    inner = []
    for (x, y), ang in zip(best_positions, best_angles):
        inner.append([x, y, ang])
    
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
