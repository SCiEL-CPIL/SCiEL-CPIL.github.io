import numpy as np
from itertools import product, combinations
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def hex_vertices(center, side, angle_deg):
    """Vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.pi / 3 * i
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def polygon_overlap(v1, v2):
    """Check if two convex polygons overlap using SAT."""
    for verts in (v1, v2):
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            min1, max1 = np.dot(v1, normal).min(), np.dot(v1, normal).max()
            min2, max2 = np.dot(v2, normal).min(), np.dot(v2, normal).max()
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
    return True

def points_in_hexagon(points, center, side, angle_deg):
    """Check if all points are inside a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    # Hexagon sides are at distance side*cos(30) = side*sqrt(3)/2 from center
    apothem = side * np.sqrt(3) / 2
    for px, py in points:
        dx, dy = px - cx, py - cy
        # Rotate into hex frame
        rx = dx * np.cos(-a) - dy * np.sin(-a)
        ry = dx * np.sin(-a) + dy * np.cos(-a)
        # Check against 3 pairs of parallel sides
        for k in range(3):
            theta = np.pi / 3 * k
            nx, ny = np.cos(theta), np.sin(theta)
            proj = rx * nx + ry * ny
            if abs(proj) > apothem + 1e-9:
                return False
    return True

def enclosing_hex_side(points, center, angle_deg):
    """Minimum side length of hexagon with given center and angle containing all points."""
    cx, cy = center
    a = np.radians(angle_deg)
    max_proj = 0
    for k in range(3):
        theta = np.pi / 3 * k
        nx, ny = np.cos(theta), np.sin(theta)
        for px, py in points:
            dx, dy = px - cx, py - cy
            rx = dx * np.cos(-a) - dy * np.sin(-a)
            ry = dx * np.sin(-a) + dy * np.cos(-a)
            proj = abs(rx * nx + ry * ny)
            max_proj = max(max_proj, proj)
    return 2 * max_proj / np.sqrt(3)

def hex_lattice_positions(n_cells, seed_shape):
    """Generate positions of hexagons on hex lattice. seed_shape is a list of (q,r) axial coords."""
    pos = []
    for q, r in seed_shape:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pos.append((x, y))
    return np.array(pos)

def generate_clusters():
    """Generate various 12-cell hexagonal cluster shapes."""
    clusters = []
    
    # Shape 1: 3x4 rectangle in hex coords
    # Axial: q in [0,2], r in [0,3] with offset
    rect = []
    for q in range(3):
        for r in range(4):
            rect.append((q, r))
    clusters.append(rect)
    
    # Shape 2: 2x6
    rect2 = []
    for q in range(2):
        for r in range(6):
            rect2.append((q, r))
    clusters.append(rect2)
    
    # Shape 3: 4x3
    rect3 = []
    for q in range(4):
        for r in range(3):
            rect3.append((q, r))
    clusters.append(rect3)
    
    # Shape 4: Hexagonal-ish: center ring + extensions
    # A hex of radius 1 has 7 cells, need 5 more
    hex1 = [(0,0)]
    for q, r in [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]:
        hex1.append((q,r))
    # Add 5 more around
    hex1 += [(2,0),(2,-1),(0,-2),(-2,0),(-2,1)]
    clusters.append(hex1)
    
    # Shape 5: Two rows of 6
    two_rows = []
    for q in range(6):
        two_rows.append((q, 0))
        two_rows.append((q, 1))
    clusters.append(two_rows)
    
    # Shape 6: Compact: 4+4+4
    compact = []
    for q in range(4):
        compact.append((q, 0))
    for q in range(4):
        compact.append((q+1, 1))
    for q in range(4):
        compact.append((q, 2))
    clusters.append(compact)
    
    # Shape 7: 5+4+3
    s7 = []
    for q in range(5):
        s7.append((q, 0))
    for q in range(4):
        s7.append((q+1, 1))
    for q in range(3):
        s7.append((q+1, 2))
    clusters.append(s7)
    
    # Shape 8: 6+6 offset
    s8 = []
    for q in range(6):
        s8.append((q, 0))
    for q in range(6):
        s8.append((q, 1))
    clusters.append(s8)
    
    # Shape 9: 3 rows: 4,4,4 with offsets
    s9 = []
    for q in range(4):
        s9.append((q, 0))
    for q in range(4):
        s9.append((q-1, 1))
    for q in range(4):
        s9.append((q, 2))
    clusters.append(s9)
    
    # Shape 10: 2 rows of 5 + 2 extra
    s10 = []
    for q in range(5):
        s10.append((q, 0))
        s10.append((q, 1))
    s10 += [(0, 2), (4, 2)]
    clusters.append(s10)
    
    # Shape 11: 3x4 with corners removed and added elsewhere
    s11 = []
    for q in range(3):
        for r in range(4):
            s11.append((q, r))
    clusters.append(s11)
    
    # Shape 12: Diamond-like
    s12 = []
    for r in range(-2, 3):
        width = 5 - abs(r)
        if width <= 0:
            continue
        start = -width // 2
        for q in range(start, start + width):
            if len(s12) < 12:
                s12.append((q, r))
    # Pad if needed
    while len(s12) < 12:
        s12.append((len(s12) - 6, 3))
    s12 = s12[:12]
    clusters.append(s12)
    
    return clusters

def check_no_overlap(inner_positions, angle_deg):
    """Check that unit hexagons at given positions and angles don't overlap."""
    verts_list = []
    for (px, py) in inner_positions:
        verts = hex_vertices((px, py), 1.0, angle_deg)
        verts_list.append(verts)
    
    for i in range(len(verts_list)):
        for j in range(i+1, len(verts_list)):
            if polygon_overlap(verts_list[i], verts_list[j]):
                return False
    return True

def optimize_cluster(cluster, time_budget):
    """Optimize a cluster configuration."""
    positions = hex_lattice_positions(12, cluster)
    
    # Center the cluster
    centroid = positions.mean(axis=0)
    positions = positions - centroid
    
    best_side = float('inf')
    best_config = None
    
    # Try various orientations
    for inner_angle in np.arange(0, 60, 5):
        if time.time() - START_TIME > time_budget:
            break
        
        # Check no overlap
        if not check_no_overlap(positions, inner_angle):
            continue
        
        # Get vertices of all hexagons
        all_verts = []
        for (px, py) in positions:
            verts = hex_vertices((px, py), 1.0, inner_angle)
            all_verts.append(verts)
        
        all_points = np.vstack(all_verts)
        
        # Try various outer hexagon orientations
        for outer_angle in np.arange(0, 30, 3):
            if time.time() - START_TIME > time_budget:
                break
            
            # Find optimal center: minimize max projection
            # The optimal center is the center of the bounding box in hex coords
            # Try center at centroid and nearby
            for cx_offset in np.arange(-1, 1.5, 0.5):
                for cy_offset in np.arange(-1, 1.5, 0.5):
                    center = (cx_offset, cy_offset)
                    side = enclosing_hex_side(all_points, center, outer_angle)
                    if side < best_side:
                        # Verify all points are inside
                        if points_in_hexagon(all_points, center, side, outer_angle):
                            best_side = side
                            best_config = (positions.copy(), inner_angle, center, side, outer_angle)
    
    return best_side, best_config

def construct():
    global START_TIME
    START_TIME = time.time()
    time_budget = TIME_LIMIT - 5.0  # 5s safety margin
    
    clusters = generate_clusters()
    
    best_overall_side = float('inf')
    best_overall_config = None
    
    for idx, cluster in enumerate(clusters):
        if time.time() - START_TIME > time_budget:
            break
        
        # Give each cluster a time slice
        cluster_time = time_budget - (time.time() - START_TIME)
        cluster_budget = min(cluster_time, 12.0)
        
        side, config = optimize_cluster(cluster, cluster_time)
        
        if side < best_overall_side:
            best_overall_side = side
            best_overall_config = config
    
    # If we found a good config, use it
    if best_overall_config is not None:
        positions, inner_angle, center, side, outer_angle = best_overall_config
        
        # Format inner hexagons
        inner = []
        for (px, py) in positions:
            inner.append([float(px), float(py), float(inner_angle)])
        
        return inner, (float(center[0]), float(center[1])), float(side), float(outer_angle)
    else:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
