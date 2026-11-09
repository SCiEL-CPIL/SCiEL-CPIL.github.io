import numpy as np
import time

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # Return the seed program's result as a safe fallback
    def seed_result():
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # For a regular hexagon with side length s, the vertices (from angle 0) are:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The distance from center to vertex (circumradius) is s
    # The distance from center to edge midpoint (apothem) is s * sqrt(3) / 2
    
    sqrt3 = np.sqrt(3.0)
    
    def hexagon_vertices(cx, cy, s, angle_deg):
        """Return vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
        a = np.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + i * np.pi / 3.0
            x = cx + s * np.cos(theta)
            y = cy + s * np.sin(theta)
            verts.append((x, y))
        return np.array(verts)
    
    def point_in_hexagon(px, py, cx, cy, s, angle_deg):
        """Check if point (px,py) is inside or on boundary of hexagon centered at (cx,cy) with side s."""
        # Transform point to local coordinates
        a = np.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        # Rotate by -a
        cos_a = np.cos(-a)
        sin_a = np.sin(-a)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # In local coords, hexagon vertices are at angles 0, 60, 120, 180, 240, 300
        # The hexagon is the intersection of 6 half-planes
        # For a hexagon with circumradius s, the apothem is s*sqrt(3)/2
        # The 6 faces are at distances apothem from center
        apothem = s * sqrt3 / 2.0
        for i in range(6):
            theta = i * np.pi / 3.0
            # Normal direction for face i
            nx = np.cos(theta)
            ny = np.sin(theta)
            # Distance from center to this face along normal is apothem
            dist = lx * nx + ly * ny
            if dist > apothem + 1e-10:
                return False
        return True
    
    def hexagons_disjoint(v1, a1, v2, a2, tol=1e-9):
        """Check if two unit hexagons at centers v1,v2 with angles a1,a2 are disjoint (no interior overlap)."""
        # Check if any vertex of one is inside the other
        verts1 = hexagon_vertices(v1[0], v1[1], 1.0, a1)
        verts2 = hexagon_vertices(v2[0], v2[1], 1.0, a2)
        
        for vx, vy in verts1:
            if point_in_hexagon(vx, vy, v2[0], v2[1], 1.0, a2):
                return False
        for vx, vy in verts2:
            if point_in_hexagon(vx, vy, v1[0], v1[1], 1.0, a1):
                return False
        return True
    
    def outer_hexagon_side_length(inner_positions, inner_angles, outer_center, outer_angle):
        """Compute the minimum side length of an outer hexagon at given center/angle that contains all inner hexagons."""
        cx, cy = outer_center
        oa = np.radians(outer_angle)
        
        # For a hexagon containing all points, we need to find the minimum s such that
        # all inner hexagon vertices are inside the outer hexagon.
        # The outer hexagon with center (cx,cy), side s, rotation oa has 6 faces.
        # A point (px,py) is inside iff for all i: dot(p - c, n_i) <= s * sqrt(3)/2
        # where n_i are the face normals.
        
        # Collect all vertices of all inner hexagons
        all_verts = []
        for pos, ang in zip(inner_positions, inner_angles):
            verts = hexagon_vertices(pos[0], pos[1], 1.0, ang)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Face normals of outer hexagon
        normals = []
        for i in range(6):
            theta = oa + i * np.pi / 3.0
            normals.append((np.cos(theta), np.sin(theta)))
        
        # For each face normal, find max dot product with (p - center)
        s_min = 0.0
        for nx, ny in normals:
            dots = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
            max_dot = np.max(dots)
            required_s = max_dot / (sqrt3 / 2.0)
            if required_s > s_min:
                s_min = required_s
        return s_min
    
    # Strategy: Try different configurations of 12 unit hexagons
    # and for each, find optimal outer hexagon center and angle.
    
    best_side = 40.0
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Configuration 1: Linear arrangement (seed)
    # Try a few different arrangements
    
    # Let's try a 3x4 grid-like arrangement
    # Hexagons can be arranged in a honeycomb pattern
    
    # Configuration: 4 rows of 3, or 3 rows of 4
    # For a honeycomb, adjacent hexagons in same row are 2 apart (side by side)
    # Offset rows are shifted by 1 horizontally and sqrt(3) vertically
    
    configs = []
    
    # Config A: Linear (seed)
    config_a = []
    for k in range(-(N // 2), N - N // 2):
        config_a.append((4.0 * k, 0.0, 0.0))
    configs.append(config_a)
    
    # Config B: 4x3 grid in honeycomb
    # Row 0: 3 hexagons at x = -2, 0, 2; y = 0
    # Row 1: 3 hexagons at x = -3, -1, 1; y = sqrt(3)
    # Row 2: 3 hexagons at x = -2, 0, 2; y = 2*sqrt(3)
    # Row 3: 3 hexagons at x = -3, -1, 1; y = 3*sqrt(3)
    config_b = []
    for row in range(4):
        for col in range(3):
            if row % 2 == 0:
                x = (col - 1) * 2.0
            else:
                x = (col - 1) * 2.0 - 1.0
            y = row * sqrt3
            config_b.append((x, y, 0.0))
    configs.append(config_b)
    
    # Config C: 3x4 grid in honeycomb (transpose)
    config_c = []
    for col in range(4):
        for row in range(3):
            if col % 2 == 0:
                x = col * 2.0
            else:
                x = col * 2.0 - 1.0
            y = (row - 1) * sqrt3
            config_c.append((x, y, 0.0))
    configs.append(config_c)
    
    # Config D: 2x6 honeycomb
    config_d = []
    for row in range(2):
        for col in range(6):
            if row == 0:
                x = (col - 2.5) * 2.0
            else:
                x = (col - 2.5) * 2.0 - 1.0
            y = row * sqrt3
            config_d.append((x, y, 0.0))
    configs.append(config_d)
    
    # Config E: 6x2 honeycomb
    config_e = []
    for col in range(6):
        for row in range(2):
            if col == 0:
                x = (row - 0.5) * 2.0
            else:
                x = (row - 0.5) * 2.0 - 1.0
            y = (col - 2.5) * sqrt3
            config_e.append((x, y, 0.0))
    configs.append(config_e)
    
    # For each config, try to optimize outer hexagon center and angle
    def optimize_outer(config, deadline):
        """Find best outer hexagon for a given inner config."""
        positions = [(p[0], p[1]) for p in config]
        angles = [p[2] for p in config]
        
        best_s = float('inf')
        best_c = (0.0, 0.0)
        best_a = 0.0
        
        # Try a grid of outer center positions and angles
        # First, estimate bounding box
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        cx_range = np.linspace(x_min - 2, x_max + 2, 15)
        cy_range = np.linspace(y_min - 2, y_max + 2, 15)
        angle_range = np.linspace(0, 180, 13)  # 12 outer angles
        
        for cx in cx_range:
            if time.time() > deadline:
                break
            for cy in cy_range:
                if time.time() > deadline:
                    break
                for oa in angle_range:
                    s = outer_hexagon_side_length(positions, angles, (cx, cy), oa)
                    if s < best_s:
                        best_s = s
                        best_c = (float(cx), float(cy))
                        best_a = float(oa)
        
        return best_s, best_c, best_a
    
    deadline = start_time + DEADLINE - 5.0
    
    for config in configs:
        if time.time() > deadline:
            break
        s, center, angle = optimize_outer(config, deadline)
        if s < best_side:
            best_side = s
            best_inner = config
            best_center = center
            best_outer_angle = angle
    
    if best_inner is not None:
        return best_inner, best_center, best_side, best_outer_angle
    
    return seed_result()
