import numpy as np
from itertools import product

N = 12

def hexagon_contains(points, center, side, angle_deg):
    """Check if all points are inside a regular hexagon."""
    angle = np.radians(angle_deg)
    apothem = side * np.sqrt(3) / 2
    # 6 normal directions for a regular hexagon
    for i in range(6):
        theta = angle + np.pi * i / 3
        nx, ny = np.cos(theta), np.sin(theta)
        for p in points:
            dx, dy = p[0] - center[0], p[1] - center[1]
            if dx * nx + dy * ny > apothem + 1e-9:
                return False
    return True

def min_outer_side_for_centers(centers, outer_angle=0.0):
    """Find minimum side length of outer hexagon containing all centers (plus hexagon extent)."""
    # For each center, the hexagon extends in all directions by its own apothem
    # We need: for all outer normals n, max_i (dot(c_i, n) + inner_apothem_i) <= outer_apothem
    # Inner hexagon apothem depends on rotation
    # Simplification: assume all inner hexagons have same orientation for now
    # Actually, we need to account for each inner hexagon's shape
    
    # For a unit hexagon with rotation alpha, its support function in direction n is:
    # h(n) = max over vertices of dot(v, n)
    # For regular hexagon with side 1, apothem = sqrt(3)/2, circumradius = 1
    # Support function: h(n) = max_k dot(v_k, n) where v_k are vertices
    
    # This is complex. Let's use a simpler approach: 
    # The minimum enclosing hexagon side can be found by binary search + feasibility check
    
    # For simplicity, let's compute the bounding box in the hexagonal sense
    # and iterate over possible outer angles
    
    best_side = float('inf')
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        # For each outer angle, find min side
        lo, hi = 1.0, 20.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if _feasible(centers, (0,0), mid, outer_angle):
                hi = mid
            else:
                lo = mid
        best_side = min(best_side, hi)
    return best_side

def _feasible(centers, outer_center, outer_side, outer_angle_deg):
    """Check if all unit hexagons (with optimal rotation per hexagon) fit inside outer hexagon."""
    # For each inner hexagon center, we need to find if there exists a rotation 
    # such that the hexagon fits inside the outer hexagon
    # This is complex, so let's assume all inner hexagons have rotation 0 or 30
    # and check containment
    
    angle = np.radians(outer_angle_deg)
    outer_apothem = outer_side * np.sqrt(3) / 2
    
    # Outer hexagon normals
    outer_normals = []
    for i in range(6):
        theta = angle + np.pi * i / 3
        outer_normals.append((np.cos(theta), np.sin(theta)))
    
    for cx, cy in centers:
        # Try inner hexagon rotations: 0, 30, 60, 90, 120, 150
        feasible = False
        for inner_angle_deg in [0, 30, 60, 90, 120, 150]:
            inner_angle = np.radians(inner_angle_deg)
            # Compute support function of unit hexagon in each outer normal direction
            ok = True
            for nx, ny in outer_normals:
                # Support function of unit regular hexagon rotated by inner_angle
                # in direction (nx, ny)
                # Vertices of unit hexagon at angles inner_angle + k*pi/3
                max_dot = 0
                for k in range(6):
                    vx = np.cos(inner_angle + np.pi * k / 3)
                    vy = np.sin(inner_angle + np.pi * k / 3)
                    dot = vx * nx + vy * ny
                    if dot > max_dot:
                        max_dot = dot
                # The hexagon centered at (cx,cy) extends to cx*nx + cy*ny + max_dot
                extent = cx * nx + cy * ny + max_dot
                if extent > outer_apothem + 1e-9:
                    ok = False
                    break
            if ok:
                feasible = True
                break
        if not feasible:
            return False
    return True

def construct():
    # Try various configurations of 12 hexagon centers
    # Configuration 1: Hexagonal grid pattern
    # In a hexagonal lattice, centers are at:
    # (i * sqrt(3) + j * sqrt(3)/2, j * 1.5) for integer i,j
    
    sqrt3 = np.sqrt(3)
    
    # Try a 3-4-3-2 pattern or similar
    # Let's try multiple configurations
    
    configs = []
    
    # Config 1: 4 rows: 3,4,3,2 = 12
    # Row spacing: 1.5 in y, offset in x
    row_counts = [3, 4, 3, 2]
    centers = []
    y_offset = 0
    for r, count in enumerate(row_counts):
        y = (r - 1.5) * 1.5  # Center the rows
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 2: 3-4-3-2 with different offset
    row_counts = [2, 3, 4, 3]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 1.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 3: 2-4-4-2
    row_counts = [2, 4, 4, 2]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 1.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 4: 3-3-3-3
    row_counts = [3, 3, 3, 3]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 1.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 5: 4-4-4
    row_counts = [4, 4, 4]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 1) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 6: 5-4-3
    row_counts = [5, 4, 3]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 1) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 7: 6-6
    row_counts = [6, 6]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 0.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 8: 1-4-5-2 (centered)
    row_counts = [1, 4, 5, 2]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 1.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 9: 2-5-4-1
    row_counts = [2, 5, 4, 1]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 1.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 10: 4-5-3
    row_counts = [4, 5, 3]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 1) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 11: 3-5-4
    row_counts = [3, 5, 4]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 1) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 12: 5-5-2
    row_counts = [5, 5, 2]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 1) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 13: 7-5
    row_counts = [7, 5]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 0.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 14: 8-4
    row_counts = [8, 4]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 0.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 15: 9-3
    row_counts = [9, 3]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 0.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 16: 10-2
    row_counts = [10, 2]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 0.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 17: 11-1
    row_counts = [11, 1]
    centers = []
    for r, count in enumerate(row_counts):
        y = (r - 0.5) * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for i in range(count):
            x = x_start + i * sqrt3
            centers.append((x, y))
    configs.append(centers)
    
    # Config 18: 12 in one row
    row_counts = [12]
    centers = []
    for i in range(12):
        x = (i - 5.5) * sqrt3
        centers.append((x, 0))
    configs.append(centers)
    
    # Also try with 30-degree rotated hexagonal lattice
    # In a rotated hex lattice, the spacing pattern changes
    # Let's try some more varied configurations
    
    # Config 19: Centered hexagonal pattern (like a flower)
    # 1 center + 6 around + 5 more = 12
    centers = [(0, 0)]
    # 6 neighbors at distance sqrt(3)
    for k in range(6):
        angle = np.pi * k / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append((x, y))
    # 5 more at next ring
    for k in range(6):
        angle = np.pi * k / 3 + np.pi / 6
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        centers.append((x, y))
    centers = centers[:12]  # Take first 12
    configs.append(centers)
    
    # Config 20: Another centered pattern
    centers = [(0, 0)]
    for k in range(6):
        angle = np.pi * k / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append((x, y))
    # Add 5 more at various positions
    extra_positions = [
        (2*sqrt3, 0), (sqrt3, 1.5*sqrt3), (-sqrt3, 1.5*sqrt3),
        (-2*sqrt3, 0), (-sqrt3, -1.5*sqrt3)
    ]
    for x, y in extra_positions:
        centers.append((x, y))
    configs.append(centers[:12])
    
    # Evaluate all configurations
    best_side = float('inf')
    best_config_idx = 0
    best_outer_angle = 0
    
    for idx, centers in enumerate(configs):
        for outer_angle in [0, 30, 60, 90, 120, 150]:
            # Binary search for minimum side
            lo, hi = 1.0, 30.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if _feasible(centers, (0, 0), mid, outer_angle):
                    hi = mid
                else:
                    lo = mid
            if hi < best_side:
                best_side = hi
                best_config_idx = idx
                best_outer_angle = outer_angle
    
    best_centers = configs[best_config_idx]
    
    # Build inner list with centers and angles
    # Assume all inner hexagons have 0 degree rotation for simplicity
    # (the _feasible function already checks if some rotation works)
    inner = []
    for cx, cy in best_centers:
        inner.append([float(cx), float(cy), 0.0])
    
    return inner, (0.0, 0.0), float(best_side), float(best_outer_angle)
