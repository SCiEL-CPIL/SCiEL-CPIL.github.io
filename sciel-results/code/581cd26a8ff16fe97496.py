"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    rx = dx * np.cos(-angle) - dy * np.sin(-angle)
    ry = dx * np.sin(-angle) + dy * np.cos(-angle)
    apothem = side * np.sqrt(3) / 2
    for k in range(6):
        edge_angle = k * np.pi / 3 + np.pi / 6
        nx, ny = np.cos(edge_angle), np.sin(edge_angle)
        if rx * nx + ry * ny > apothem:
            return False
    return True

def hexagon_contains_all(hex_center, hex_side, hex_angle, inner_hexes):
    """Check if the outer hexagon contains all inner hexagons."""
    hx, hy = hex_center
    for (ix, iy, iangle) in inner_hexes:
        verts = hex_vertices(ix, iy, 1.0, iangle)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, hx, hy, hex_side, hex_angle):
                return False
    return True

def compute_min_containing_hexagon(inner_hexes, num_angles=12):
    """Compute the minimum side length of a regular hexagon that contains all inner hexagons."""
    all_points = []
    for (ix, iy, iangle) in inner_hexes:
        verts = hex_vertices(ix, iy, 1.0, iangle)
        all_points.extend(verts)
    all_points = np.array(all_points)
    cx = np.mean(all_points[:, 0])
    cy = np.mean(all_points[:, 1])
    
    best_side = float('inf')
    best_angle = 0.0
    
    for angle_deg in np.linspace(0, 60, num_angles, endpoint=False):
        angle = np.radians(angle_deg)
        dx = all_points[:, 0] - cx
        dy = all_points[:, 1] - cy
        rx = dx * np.cos(-angle) - dy * np.sin(-angle)
        ry = dx * np.sin(-angle) + dy * np.cos(-angle)
        
        max_proj = 0
        for k in range(6):
            edge_angle = k * np.pi / 3 + np.pi / 6
            nx, ny = np.cos(edge_angle), np.sin(edge_angle)
            projections = rx * nx + ry * ny
            max_proj = max(max_proj, np.max(np.abs(projections)))
        
        side = 2 * max_proj / np.sqrt(3)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    return best_side, best_angle, (cx, cy)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    spacing = np.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = spacing * (q + r / 2.0)
        y = spacing * 3.0 / 2.0 * r
        return x, y
    
    best_result = None
    best_score = float('inf')
    
    first_ring = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    second_ring_all = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
                       (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)]
    
    for second_subset in itertools.combinations(range(12), 4):
        if time.time() > deadline:
            break
        
        positions_axial = [(0, 0)] + first_ring + [second_ring_all[i] for i in second_subset]
        positions = [axial_to_cartesian(q, r) for q, r in positions_axial]
        
        for angle_inner in [0.0, 30.0]:
            inner_hexes = [(x, y, angle_inner) for x, y in positions]
            side, angle, center = compute_min_containing_hexagon(inner_hexes)
            if side < best_score:
                best_score = side
                best_result = (inner_hexes, center, side, angle)
    
    # Try various row-based arrangements
    arrangements = []
    
    # 6-5
    arrangements.append([(spacing * k, 0.0) for k in range(6)] + 
                        [(spacing * (k + 0.5), spacing * 1.5) for k in range(5)])
    # 5-6
    arrangements.append([(spacing * k, 0.0) for k in range(5)] + 
                        [(spacing * (k + 0.5), spacing * 1.5) for k in range(6)])
    # 3-4-4
    arrangements.append([(spacing * k, 0.0) for k in range(3)] + 
                        [(spacing * (k + 0.5), spacing * 1.5) for k in range(4)] + 
                        [(spacing * k, spacing * 3.0) for k in range(4)])
    # 4-4-3
    arrangements.append([(spacing * k, 0.0) for k in range(4)] + 
                        [(spacing * (k + 0.5), spacing * 1.5) for k in range(4)] + 
                        [(spacing * k, spacing * 3.0) for k in range(3)])
    # 3-5-3
    arrangements.append([(spacing * k, 0.0) for k in range(3)] + 
                        [(spacing * (k + 0.5), spacing * 1.5) for k in range(5)] + 
                        [(spacing * k, spacing * 3.0) for k in range(3)])
    # 2-3-4-2
    arrangements.append([(spacing * k, 0.0) for k in range(2)] + 
                        [(spacing * (k + 0.5), spacing * 1.5) for k in range(3)] + 
                        [(spacing * k, spacing * 3.0) for k in range(4)] + 
                        [(spacing * (k + 0.5), spacing * 4.5) for k in range(2)])
    # 2-4-3-2
    arrangements.append([(spacing * k, 0.0) for k in range(2)] + 
                        [(spacing * (k + 0.5), spacing * 1.5) for k in range(4)] + 
                        [(spacing * k, spacing * 3.0) for k in range(3)] + 
                        [(spacing * (k + 0.5), spacing * 4.5) for k in range(2)])
    # 1-2-3-4-1
    arrangements.append([(0.0, 0.0)] + 
                        [(spacing * (k + 0.5), spacing * 1.5) for k in range(2)] + 
                        [(spacing * k, spacing * 3.0) for k in range(3)] + 
                        [(spacing * (k + 0.5), spacing * 4.5) for k in range(4)] + 
                        [(0.0, spacing * 6.0)])
    # 1-3-3-3-1
    arrangements.append([(0.0, 0.0)] + 
                        [(spacing * (k - 1), spacing * 1.5) for k in range(3)] + 
                        [(spacing * (k - 1), spacing * 3.0) for k in range(3)] + 
                        [(spacing * (k - 1), spacing * 4.5) for k in range(3)] + 
                        [(0.0, spacing * 6.0)])
    # 2-3-3-3
    arrangements.append([(spacing * k, 0.0) for k in range(2)] + 
                        [(spacing * (k + 0.5), spacing * 1.5) for k in range(3)] + 
                        [(spacing * k, spacing * 3.0) for k in range(3)] + 
                        [(spacing * (k + 0.5), spacing * 4.5) for k in range(3)])
    # 3-3-3-2
    arrangements.append([(spacing * k, 0.0) for k in range(3)] + 
                        [(spacing * (k + 0.5), spacing * 1.5) for k in range(3)] + 
                        [(spacing * k, spacing * 3.0) for k in range(3)] + 
                        [(spacing * (k + 0.5), spacing * 4.5) for k in range(2)])
    # Linear
    arrangements.append([(spacing * k, 0.0) for k in range(N)])
    
    for positions in arrangements:
        if time.time() > deadline:
            break
        for angle_inner in [0.0, 30.0]:
            inner_hexes = [(x, y, angle_inner) for x, y in positions]
            side, angle, center = compute_min_containing_hexagon(inner_hexes)
            if side < best_score:
                best_score = side
                best_result = (inner_hexes, center, side, angle)
    
    if best_result is None:
        inner = [[spacing * k, 0.0, 0.0] for k in range(N)]
        side, angle, center = compute_min_containing_hexagon(inner)
        return inner, center, side, angle
    
    return best_result
# EVOLVE-BLOCK-END
