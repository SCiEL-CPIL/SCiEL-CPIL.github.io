"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(cx, cy, side, angle_deg, px, py):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side and rotation."""
    angle_rad = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    apothem = side * np.sqrt(3) / 2
    normals = np.column_stack([np.cos(np.arange(6) * np.pi / 3), np.sin(np.arange(6) * np.pi / 3)])
    dots = np.abs(normals @ np.array([lx, ly]))
    return np.all(dots <= apothem + 1e-9)

def hexagons_disjoint(pos1, ang1, pos2, ang2, tol=1e-6):
    """Check if two unit hexagons are disjoint."""
    verts1 = hexagon_vertices(pos1[0], pos1[1], 1.0, ang1)
    verts2 = hexagon_vertices(pos2[0], pos2[1], 1.0, ang2)
    for v in verts1:
        if point_in_hexagon(pos2[0], pos2[1], 1.0, ang2, v[0], v[1]):
            return False
    for v in verts2:
        if point_in_hexagon(pos1[0], pos1[1], 1.0, ang1, v[0], v[1]):
            return False
    return True

def enclosing_hexagon_side(points, angle_deg):
    """Compute side length of enclosing hexagon with given rotation containing all points."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)
    lx = points[:, 0] * cos_a - points[:, 1] * sin_a
    ly = points[:, 0] * sin_a + points[:, 1] * cos_a
    normals = np.column_stack([np.cos(np.arange(6) * np.pi / 3), np.sin(np.arange(6) * np.pi / 3)])
    dots = np.abs(np.column_stack([lx, ly]) @ normals.T)
    max_dot = np.max(dots)
    return 2 * max_dot / np.sqrt(3)

def minimal_enclosing_hexagon(points):
    """Find minimal side length and optimal rotation for enclosing hexagon."""
    best_side = np.inf
    best_angle = 0
    for angle in np.linspace(0, 60, 121):
        s = enclosing_hexagon_side(points, angle)
        if s < best_side:
            best_side = s
            best_angle = angle
    return best_side, best_angle

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Generate hexagonal lattice positions with sufficient spacing
    # For axis-aligned unit hexagons, center-to-center distance of sqrt(3) ensures disjointness
    # Use a triangular/hexagonal grid with spacing 2.0 to be safe
    spacing = 2.0
    positions = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = spacing * 1.5 * q
            y = spacing * np.sqrt(3) / 2 * q + spacing * np.sqrt(3) * r
            positions.append((x, y))
    
    # Sort by distance from origin
    positions_with_dist = [(p[0], p[1], np.sqrt(p[0]**2 + p[1]**2)) for p in positions]
    positions_with_dist.sort(key=lambda x: x[2])
    
    # Select N closest positions
    selected = positions_with_dist[:N]
    selected_points = np.array([[p[0], p[1]] for p in selected])
    
    # Compute enclosing hexagon
    side, angle = minimal_enclosing_hexagon(selected_points)
    
    inner = [[p[0], p[1], 0.0] for p in selected]
    
    # Verify disjointness
    def check_disjoint(arrangement):
        for i in range(len(arrangement)):
            for j in range(i+1, len(arrangement)):
                if not hexagons_disjoint((arrangement[i][0], arrangement[i][1]), arrangement[i][2],
                                         (arrangement[j][0], arrangement[j][1]), arrangement[j][2]):
                    return False
        return True
    
    if not check_disjoint(inner):
        # Increase spacing and retry
        spacing = 2.5
        positions = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = spacing * 1.5 * q
                y = spacing * np.sqrt(3) / 2 * q + spacing * np.sqrt(3) * r
                positions.append((x, y))
        positions_with_dist = [(p[0], p[1], np.sqrt(p[0]**2 + p[1]**2)) for p in positions]
        positions_with_dist.sort(key=lambda x: x[2])
        selected = positions_with_dist[:N]
        selected_points = np.array([[p[0], p[1]] for p in selected])
        side, angle = minimal_enclosing_hexagon(selected_points)
        inner = [[p[0], p[1], 0.0] for p in selected]
    
    # Check time
    if time.time() > deadline - 5.0:
        return inner, (0.0, 0.0), side, angle
    
    # Try to improve by optimizing the outer hexagon rotation more finely
    best_side = side
    best_angle = angle
    for angle in np.linspace(0, 60, 241):
        s = enclosing_hexagon_side(selected_points, angle)
        if s < best_side:
            best_side = s
            best_angle = angle
    
    return inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
