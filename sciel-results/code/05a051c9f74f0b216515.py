import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side length, rotation."""
    angle = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    # Vertex at angle 0 is at (side, 0) from center (pointy-top orientation)
    # For flat-top, vertices are at 30, 90, 150, 210, 270, 330
    # Let's use: vertices at angle + k*60 for k=0..5, at distance 'side' from center
    angles = angle + np.linspace(0, 2*np.pi, 6, endpoint=False)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def polygon_min_distance(v1, v2):
    """Compute minimum distance between two convex polygons (v1, v2 are Nx2 arrays).
    Returns 0 if they intersect or touch, positive otherwise.
    Uses separating axis theorem for intersection, then vertex-edge distance."""
    # Check if polygons intersect using SAT
    def get_axes(poly):
        n = len(poly)
        axes = []
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1) % n]
            edge = p2 - p1
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                axes.append(normal / norm)
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    
    for axis in axes:
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 or max2 < min1:
            return max(min2 - max1, max2 - min1)  # Gap found
    
    # Polygons intersect or one contains the other
    return 0.0

def point_in_hexagon(p, center, side, angle_deg):
    """Check if point p is inside a regular hexagon."""
    verts = hexagon_vertices(center[0], center[1], side, angle_deg)
    # Use cross product method
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i+1) % n]
        edge = b - a
        ap = p - a
        cross = edge[0] * ap[1] - edge[1] * ap[0]
        if cross < -1e-10:
            return False
    return True

def min_outer_side(inner_hexes, outer_center, outer_angle):
    """Compute minimum outer hexagon side length to contain all inner hexagons.
    inner_hexes: list of (cx, cy, angle_deg) for unit hexagons
    outer_center: (cx, cy)
    outer_angle: rotation of outer hexagon in degrees
    """
    # For each inner hexagon, get its vertices
    all_verts = []
    for (cx, cy, ang) in inner_hexes:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Outer hexagon normals: 6 directions
    outer_angle_rad = np.radians(outer_angle)
    normals = []
    for k in range(6):
        ang = outer_angle_rad + k * np.pi / 3 + np.pi / 6  # Face normals are offset by 30 degrees from vertices
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)  # 6x2
    
    # For containment: for each normal, the max projection of (p - center) must be <= apothem = side * sqrt(3)/2
    # So side >= (2/sqrt(3)) * max_projection
    rel = all_verts - np.array(outer_center)
    projections = rel @ normals.T  # (n_verts, 6)
    max_proj = np.abs(projections).max(axis=0)  # Max in each of 6 directions
    max_overall = max_proj.max()
    
    return (2.0 / np.sqrt(3.0)) * max_overall

def compute_objective(params, outer_center, outer_angle):
    """Compute objective: outer side length + penalty for overlaps.
    params: flat array of length 3*N: [x1, y1, a1, x2, y2, a2, ...]
    """
    inner_hexes = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_hexes.append((x, y, a))
    
    # Compute minimum outer side
    side = min_outer_side(inner_hexes, outer_center, outer_angle)
    
    # Penalty for overlaps
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            v1 = hexagon_vertices(*inner_hexes[i])
            v2 = hexagon_vertices(*inner_hexes[j])
            dist = polygon_min_distance(v1, v2)
            if dist < 0.01:
                overlap_penalty += max(0, 0.01 - dist) ** 2
    
    return side + 100.0 * overlap_penalty

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial arrangement: honeycomb-like pattern
    # Place hexagons in a hexagonal grid
    # Center at origin, then rings around
    # For 12 hexagons: 1 center + 6 in first ring + 5 in second ring (partial)
    # Or: 4 rows of 3, 4, 3, 1? Let's do a more symmetric arrangement.
    
    # Honeycomb positions for unit hexagons (flat-top orientation, angle=0)
    # Spacing: horizontal = 1.5, vertical = sqrt(3)/2 for adjacent hexagons
    # Actually for pointy-top (angle=0 in our definition), the spacing is different.
    
    # Let's use a simple grid initialization
    # Hexagons in a 3x4 grid with hexagonal offsets
    positions = []
    dx = 1.5  # horizontal spacing
    dy = np.sqrt(3) / 2  # vertical spacing
    
    # Row 0: 4 hexagons
    for k in range(4):
        positions.append((k * dx, 0, 0.0))
    # Row 1: 4 hexagons, offset
    for k in range(4):
        positions.append((k * dx + dx/2, dy, 0.0))
    # Row 2: 4 hexagons
    for k in range(4):
        positions.append((k * dx, 2*dy, 0.0))
    
    # That's 12 positions
    # Center them
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(x - cx, y - cy, a) for (x, y, a) in positions]
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Initial params
    x0 = []
    for (x, y, a) in positions:
        x0.extend([x, y, a])
    
    # Run optimization with time guard
    best_params = np.array(x0)
    best_obj = compute_objective(best_params, outer_center, outer_angle)
    
    # Try a few optimization attempts
    n_attempts = 0
    while time.time() < deadline - 5:
        n_attempts += 1
        try:
            # Use Nelder-Mead for robustness
            result = minimize(
                compute_objective,
                best_params,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 5:
            break
        
        # Try a perturbation for next attempt
        if n_attempts < 5:
            perturbation = np.random.normal(0, 0.1, size=len(x0))
            try_start = best_params + perturbation
            try_obj = compute_objective(try_start, outer_center, outer_angle)
            if try_obj < best_obj:
                best_params = try_start
                best_obj = try_obj
    
    # Extract final positions
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
    
    # Compute final outer side length
    outer_side = min_outer_side(inner, outer_center, outer_angle)
    
    return inner, outer_center, outer_side, outer_angle
