"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    theta = np.deg2rad(angle_deg)
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def sat_test(vertices_a, vertices_b, eps=1e-10):
    """Return True if two convex polygons (arrays of (x,y)) overlap (interior intersection)."""
    # Separating Axis Theorem
    for verts in (vertices_a, vertices_b):
        for i in range(len(verts)):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % len(verts)]
            # Normal to edge
            dx = x2 - x1
            dy = y2 - y1
            # Normal (perpendicular)
            nx, ny = -dy, dx
            # Project both polygons onto this normal
            proj_a = np.dot(vertices_a, [nx, ny])
            proj_b = np.dot(vertices_b, [nx, ny])
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            if max_a < min_b - eps or max_b < min_a - eps:
                return False  # Separated
    return True  # No separating axis found -> overlap

def fits_in_outer(inner_hexes, outer_center, outer_side, outer_angle, eps=1e-9):
    """Check if all inner hexagons are inside the outer hexagon."""
    outer_verts = hex_vertices(*outer_center, outer_side, outer_angle)
    for (ix, iy, iangle) in inner_hexes:
        inner_verts = hex_vertices(ix, iy, 1.0, iangle)
        # Check all inner vertices are inside outer
        for (vx, vy) in inner_verts:
            # Point in convex polygon test using cross products
            inside = True
            for j in range(6):
                x1, y1 = outer_verts[j]
                x2, y2 = outer_verts[(j + 1) % 6]
                # Cross product to check if point is on the inside of edge
                cross = (x2 - x1) * (vy - y1) - (y2 - y1) * (vx - x1)
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def disjoint_check(inner_hexes, eps=1e-9):
    """Check if all inner hexagons are pairwise disjoint."""
    n = len(inner_hexes)
    verts_list = []
    for (ix, iy, iangle) in inner_hexes:
        verts_list.append(hex_vertices(ix, iy, 1.0, iangle))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_test(verts_list[i], verts_list[j], eps):
                return False
    return True

def evaluate_config(inner, center, side, angle):
    """Return side length if valid, else None."""
    if not fits_in_outer(inner, center, side, angle):
        return None
    if not disjoint_check(inner):
        return None
    return side

# EVOLVE-BLOCK-START
def construct():
    # Try a few simple configurations and pick the best valid one
    best = (None, 40.0)  # (inner, side)
    
    # Configuration 1: All in a row along x-axis, centered at origin
    # Inner hexagons at positions (-5,0), (-4,0), ..., (5,0) with angle 0
    # The outer hexagon needs to enclose all these.
    # For a row of N unit hexagons along x, the bounding box is roughly 2*N wide.
    # A regular hexagon enclosing this needs side length ~ N or so.
    # Let's try to optimize the outer side length for this configuration.
    
    # Actually, let's try a few different outer side lengths and check validity.
    # For the row configuration:
    inner_row = [[float(k), 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    
    # Try different outer side lengths
    for side in np.linspace(5.0, 15.0, 100):
        config = (inner_row, (0.0, 0.0), side, 0.0)
        result = evaluate_config(*config)
        if result is not None:
            if best[0] is None or side < best[1]:
                best = (inner_row, side)
    
    # If we found a valid config, return it
    if best[0] is not None:
        return best[0], (0.0, 0.0), best[1], 0.0
    
    # Fallback: return the original configuration (should be valid for large side)
    return inner_row, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
