"""Packing unit hexagons inside a regular hexagon using radial ring optimization."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_rad):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def _point_in_hexagon(px, py, cx, cy, side, angle_rad):
    """Check if point (px, py) is inside the hexagon centered at (cx, cy)."""
    # Transform to local coordinates
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    # For a regular hexagon with flat-top orientation (vertices at 0, 60, 120, ...),
    # but we defined vertices starting at 30 degrees offset.
    # Let's use the property: distance from center in hexagonal metric <= side
    # A point (lx, ly) is inside a regular hexagon of side s (with vertices at angles 30, 90, 150, 210, 270, 330)
    # if: |lx| <= s, |ly| <= s, |lx| + sqrt(3)*|ly| <= 2*s ... wait, that's for pointy-top.
    
    # Let's use a simpler check: the hexagon defined by _hex_vertices has vertices at:
    # angle_rad + 30, 90, 150, 210, 270, 330 degrees.
    # This is a "pointy-top" hexagon when angle_rad=0.
    # For a pointy-top hexagon of side s:
    # The conditions are:
    # |ly| <= s
    # |lx| <= s * sqrt(3)
    # |lx|/sqrt(3) + |ly| <= s
    
    # Actually, let me just use a robust point-in-polygon test with the vertices.
    vx, vy = _hex_vertices(cx, cy, side, angle_rad)
    # Use winding number or ray casting
    inside = False
    n = len(vx)
    j = n - 1
    for i in range(n):
        if ((vy[i] > py) != (vy[j] > py)) and (px < (vx[j] - vx[i]) * (py - vy[i]) / (vy[j] - vy[i]) + vx[i]):
            inside = not inside
        j = i
    return inside

def _hexagons_disjoint(hex1, hex2):
    """Check if two unit hexagons are disjoint (no overlapping interior).
    hex1, hex2: (cx, cy, angle_deg)
    We check if any vertex of one is inside the other, or if edges intersect.
    For simplicity and robustness, we check vertex-in-polygon both ways.
    If no vertices are inside, they might still overlap via edge crossings,
    but for convex shapes, if no vertices are inside the other, they are disjoint.
    """
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    
    # Check vertices of hex1 in hex2
    v1x, v1y = _hex_vertices(cx1, cy1, 1.0, np.radians(a1))
    for i in range(6):
        if _point_in_hexagon(v1x[i], v1y[i], cx2, cy2, 1.0, np.radians(a2)):
            return False
    
    # Check vertices of hex2 in hex1
    v2x, v2y = _hex_vertices(cx2, cy2, 1.0, np.radians(a2))
    for i in range(6):
        if _point_in_hexagon(v2x[i], v2y[i], cx1, cy1, 1.0, np.radians(a1)):
            return False
    
    return True

def _outer_hexagon_contains_inner(inner_hex, outer_center, outer_side, outer_angle_deg):
    """Check if inner hexagon is fully inside the outer hexagon."""
    cx, cy, a_deg = inner_hex
    a_rad = np.radians(a_deg)
    ox, oy = outer_center
    os = outer_side
    oa_rad = np.radians(outer_angle_deg)
    
    # Check all vertices of inner hexagon are inside outer hexagon
    vx, vy = _hex_vertices(cx, cy, 1.0, a_rad)
    for i in range(6):
        if not _point_in_hexagon(vx[i], vy[i], ox, oy, os, oa_rad):
            return False
    return True

def _compute_outer_side(inner_positions, outer_center=(0.0, 0.0), outer_angle_deg=0.0):
    """Compute the minimum outer hexagon side length that contains all inner hexagons.
    inner_positions: list of (x, y, angle_deg)
    Returns the minimal side length.
    """
    ox, oy = outer_center
    oa_rad = np.radians(outer_angle_deg)
    
    # For a regular hexagon, the distance from center to any point inside is bounded by
    # the support function. The minimal enclosing hexagon (with fixed orientation) for a set of points
    # has side length = max over all points of (distance in hexagonal norm).
    
    # Actually, let's think differently. The outer hexagon has 6 sides. Each side is a line.
    # For the hexagon to contain all points, each point must be on the correct side of each edge.
    # The side length s determines the distance from center to each edge: d = s * sqrt(3) / 2.
    # The 6 edge normals are at angles: oa_rad + 0, 60, 120, 180, 240, 300 degrees (for flat-top)
    # or oa_rad + 30, 90, ... (for pointy-top).
    
    # Our _hex_vertices uses angles: angle_rad + 30, 90, 150, 210, 270, 330.
    # So the edges are perpendicular to directions: 0, 60, 120, 180, 240, 300 (offset by 90 from vertices? No.)
    # Vertex at 30 deg means edge normal is at 0 deg? Let's verify.
    # Vertices at 30, 90, 150, 210, 270, 330.
    # Edge between vertex at 30 and 90: midpoint at 60, normal at 60.
    # Edge between 90 and 150: midpoint at 120, normal at 120.
    # So normals are at 60, 120, 180, 240, 300, 0 degrees (i.e., 0, 60, 120, 180, 240, 300).
    
    # Distance from center to each edge = s * cos(30) = s * sqrt(3)/2.
    
    # For a point p, the condition to be inside is:
    # For each normal direction n_i (unit vector), dot(p - center, n_i) <= s * sqrt(3)/2
    
    normals = []
    for k in range(6):
        angle = np.radians(outer_angle_deg) + k * np.pi / 3
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    
    max_val = 0.0
    for (px, py, _) in inner_positions:
        for n in normals:
            val = (px - ox) * n[0] + (py - oy) * n[1]
            if val > max_val:
                max_val = val
        # Also check negative directions? No, the 6 normals cover all 6 sides.
        # But wait, we need to check both positive and negative? 
        # The 6 normals at 0, 60, 120, 180, 240, 300 already cover all directions.
    
    # The required distance from center to each edge is max_val.
    # So s * sqrt(3)/2 >= max_val => s >= max_val * 2 / sqrt(3)
    if max_val < 0:
        return 0.0
    return max_val * 2.0 / np.sqrt(3.0)

def construct():
    # Initialize inner hexagons in a hexagonal lattice pattern
    # For N=11, we can try a centered hexagonal arrangement
    # Ring 0: 1 hexagon at center
    # Ring 1: 6 hexagons
    # Ring 2: 12 hexagons
    # Total: 1 + 6 + 12 = 19 > 11, so we need to prune
    
    # Let's start with a simple linear arrangement or a small hexagonal cluster
    # Hexagonal cluster of radius r has 1 + 3*r*(r+1) hexagons
    # r=0: 1, r=1: 7, r=2: 19
    # So for 11, we take r=1 (7 hexagons) and add 4 more
    
    # Initial positions: hexagonal lattice with spacing ~2 (to avoid overlap)
    # Hexagonal lattice basis:
    # e1 = (2, 0), e2 = (1, sqrt(3))
    # Actually for unit hexagons, the minimum distance between centers is 2*sqrt(3) for adjacent in same row?
    # No, for pointy-top hexagons, adjacent centers are 2*sqrt(3)/2 = sqrt(3) apart? 
    # Let me reconsider. Two unit hexagons (side=1) touching: center distance = 2 * (apothem) = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732?
    # No, apothem = sqrt(3)/2 * side = sqrt(3)/2. Two touching hexagons: center distance = 2 * apothem = sqrt(3).
    # But they also need to be rotated appropriately. For simplicity, let's use a spacing of 2.0 to be safe.
    
    # Let's use a grid-like initial arrangement
    inner = []
    # Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # 6 neighbors at distance ~1.8 (slightly more than sqrt(3) to avoid overlap)
    for k in range(6):
        angle = k * np.pi / 3
        x = 1.8 * np.cos(angle)
        y = 1.8 * np.sin(angle)
        inner.append([x, y, 0.0])
    
    # 4 more hexagons further out
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 4
        x = 3.5 * np.cos(angle)
        y = 3.5 * np.sin(angle)
        inner.append([x, y, 0.0])
    
    # Optimize positions to minimize outer hexagon side length
    # Parameters: x, y for each hexagon (angles fixed at 0 for simplicity)
    def objective(params):
        positions = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i + 1]
            positions.append((x, y, 0.0))
        
        # Check disjointness - if any overlap, return large penalty
        for i in range(N):
            for j in range(i+1, N):
                if not _hexagons_disjoint(positions[i], positions[j]):
                    return 1e6
        
        # Compute outer side length
        side = _compute_outer_side(positions)
        return side
    
    # Initial parameters
    x0 = np.zeros(2 * N)
    for i in range(N):
        x0[2*i] = inner[i][0]
        x0[2*i + 1] = inner[i][1]
    
    # Use Nelder-Mead for robustness
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        if result.success or result.fun < 1e5:
            best_params = result.x
        else:
            best_params = x0
    except:
        best_params = x0
    
    # Extract final positions
    final_inner = []
    for i in range(N):
        x = best_params[2*i]
        y = best_params[2*i + 1]
        final_inner.append([float(x), float(y), 0.0])
    
    # Compute outer hexagon
    outer_side = _compute_outer_side(final_inner)
    
    # Ensure minimum side length
    if outer_side < 1.0:
        outer_side = 1.0
    
    return final_inner, (0.0, 0.0), float(outer_side), 0.0
# EVOLVE-BLOCK-END
