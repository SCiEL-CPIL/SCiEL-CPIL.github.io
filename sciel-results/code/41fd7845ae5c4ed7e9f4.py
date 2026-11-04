"""Packing 12 unit regular hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 130.0  # seconds, safety margin under 150s

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    x = side * np.cos(angles)
    y = side * np.sin(angles)
    # Rotate
    rx = x * cos_t - y * sin_t
    ry = x * sin_t + y * cos_t
    return np.column_stack([rx + cx, ry + cy])

def min_dist_hexagons(v1, v2):
    """Minimum distance between vertices of two convex polygons (negative if overlapping)."""
    # For convex polygons, min distance is min over all pairs of (vertex-edge distance)
    # But for simplicity and speed, we use vertex-to-vertex and check if any vertex is inside
    # Actually, let's use a more robust approach: minimum distance between two convex polygons
    # We'll compute the minimum distance between all pairs of edges and vertices
    
    # Simple approach: min over all vertex pairs of distance, but this only works if they don't overlap
    # Better: check if polygons overlap using SAT (Separating Axis Theorem)
    
    # For two convex polygons, they are separated if there exists a line that separates them
    # The candidate separating lines are perpendicular to each edge of either polygon
    
    def project(points, axis):
        return np.dot(points, axis)
    
    def get_axes(poly):
        axes = []
        n = len(poly)
        for i in range(n):
            edge = poly[(i+1) % n] - poly[i]
            # Perpendicular to edge
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm > 1e-12:
                axis = axis / norm
            axes.append(axis)
        return axes
    
    axes1 = get_axes(v1)
    axes2 = get_axes(v2)
    all_axes = axes1 + axes2
    
    for axis in all_axes:
        proj1 = project(v1, axis)
        proj2 = project(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            # Separated: distance is the gap
            if max1 < min2:
                return min2 - max1
            else:
                return min1 - max2
    
    # If no separating axis found, polygons overlap
    # Return negative of the maximum penetration depth (approximate)
    # For now, return a small negative value
    return -0.01

def outer_side_length(inner_hexagons, center, outer_angle_deg):
    """
    Compute the minimum side length of a regular hexagon centered at 'center' 
    with rotation 'outer_angle_deg' that contains all inner hexagons.
    
    A regular hexagon can be defined by 6 half-planes. The side length s determines
    the distance from center to each edge, which is s * sqrt(3) / 2.
    The hexagon edges are at angles outer_angle + 0, 60, 120, 180, 240, 300 degrees.
    
    For a regular hexagon with side s, the apothem (distance from center to edge) is a = s * sqrt(3)/2.
    So s = 2 * a / sqrt(3).
    
    For each edge direction, we need the maximum projection of all inner vertices onto that direction
    (relative to center) to determine the required apothem.
    """
    all_vertices = np.vstack([vh for vh in inner_hexagons])
    
    # Directions perpendicular to each edge of the outer hexagon
    # Edge i is perpendicular to direction at angle (outer_angle + 60*i) degrees
    base_angle = np.deg2rad(outer_angle_deg)
    edge_angles = base_angle + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    
    max_apothem = 0.0
    for ea in edge_angles:
        # Normal vector pointing outward for this edge
        normal = np.array([np.cos(ea), np.sin(ea)])
        # Project all vertices relative to center
        rel = all_vertices - np.array(center)
        projections = np.dot(rel, normal)
        # The required apothem in this direction is the max projection
        a = np.max(projections)
        if a > max_apothem:
            max_apothem = a
    
    # Also need to handle the case where vertices extend in the opposite direction
    # Actually, for a convex hexagon centered at 'center', the constraint is:
    # For each edge with outward normal n_i, we need n_i . (v - center) <= apothem for all vertices v
    # So apothem >= max over all vertices of n_i . (v - center)
    # But we also need to consider that the hexagon extends in both directions
    # Wait, a regular hexagon centered at origin with apothem a is the intersection of:
    # n_i . x <= a for i = 0..5, where n_i are the 6 outward normals
    # This automatically handles both sides since the 6 normals cover all directions
    
    side = 2.0 * max_apothem / np.sqrt(3.0)
    return max(side, 0.0)

def pack_cost(params, initial_config=None):
    """
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx, cy, cangle]
    where ai are angles in degrees, cangle is outer hexagon rotation in degrees
    Returns: outer side length + penalty for overlaps
    """
    cx = params[0:N]
    cy = params[N:2*N]
    angles = params[2*N:3*N]
    outer_cx = params[3*N]
    outer_cy = params[3*N+1]
    outer_angle = params[3*N+2]
    
    # Compute inner hexagon vertices
    inner_hexagons = []
    for i in range(N):
        v = hex_vertices(cx[i], cy[i], 1.0, angles[i])
        inner_hexagons.append(v)
    
    # Compute outer side length
    side = outer_side_length(inner_hexagons, (outer_cx, outer_cy), outer_angle)
    
    # Check overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = min_dist_hexagons(inner_hexagons[i], inner_hexagons[j])
            if d < 0:
                penalty += (-d) ** 2 * 100.0
    
    return side + penalty

def construct():
    start_time = time.time()
    
    # Initial configuration: hexagonal lattice arrangement
    # Place hexagons in a honeycomb pattern
    # For 12 hexagons, a reasonable initial layout:
    # Row 1: 3 hexagons
    # Row 2: 4 hexagons (offset)
    # Row 3: 3 hexagons
    # Row 4: 2 hexagons (offset)
    # Or simpler: just place them in a grid-like pattern
    
    # Let's use a centered hexagonal arrangement
    positions = []
    angles_init = []
    
    # Place 12 hexagons in a rough hexagonal pattern
    # Center + 5 around it + 6 more around those
    # Or use a simple grid
    
    # Simple approach: place in 2 rows of 6, with offset
    # Actually, let's use multiple random initializations and pick the best
    
    np.random.seed(42)
    
    best_params = None
    best_cost = float('inf')
    
    # Try several initial configurations
    for init_idx in range(5):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        if init_idx == 0:
            # Hexagonal lattice: center + 6 around + 5 more
            # Positions in terms of hex grid coordinates
            # Hex grid: axial coordinates (q, r), distance = sqrt(3)/2 * (q^2 + qr + r^2) for side 1
            # Actually for unit hexagons packed, center-to-center distance is sqrt(3) for adjacent
            hex_positions = [
                (0, 0),
                (np.sqrt(3), 0),
                (np.sqrt(3)/2, 1.5),
                (-np.sqrt(3)/2, 1.5),
                (-np.sqrt(3), 0),
                (-np.sqrt(3)/2, -1.5),
                (np.sqrt(3)/2, -1.5),
                # Second ring
                (2*np.sqrt(3), 0),
                (np.sqrt(3), 1.5*3),
                (-np.sqrt(3), 1.5*3),
                (-2*np.sqrt(3), 0),
                (-np.sqrt(3), -1.5*3),
                (np.sqrt(3), -1.5*3),
            ][:N]
            cx = [p[0] for p in hex_positions]
            cy = [p[1] for p in hex_positions]
            angles_init = [0.0] * N
        elif init_idx == 1:
            # Random positions in a disk
            radius = 4.0
            cx = np.random.uniform(-radius, radius, N)
            cy = np.random.uniform(-radius, radius, N)
            angles_init = np.random.uniform(0, 60, N)
        elif init_idx == 2:
            # Linear arrangement (like seed but better)
            cx = np.arange(N) * np.sqrt(3) - (N-1)*np.sqrt(3)/2
            cy = np.zeros(N)
            angles_init = np.zeros(N)
        elif init_idx == 3:
            # Two rows
            cx = []
            cy = []
            for k in range(6):
                cx.append(k * np.sqrt(3) - 2.5*np.sqrt(3))
                cy.append(0.0)
            for k in range(6):
                cx.append(k * np.sqrt(3) - 2.5*np.sqrt(3) + np.sqrt(3)/2)
                cy.append(1.5)
            angles_init = [0.0] * N
        else:
            # Another random
            radius = 3.0
            cx = np.random.uniform(-radius, radius, N)
            cy = np.random.uniform(-radius, radius, N)
            angles_init = np.random.uniform(0, 60, N)
        
        params0 = np.array([
            cx, cy, angles_init, 0.0, 0.0, 0.0
        ]).flatten()
        
        # Optimize
        try:
            result = minimize(
                pack_cost,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            cost = result.fun
            if cost < best_cost:
                best_cost = cost
                best_params = result.x
        except Exception:
            pass
        
        if time.time() - start_time > DEADLINE - 15:
            break
    
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    cx = best_params[0:N]
    cy = best_params[N:2*N]
    angles = best_params[2*N:3*N]
    outer_cx = best_params[3*N]
    outer_cy = best_params[3*N+1]
    outer_angle = best_params[3*N+2]
    
    # Final polish: try to tighten the outer hexagon
    # Recompute the optimal outer hexagon given the inner positions
    inner_hexagons = []
    for i in range(N):
        v = hex_vertices(cx[i], cy[i], 1.0, angles[i])
        inner_hexagons.append(v)
    
    # Try different outer angles to find minimum side
    best_outer_side = float('inf')
    best_outer_angle = 0.0
    best_center = (outer_cx, outer_cy)
    
    for oa in np.linspace(0, 60, 12):
        for ocx in [outer_cx, 0.0]:
            for ocy in [outer_cy, 0.0]:
                s = outer_side_length(inner_hexagons, (ocx, ocy), oa)
                if s < best_outer_side:
                    best_outer_side = s
                    best_outer_angle = oa
                    best_center = (ocx, ocy)
    
    inner = [[float(cx[i]), float(cy[i]), float(angles[i])] for i in range(N)]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_outer_side), float(best_outer_angle)
