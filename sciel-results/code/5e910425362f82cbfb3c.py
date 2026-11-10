# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _edge_normals(vertices):
    """Return outward normals of each edge of a polygon given in CCW order."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # outward normal for CCW polygon: rotate edge by -90 deg (clockwise)
        # edge = (dx, dy), outward normal = (dy, -dx) normalized
        norm = np.array([edge[1], -edge[0]])
        norm_len = np.linalg.norm(norm)
        if norm_len > 1e-12:
            normals[i] = norm / norm_len
    return normals


def _separation_distance(v1, n1, v2, n2):
    """
    Compute the minimum separation between two convex polygons using SAT.
    v1, n1: vertices and outward normals of polygon 1
    v2, n2: vertices and outward normals of polygon 2
    Returns the minimum gap (positive if separated, negative if overlapping).
    """
    min_gap = np.inf
    # Test axes from polygon 1
    for i in range(len(n1)):
        axis = n1[i]
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        gap = max(min1, min2) - min(max1, max2)
        if gap < min_gap:
            min_gap = gap
    # Test axes from polygon 2
    for i in range(len(n2)):
        axis = n2[i]
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        gap = max(min1, min2) - min(max1, max2)
        if gap < min_gap:
            min_gap = gap
    return min_gap


def _outer_constraint_penalty(inner_centers, inner_angles, outer_center, outer_angle, s):
    """
    Compute penalty for inner hexagons being outside the outer hexagon.
    """
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
    # Ensure CCW order
    area = 0
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        area += p1[0] * p2[1] - p2[0] * p1[1]
    if area < 0:
        outer_verts = outer_verts[::-1]
    
    penalty = 0.0
    outer_normals = _edge_normals(outer_verts)
    
    for i in range(12):
        cx = inner_centers[i, 0]
        cy = inner_centers[i, 1]
        ang = inner_angles[i]
        inner_verts = _hex_vertices(cx, cy, 1.0, ang)
        
        for v in inner_verts:
            # Check if v is inside outer hexagon
            # For each edge, check cross product
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                edge = p2 - p1
                vec = v - p1
                cross = edge[0] * vec[1] - edge[1] * vec[0]
                # For CCW polygon, interior is to the left, so cross should be >= 0
                if cross < 0:
                    # Distance from point to edge line
                    edge_len = np.linalg.norm(edge)
                    if edge_len > 1e-12:
                        dist = -cross / edge_len
                        penalty += dist ** 2
    return penalty


def _overlap_penalty(inner_centers, inner_angles):
    """
    Compute penalty for overlapping inner hexagons.
    """
    penalty = 0.0
    for i in range(12):
        v1 = _hex_vertices(inner_centers[i, 0], inner_centers[i, 1], 1.0, inner_angles[i])
        n1 = _edge_normals(v1)
        for j in range(i + 1, 12):
            v2 = _hex_vertices(inner_centers[j, 0], inner_centers[j, 1], 1.0, inner_angles[j])
            n2 = _edge_normals(v2)
            gap = _separation_distance(v1, n1, v2, n2)
            if gap < 0:
                penalty += (-gap) ** 2
    return penalty


def _objective(params):
    """
    Objective function to minimize.
    params: [cx1, cy1, ax1, ..., cx12, cy12, ax12, ox, oy, oa, s]
    """
    n = 12
    inner_centers = params[:3 * n].reshape(n, 3)
    outer_center = params[3 * n:3 * n + 2]
    outer_angle = params[3 * n + 2]
    s = params[3 * n + 3]
    
    # Ensure s is positive
    if s < 1:
        s = 1.0
    
    # Overlap penalty
    pen_overlap = _overlap_penalty(inner_centers, inner_centers[:, 2])
    
    # Outer constraint penalty
    pen_outer = _outer_constraint_penalty(inner_centers, inner_centers[:, 2], outer_center, outer_angle, s)
    
    # Total cost: minimize s + large penalty for violations
    cost = s + 100.0 * pen_overlap + 100.0 * pen_outer
    return cost


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    using physics-based relaxation to minimize the outer hexagon side length.
    """
    n = 12
    
    # Initial guess: hexagonal lattice arrangement
    # Unit hexagons in a honeycomb pattern
    # Row 0: 4 hexagons at y=0, x = -3, -1, 1, 3
    # Row 1: 3 hexagons at y=sqrt(3)*1.5, x = -2, 0, 2
    # Row 2: 4 hexagons at y=2*sqrt(3)*1.5, x = -3, -1, 1, 3
    # Wait, that's 11. Let me think of a better arrangement.
    
    # Actually, let's use a simple grid:
    # 4 rows of 3 hexagons each, but hexagons don't tile in a square grid.
    # Let's use the honeycomb arrangement:
    # Row 0 (y=0): x = -3, -1, 1, 3 (4 hexagons)
    # Row 1 (y=3*sqrt(3)/2 ≈ 2.598): x = -2, 0, 2 (3 hexagons)
    # Row 2 (y=3*sqrt(3)): x = -3, -1, 1, 3 (4 hexagons)
    # Row 3 (y=9*sqrt(3)/2): x = -2, 0, 2 (3 hexagons) -- that's 14, too many.
    
    # Let's just use a 3x4 grid with appropriate spacing:
    # Hexagon spacing: horizontal = 3, vertical = 3*sqrt(3)/2 ≈ 2.598
    # Offset alternating rows by 1.5
    
    sqrt3 = np.sqrt(3)
    initial_centers = []
    for row in range(4):
        y = row * 1.5 * sqrt3
        if row % 2 == 0:
            x_offsets = [-3, -1, 1, 3]
        else:
            x_offsets = [-2, 0, 2]
        for x in x_offsets:
            initial_centers.append([x, y, 0.0])
    
    # That's 4 + 3 + 4 + 3 = 14, too many. Let me adjust.
    # Use 3 rows:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # Total: 12
    
    initial_centers = []
    for row in range(3):
        y = row * 1.5 * sqrt3
        if row % 2 == 0:
            x_offsets = [-3, -1, 1, 3]
        else:
            x_offsets = [-4, -2, 0, 2]  # Shifted to make 4
        for x in x_offsets:
            initial_centers.append([x, y, 0.0])
    
    # Center the arrangement
    cx_mean = np.mean([c[0] for c in initial_centers])
    cy_mean = np.mean([c[1] for c in initial_centers])
    for c in initial_centers:
        c[0] -= cx_mean
        c[1] -= cy_mean
    
    inner_centers = np.array(initial_centers)
    
    # Initial outer hexagon: large enough
    max_dist = np.max(np.sqrt(inner_centers[:, 0] ** 2 + inner_centers[:, 1] ** 2)) + 2
    s_init = max_dist * 2
    
    # Initial params: [cx1, cy1, ax1, ..., cx12, cy12, ax12, ox, oy, oa, s]
    x0 = np.zeros(3 * n + 4)
    for i in range(n):
        x0[3 * i] = inner_centers[i, 0]
        x0[3 * i + 1] = inner_centers[i, 1]
        x0[3 * i + 2] = inner_centers[i, 2]
    x0[3 * n] = 0.0  # outer center x
    x0[3 * n + 1] = 0.0  # outer center y
    x0[3 * n + 2] = 0.0  # outer angle
    x0[3 * n + 3] = s_init
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((-50, 50))  # cx
        bounds.append((-50, 50))  # cy
        bounds.append((0, 360))  # angle
    bounds.append((-50, 50))  # outer cx
    bounds.append((-50, 50))  # outer cy
    bounds.append((0, 360))  # outer angle
    bounds.append((np.sqrt(12), 100))  # s
    
    # Optimize
    result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
    
    # Extract optimal params
    opt_params = result.x
    inner_centers_opt = opt_params[:3 * n].reshape(n, 3)
    outer_center = opt_params[3 * n:3 * n + 2]
    outer_angle = opt_params[3 * n + 2]
    s_opt = opt_params[3 * n + 3]
    
    # Ensure s is at least sqrt(12)
    s_opt = max(s_opt, np.sqrt(12))
    
    # Normalize angles to [0, 360)
    for i in range(n):
        inner_centers_opt[i, 2] = inner_centers_opt[i, 2] % 360
    outer_angle = outer_angle % 360
    
    return inner_centers_opt, np.array([outer_center[0], outer_center[1], outer_angle]), s_opt


# EVOLVE-BLOCK-END
