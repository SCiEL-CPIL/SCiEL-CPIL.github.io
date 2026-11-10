# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(vertices):
    """Return unit edge normals for a hexagon (6, 2) array."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    # inward normals: rotate edge by -90 degrees (for CCW vertices)
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    # ensure they point inward: dot with (center - edge_midpoint) should be positive
    center = vertices.mean(axis=0)
    mids = (vertices + np.roll(vertices, -1, axis=0)) / 2
    dots = np.sum(normals * (center - mids), axis=1)
    normals[dots < 0] *= -1
    return normals

def _sat_overlap_penalty(v1, v2, normals1, normals2, eps=1e-8):
    """
    Compute separation along all 12 axes. Returns 0 if separated,
    penalty = (overlap_amount)^2 for each axis where they overlap.
    """
    all_normals = np.vstack([normals1, normals2])
    total_penalty = 0.0
    for n in all_normals:
        p1 = v1 @ n
        p2 = v2 @ n
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()
        overlap = min(max1, max2) - max(min1, min2)
        if overlap > 0:
            total_penalty += overlap * overlap
    return total_penalty

def _point_outside_hex_penalty(point, outer_vertices):
    """
    For each outer edge (p1, p2), compute cross(p2-p1, point-p1).
    If any is negative, the point is outside. Penalty = sum of (negative values)^2.
    """
    penalty = 0.0
    for i in range(6):
        p1 = outer_vertices[i]
        p2 = outer_vertices[(i + 1) % 6]
        edge = p2 - p1
        to_point = point - p1
        cross_val = edge[0] * to_point[1] - edge[1] * to_point[0]
        if cross_val < 0:
            penalty += cross_val * cross_val
    return penalty

def _objective(params, s, angle_inner, angle_outer):
    """
    params: 24 floats = 12 inner hex centers (x,y pairs)
    Returns total penalty.
    """
    centers = params.reshape(12, 2)
    penalty = 0.0
    
    # Precompute inner hex vertices and normals
    inner_verts = []
    inner_normals = []
    for i in range(12):
        cx, cy = centers[i]
        v = _hex_vertices(cx, cy, 1.0, angle_inner)
        inner_verts.append(v)
        inner_normals.append(_hex_edge_normals(v))
    
    # Pairwise overlap penalties
    for i in range(12):
        for j in range(i + 1, 12):
            # Quick reject: if centers are far apart, skip
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 9.0:  # max distance for two unit hexagons to possibly overlap is ~3
                continue
            penalty += _sat_overlap_penalty(inner_verts[i], inner_verts[j], 
                                            inner_normals[i], inner_normals[j])
    
    # Outer hexagon vertices
    outer_verts = _hex_vertices(0, 0, s, angle_outer)
    
    # Boundary violation: all inner vertices must be inside outer hexagon
    for i in range(12):
        for v in inner_verts[i]:
            penalty += _point_outside_hex_penalty(v, outer_verts)
    
    return penalty

def _is_feasible(params, s, angle_inner, angle_outer, tol=1e-6):
    """Check if the configuration is feasible with given s."""
    centers = params.reshape(12, 2)
    
    # Precompute inner hex vertices and normals
    inner_verts = []
    inner_normals = []
    for i in range(12):
        cx, cy = centers[i]
        v = _hex_vertices(cx, cy, 1.0, angle_inner)
        inner_verts.append(v)
        inner_normals.append(_hex_edge_normals(v))
    
    # Pairwise separation check
    for i in range(12):
        for j in range(i + 1, 12):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 9.0:
                continue
            all_normals = np.vstack([inner_normals[i], inner_normals[j]])
            for n in all_normals:
                p1 = inner_verts[i] @ n
                p2 = inner_verts[j] @ n
                min1, max1 = p1.min(), p1.max()
                min2, max2 = p2.min(), p2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap < -tol:
                    return False
    
    # Boundary check
    outer_verts = _hex_vertices(0, 0, s, angle_outer)
    for i in range(12):
        for v in inner_verts[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                to_point = v - p1
                cross_val = edge[0] * to_point[1] - edge[1] * to_point[0]
                if cross_val < -tol:
                    return False
    
    return True

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    """
    start_time = time.time()
    deadline = 620.0  # safety margin
    
    best_score = 0.0
    best_result = None
    
    # Define the compact hex grid packing
    # Angle 0: Pointy top
    # Centers for a hexagonal grid
    # We use axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    sqrt3 = np.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    # Generate centers for a hexagon of radius 1 (7 hexagons)
    # and select 5 more from radius 2 to make 12.
    # Radius 1 centers:
    centers_r1 = [
        axial_to_cartesian(0, 0),
        axial_to_cartesian(1, 0),
        axial_to_cartesian(1, -1),
        axial_to_cartesian(0, -1),
        axial_to_cartesian(-1, 0),
        axial_to_cartesian(-1, 1),
        axial_to_cartesian(0, 1)
    ]
    
    # Radius 2 centers (neighbors of R1)
    # We want to pick 5 that keep the bounding box tight.
    # The R2 ring has 12 hexagons.
    # Let's pick a contiguous block or symmetric set.
    # Symmetric set: 5 hexagons spread out?
    # Or just the first 5 in angular order?
    # Let's try to pick 5 that are "closest" to the center in terms of max norm? They are all equidistant.
    # To minimize the outer hexagon side s, we want to minimize the max distance of any vertex.
    # The vertices of the R2 hexagons are at distance 2*sqrt(3) + 1 from origin.
    # So s must be at least 2*sqrt(3) + 1.
    # Any 5 will result in s >= 2*sqrt(3) + 1 if they are on the R2 ring.
    # So let's just pick 5 arbitrary ones, e.g., q,r pairs:
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1) ? 
    # Let's just take the first 5 from a list of R2 centers.
    r2_centers = [
        axial_to_cartesian(2, 0),
        axial_to_cartesian(2, -1),
        axial_to_cartesian(1, -2),
        axial_to_cartesian(0, -2),
        axial_to_cartesian(-1, -1)
    ]
    
    all_centers = centers_r1 + r2_centers
    x0 = np.array(all_centers).flatten()
    
    angle_inner = 0.0
    angle_outer = 0.0
    
    # Calculate the theoretical minimum s for this configuration
    # Max distance from origin to any vertex
    max_dist = 0
    for c in all_centers:
        verts = _hex_vertices(c[0], c[1], 1.0, angle_inner)
        dists = np.linalg.norm(verts, axis=1)
        max_dist = max(max_dist, np.max(dists))
    
    # The outer hexagon with side s has apothem s*sqrt(3)/2 and vertex dist s.
    # The constraint is tighter for the apothem if the point is near an edge, or vertex if near a corner.
    # For a pointy-top hexagon, the vertices are at angles 0, 60, ...
    # The R2 centers are at angles 0, 60, ...
    # The vertex of the inner hex at (2*sqrt(3), 0) is at (2*sqrt(3)+1, 0).
    # This aligns with the outer hex vertex at (s, 0).
    # So s must be >= 2*sqrt(3) + 1.
    
    s_min_theoretical = 2 * sqrt3 + 1.0
    
    # We will try to optimize around this s.
    # Start with s = s_min_theoretical + small_epsilon
    s_start = s_min_theoretical + 0.1
    
    # Check if feasible at s_start
    if _is_feasible(x0, s_start, angle_inner, angle_outer):
        # Try to reduce s
        s_low = s_min_theoretical
        s_high = s_start
        
        for _ in range(10):
            if time.time() > deadline:
                break
            s_mid = (s_low + s_high) / 2
            if _is_feasible(x0, s_mid, angle_inner, angle_outer):
                s_high = s_mid
            else:
                s_low = s_mid
        
        best_s = s_high
        # Verify and store
        if _is_feasible(x0, best_s, angle_inner, angle_outer):
            centers = x0.reshape(12, 2)
            inner_data = np.column_stack([centers, np.full(12, angle_inner)])
            outer_data = np.array([0, 0, angle_outer])
            best_result = (inner_data, outer_data, best_s)
            best_score = 1.0 / best_s
    
    # If the analytic packing didn't work or we can do better with optimization
    # Run a short optimization to see if we can squeeze s further
    if time.time() < deadline - 60:
        # Try optimizing the positions slightly
        # We fix s at the best found so far (or theoretical min) and minimize penalty
        s_opt = best_s if best_result else s_min_theoretical
        
        # We want to minimize s, so we can optimize positions for a given s.
        # But s is discrete in our binary search.
        # Let's try to optimize positions for s = s_min_theoretical to see if it's feasible.
        # If feasible, s_min_theoretical is the answer.
        
        result = minimize(_objective, x0, args=(s_min_theoretical, angle_inner, angle_outer),
                          method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
        
        if result.fun < 1e-8:
            if _is_feasible(result.x, s_min_theoretical, angle_inner, angle_outer):
                centers = result.x.reshape(12, 2)
                inner_data = np.column_stack([centers, np.full(12, angle_inner)])
                outer_data = np.array([0, 0, angle_outer])
                new_result = (inner_data, outer_data, s_min_theoretical)
                new_score = 1.0 / s_min_theoretical
                if new_score > best_score:
                    best_result = new_result
                    best_score = new_score

    if best_result is None:
        # Fallback to seed if all else fails (should not happen with the compact packing)
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    return best_result

# EVOLVE-BLOCK-END
