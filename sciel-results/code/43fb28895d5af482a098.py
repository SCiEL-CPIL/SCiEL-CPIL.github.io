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

def _objective(params, s, angle_inner, angle_outer, fixed_centers):
    """
    params: 10 floats = 5 inner hex centers (x,y pairs) for the movable ones.
    fixed_centers: (7, 2) array of the fixed 7-hex flower.
    """
    movable_centers = params.reshape(5, 2)
    centers = np.vstack([fixed_centers, movable_centers])
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
            if dist_sq > 9.0:
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

def _is_feasible(params, s, angle_inner, angle_outer, fixed_centers, tol=1e-6):
    """Check if the configuration is feasible with given s."""
    movable_centers = params.reshape(5, 2)
    centers = np.vstack([fixed_centers, movable_centers])
    
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
    
    # 7-hex flower (fixed)
    fixed_centers = np.array([
        [0, 0],
        [2, 0],
        [1, np.sqrt(3)],
        [-1, np.sqrt(3)],
        [-2, 0],
        [-1, -np.sqrt(3)],
        [1, -np.sqrt(3)]
    ])
    
    # Initial guess for the 5 movable hexagons
    # Place them at the midpoints of the outer hexagon sides, pushed in
    # Outer hexagon side s=4, apothem = 2*sqrt(3) ~ 3.46
    # Let's start them at distance 3.0 from center, at angles 30, 90, 150, 210, 270
    angles_5 = np.deg2rad([30, 90, 150, 210, 270])
    dist_5 = 3.0
    x0_5 = np.column_stack([dist_5 * np.cos(angles_5), dist_5 * np.sin(angles_5)]).flatten()
    
    angle_inner = 0
    angle_outer = 0
    
    # Try to find a valid packing for s=4
    for s in [4.0, 4.1, 4.2, 4.5, 5.0]:
        if time.time() > deadline:
            break
            
        # Optimize the 5 movable centers
        result = minimize(_objective, x0_5, args=(s, angle_inner, angle_outer, fixed_centers),
                         method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        if result.fun < 1e-8:
            # Verify feasibility
            if _is_feasible(result.x, s, angle_inner, angle_outer, fixed_centers):
                centers = np.vstack([fixed_centers, result.x.reshape(5, 2)])
                inner_data = np.column_stack([centers, np.full(12, angle_inner)])
                outer_data = np.array([0, 0, angle_outer])
                best_result = (inner_data, outer_data, s)
                best_score = 1.0 / s
                break # Found a good one, stop
    
    if best_result is None:
        # Fallback: return the seed solution (or a known valid one if we can find it quickly)
        # The seed solution is s=8.
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
