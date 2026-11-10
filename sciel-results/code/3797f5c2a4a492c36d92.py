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
    Returns total penalty + lambda * s (to minimize s implicitly via constraint)
    Actually, we'll use a different approach: fix s, check feasibility.
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
    
    # Initial configuration: flower-like arrangement
    # Center + 6 around + 6 more in a ring
    def make_initial_config(scale=1.0):
        centers = np.zeros((12, 2))
        # Center
        centers[0] = [0, 0]
        # First ring: 6 hexagons at distance 2.5 (touching for flat-top)
        # For unit hexagons, center-to-center distance for touching is 2 * (sqrt(3)/2 * 2) = 2*sqrt(3) ~ 3.464
        # Actually for flat-top hexagons, horizontal neighbors are 3 apart, diagonal are 2*sqrt(3) apart
        # Let's use the standard hex grid: flat-top, neighbor offsets
        # Flat-top hex with side 1: width = 2, height = sqrt(3)
        # Horizontal neighbors: dx=3, dy=0
        # Diagonal neighbors: dx=1.5, dy=sqrt(3)/2 * 2 = sqrt(3)
        
        # Let's place them in a hexagonal pattern
        # Row 0: 3 hexagons
        # Row 1: 4 hexagons  
        # Row 2: 3 hexagons
        # Row 3: 2 hexagons
        # That's 12
        
        # Actually, let's use a simpler approach: optimize from a reasonable starting point
        # Start with the seed arrangement but scaled
        base = np.array([
            [0, 0],
            [-2.5, 0],
            [2.5, 0],
            [-1.25, 2.17],
            [1.25, 2.17],
            [-1.25, -2.17],
            [1.25, -2.17],
            [-3.75, 2.17],
            [3.75, 2.17],
            [-3.75, -2.17],
            [3.75, -2.17],
            [0, -4],
        ])
        return base * scale
    
    # Try multiple initial configurations and angles
    angle_options = [0, 30]  # flat-top and pointy-top
    
    for angle_outer in angle_options:
        for angle_inner in angle_options:
            if time.time() > deadline:
                break
                
            for scale in [1.0, 0.9, 1.1]:
                if time.time() > deadline:
                    break
                    
                x0 = make_initial_config(scale).flatten()
                
                # Binary search on s to find minimum feasible s
                # Start with a large s and decrease
                s_low = 4.0  # theoretical minimum for 12 hexagons
                s_high = 8.0
                
                # First, verify s_high is feasible
                if not _is_feasible(x0, s_high, angle_inner, angle_outer):
                    # Try to optimize at s_high first
                    result = minimize(_objective, x0, args=(s_high, angle_inner, angle_outer),
                                     method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
                    if result.fun < 1e-6:
                        x0 = result.x
                    else:
                        continue
                
                # Binary search for minimum s
                best_s = s_high
                best_x = x0.copy()
                
                for iteration in range(20):
                    if time.time() > deadline:
                        break
                    
                    s_mid = (s_low + s_high) / 2
                    
                    # Try to optimize at s_mid
                    result = minimize(_objective, x0, args=(s_mid, angle_inner, angle_outer),
                                     method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12})
                    
                    if result.fun < 1e-8:
                        # Feasible
                        if _is_feasible(result.x, s_mid, angle_inner, angle_outer):
                            best_s = s_mid
                            best_x = result.x
                            s_high = s_mid
                            x0 = result.x
                        else:
                            s_low = s_mid
                    else:
                        # Not feasible at s_mid
                        s_low = s_mid
                
                # Verify best solution
                if _is_feasible(best_x, best_s, angle_inner, angle_outer):
                    score = 1.0 / best_s
                    if score > best_score:
                        best_score = score
                        centers = best_x.reshape(12, 2)
                        inner_data = np.column_stack([centers, np.full(12, angle_inner)])
                        outer_data = np.array([0, 0, angle_outer])
                        best_result = (inner_data, outer_data, best_s)
    
    if best_result is None:
        # Fallback: return the seed solution
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
