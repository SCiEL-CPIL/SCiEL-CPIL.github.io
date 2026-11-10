import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _check_validity(inner, outer, s):
    """Validate packing; return (is_valid, error_msg)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < np.sqrt(12.0) - 1e-12:
        return False, "s < sqrt(12)"
    
    # Outer hexagon vertices must be distinct
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False, "outer vertices not distinct"
    
    # Check unit edge length for inner hexagons
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for i in range(6):
            edge_len = np.linalg.norm(hv[(i + 1) % 6] - hv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False, f"unit edge length differs: {edge_len}"
    
    # Check inner-inner separation using SAT
    def _separated(v1, v2):
        """Check if two hexagons are separated (non-overlapping) using SAT."""
        edges = []
        for i in range(6):
            edges.append(v1[(i + 1) % 6] - v1[i])
            edges.append(v2[(i + 1) % 6] - v2[i])
        
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap1 = min2 - max1
            gap2 = min1 - max2
            if gap1 > 1e-6 or gap2 > 1e-6:
                return True
        
        return False
    
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separated(inner_verts[i], inner_verts[j]):
                return False, f"inner hexagons {i} and {j} overlap"
    
    # Check inner vertices inside outer hexagon
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False, f"inner vertex outside outer hexagon"
    
    return True, ""


def _compute_min_s(inner):
    """
    Compute the minimum outer hexagon side length needed to contain all inner hexagons.
    Assumes outer hexagon is centered at origin with angle 0.
    """
    normal_angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_projections = np.zeros(6)
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        projections = hv @ normals.T
        max_projections = np.maximum(max_projections, np.max(projections, axis=0))
    
    s_min = np.max(max_projections) / (np.sqrt(3.0) / 2.0)
    return s_min + 1e-4


def _lattice_to_cartesian(lattice_coords, rotation_angle):
    """
    Convert hexagonal lattice coordinates to Cartesian.
    
    In a hexagonal lattice of unit hexagons (side 1), adjacent centers are
    separated by sqrt(3). We use axial coordinates (q, r) where:
    - Moving in q direction: offset (sqrt(3), 0)
    - Moving in r direction: offset (sqrt(3)/2, 3/2)
    
    Actually, for regular hexagons with side length 1, the distance between
    centers of edge-adjacent hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    
    Using axial coordinates (q, r):
    x = sqrt(3) * (q + r/2)
    y = 3/2 * r
    
    This gives 6-neighbor connectivity with correct distances.
    """
    q = lattice_coords[:, 0]
    r = lattice_coords[:, 1]
    
    x = np.sqrt(3.0) * (q + r / 2.0)
    y = 1.5 * r
    
    # Apply rotation
    theta = np.deg2rad(rotation_angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    
    return np.column_stack([x_rot, y_rot])


def _min_containing_hex_side(points, outer_angle=0.0):
    """
    Compute minimum side length s of a regular hexagon (centered at origin,
    with given angle) that contains all given points.
    
    For a regular hexagon with side s centered at origin and angle a:
    The 6 inward-facing normals to the sides are at angles a + 30 + 60*i.
    The distance from center to each side is s * sqrt(3)/2.
    
    So we need: max over all points and all 6 normals of |dot(point, normal)| <= s * sqrt(3)/2
    """
    normal_angles = np.deg2rad(outer_angle + 30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Project all points onto each normal
    projections = points @ normals.T  # shape (n_points, 6)
    
    # For containment, we need both positive and negative bounds
    # The hexagon is symmetric, so we need max |projection| in each normal direction
    # Actually, for a convex hexagon containing origin, we need:
    # For each side with outward normal n, all points must satisfy dot(point, n) <= s*sqrt(3)/2
    # And also dot(point, -n) <= s*sqrt(3)/2 (which is dot(point, n) >= -s*sqrt(3)/2)
    
    max_abs_proj = np.max(np.abs(projections), axis=0)  # shape (6,)
    s_min = np.max(max_abs_proj) / (np.sqrt(3.0) / 2.0)
    
    return s_min


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Uses hexagonal lattice representation: 12 hexagons placed on a dual triangular grid
    with integer axial coordinates, ensuring non-overlap by construction.
    Optimizes rotation angle and translation to minimize the containing regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin
    
    # Define 12 lattice positions in axial coordinates (q, r)
    # We need 12 hexagons that form a compact cluster.
    # 
    # A hexagonal cluster of 12 hexagons:
    # Center (0,0)
    # Ring 1: 6 hexagons at distance 1
    # Ring 2: 5 more hexagons to make 12 total
    #
    # Axial coordinates for a hexagonal grid:
    # Ring 1 (6 hexagons): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: we need 5 more. Let's pick a compact set.
    #
    # Alternative: use a 3x4 rectangular-ish arrangement or other compact shape.
    #
    # Let's try: center + ring1 (6) + 5 from ring2
    # Ring 2 positions include: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-1)
    # Wait, let me be more careful.
    
    # Axial coordinate neighbors of (0,0): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Axial coordinate neighbors at distance 2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-1)
    # Actually let me list them properly:
    # From (1,0): (2,0), (1,1), (2,-1)
    # From (0,1): (1,1), (0,2), (-1,2)
    # From (-1,1): (-1,2), (-2,2), (-2,1)
    # From (-1,0): (-2,1), (-2,0), (-1,-1)
    # From (0,-1): (-1,-1), (0,-2), (1,-2)
    # From (1,-1): (2,-1), (1,-2), (2,-2)
    
    # Unique ring-2 positions: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)
    # That's 12 positions in ring 2. We need 5 of them.
    
    # For compactness, let's pick 5 that form a nice shape.
    # Try: (2,0), (1,1), (0,2), (-1,2), (-2,1) - these form an arc
    
    lattice_positions = np.array([
        [0, 0],    # center
        [1, 0],    # ring 1
        [0, 1],
        [-1, 1],
        [-1, 0],
        [0, -1],
        [1, -1],
        [2, 0],    # ring 2 (5 more)
        [1, 1],
        [0, 2],
        [-1, 2],
        [-2, 1],
    ])
    
    # Alternative compact configurations to try
    alt_configs = [
        np.array([
            [0, 0],
            [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
            [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 1],
        ]),
        np.array([
            [0, 0],
            [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
            [2, 0], [2, -1], [1, -2], [0, -2], [-1, -1],
        ]),
        np.array([
            [0, 0],
            [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
            [2, 0], [1, 1], [0, 2], [-1, 2], [2, -1],
        ]),
        np.array([
            [0, 0],
            [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
            [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 2],
        ]),
        # More symmetric: center + 6 ring1 + 5 spread evenly
        np.array([
            [0, 0],
            [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
            [2, 0], [0, 2], [-2, 2], [-2, 0], [0, -2],
        ]),
        # Try a different cluster shape
        np.array([
            [0, 0],
            [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
            [2, 0], [1, 1], [0, 2], [-1, 2], [1, -2],
        ]),
    ]
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    for config_idx, lattice_pos in enumerate(alt_configs):
        if time.time() > deadline:
            break
        
        # The lattice positions define relative positions.
        # We need to:
        # 1. Rotate the entire configuration by angle theta
        # 2. Translate so that the center of mass is at origin (or optimize translation)
        # 3. Find the minimum containing regular hexagon
        
        # For a given rotation angle, compute Cartesian positions
        # Then center them (subtract centroid)
        # Then find min containing hexagon
        
        def objective(params):
            """
            params: [theta_deg, tx, ty]
            theta: rotation angle in degrees
            tx, ty: translation
            """
            theta = params[0]
            tx = params[1]
            ty = params[2]
            
            # Convert lattice to Cartesian with rotation
            cart = _lattice_to_cartesian(lattice_pos, theta)
            
            # Center the configuration (subtract centroid) then add translation
            centroid = cart.mean(axis=0)
            cart_centered = cart - centroid + np.array([tx, ty])
            
            # Compute min containing hexagon side
            s = _min_containing_hex_side(cart_centered)
            
            return s
        
        # Multi-start optimization over rotation angle
        # The optimal rotation will align the lattice with the outer hexagon
        # Try multiple initial angles
        
        for theta_init in range(0, 360, 15):
            if time.time() > deadline:
                break
            
            x0 = np.array([float(theta_init), 0.0, 0.0])
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                
                if result.fun < best_s:
                    best_s = result.fun
                    theta_opt = result.x[0]
                    tx_opt = result.x[1]
                    ty_opt = result.x[2]
                    
                    # Compute final positions
                    cart = _lattice_to_cartesian(lattice_pos, theta_opt)
                    centroid = cart.mean(axis=0)
                    cart_final = cart - centroid + np.array([tx_opt, ty_opt])
                    
                    # Build inner array: (x, y, angle_deg) for each hexagon
                    # All hexagons have the same orientation (angle = theta_opt)
                    inner = np.column_stack([
                        cart_final[:, 0],
                        cart_final[:, 1],
                        np.full(12, theta_opt)
                    ])
                    outer = np.array([0.0, 0.0, 0.0])
                    
                    # Validate
                    s_val = best_s + 1e-4  # small margin
                    valid, msg = _check_validity(inner, outer, s_val)
                    if valid:
                        best_inner = inner
                        best_outer = outer
                        best_s = s_val
                        print(f"Config {config_idx}, theta={theta_init}: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
                        
            except Exception:
                pass
        
        # Also try local refinement around best angles found
        if time.time() < deadline - 30 and best_inner is not None:
            # Fine-tune the best result
            theta_fine = best_inner[0, 2]  # current angle
            for dtheta in [-5, -2, -1, 1, 2, 5]:
                if time.time() > deadline:
                    break
                x0 = np.array([theta_fine + dtheta, 0.0, 0.0])
                try:
                    result = minimize(objective, x0, method='Nelder-Mead',
                                     options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9})
                    if result.fun + 1e-4 < best_s:
                        s_val = result.fun + 1e-4
                        theta_opt = result.x[0]
                        tx_opt = result.x[1]
                        ty_opt = result.x[2]
                        
                        cart = _lattice_to_cartesian(lattice_pos, theta_opt)
                        centroid = cart.mean(axis=0)
                        cart_final = cart - centroid + np.array([tx_opt, ty_opt])
                        
                        inner = np.column_stack([
                            cart_final[:, 0],
                            cart_final[:, 1],
                            np.full(12, theta_opt)
                        ])
                        outer = np.array([0.0, 0.0, 0.0])
                        
                        valid, msg = _check_validity(inner, outer, s_val)
                        if valid:
                            best_inner = inner
                            best_outer = outer
                            best_s = s_val
                            print(f"Refined: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
                except Exception:
                    pass
    
    # Fallback if no valid solution found
    if best_inner is None:
        print("No valid solution found, using fallback")
        inner = np.array([
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [-2.5, 0.0, 0.0],
            [1.25, 2.1650635, 0.0],
            [-1.25, 2.1650635, 0.0],
            [1.25, -2.1650635, 0.0],
            [-1.25, -2.1650635, 0.0],
            [3.75, 2.1650635, 0.0],
            [-3.75, 2.1650635, 0.0],
            [3.75, -2.1650635, 0.0],
            [-3.75, -2.1650635, 0.0],
            [0.0, -4.330127, 0.0],
        ])
        outer = np.array([0.0, 0.0, 0.0])
        s = _compute_min_s(inner)
        valid, msg = _check_validity(inner, outer, s)
        if not valid:
            s = 20.0
        return inner, outer, s
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
