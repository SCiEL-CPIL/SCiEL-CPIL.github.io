# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(verts):
    """Compute outward edge normals for a hexagon (6 edges)."""
    n = len(verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        # Outward normal: perpendicular to edge, pointing outward
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm > 1e-15:
            normals[i] = [nx / norm, ny / norm]
    return normals

def _separation_check(verts1, verts2, tol=1e-6):
    """Separating axis test: return True if separated (non-overlapping)."""
    # Get all edge normals from both hexagons
    normals1 = _hex_edge_normals(verts1)
    normals2 = _hex_edge_normals(verts2)
    all_normals = np.vstack([normals1, normals2])
    
    for normal in all_normals:
        proj1 = verts1 @ normal
        proj2 = verts2 @ normal
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # If intervals don't overlap (with tolerance), they're separated
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    return False

def _is_inside_outer(verts_inner, outer_verts, tol=1e-6):
    """Check if all vertices of inner hex are inside outer hex."""
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        # Cross product: cross(edge, v - p1) should be >= -tol for CCW polygon
        for v in verts_inner:
            cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
            if cross < -tol:
                return False
    return True

def _validate(inner_data, outer_data, s):
    """Full validation check. Returns True if valid."""
    n_inner = len(inner_data)
    if n_inner != 12:
        return False
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    # Compute outer hex vertices
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons are unit
    inner_verts_list = []
    for i in range(n_inner):
        x, y, ang = inner_data[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
        # Check edge lengths
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False
    
    # Check separation between all pairs
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            if not _separation_check(inner_verts_list[i], inner_verts_list[j]):
                return False
    
    # Check all inner vertices inside outer
    for i in range(n_inner):
        if not _is_inside_outer(inner_verts_list[i], outer_verts):
            return False
    
    return True

def _compute_min_outer(inner_data, outer_center, outer_angle):
    """Find minimum side length s such that all inner hexagons fit inside outer hex."""
    # For a regular hexagon with center c, angle a, side s:
    # A point p is inside if for each edge, the cross product condition holds.
    # The constraint is: for each edge i, the distance from p to the line containing
    # edge i (on the interior side) must be <= apothem = s * sqrt(3)/2
    
    # Actually, let's use the vertex-based check and binary search on s
    lo = 1.0
    hi = 50.0
    
    # Quick check: if hi doesn't work, expand
    for _ in range(20):
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], hi, outer_angle)
        ok = True
        for i in range(len(inner_data)):
            x, y, ang = inner_data[i]
            verts = _hex_vertices(x, y, 1.0, ang)
            if not _is_inside_outer(verts, outer_verts):
                ok = False
                break
        if ok:
            break
        hi *= 2
    
    if not ok:
        return hi  # Return large value, invalid
    
    # Binary search
    for _ in range(60):
        mid = (lo + hi) / 2
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], mid, outer_angle)
        ok = True
        for i in range(len(inner_data)):
            x, y, ang = inner_data[i]
            verts = _hex_vertices(x, y, 1.0, ang)
            if not _is_inside_outer(verts, outer_verts):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    
    return hi

def _objective(params, lattice_offsets, n_inner=12):
    """
    params: (a, b, theta, phi) where:
      a, b: lattice basis vector lengths
      theta: angle between basis vectors
      phi: global rotation of the lattice
    Returns: 1/s (higher is better), or -inf if invalid.
    """
    a, b, theta, phi = params
    
    # Lattice basis vectors
    e1 = a * np.array([np.cos(phi), np.sin(phi)])
    e2 = b * np.array([np.cos(phi + np.radians(theta)), np.sin(phi + np.radians(theta))])
    
    # Generate 12 lattice points (hexagonal arrangement: 1 center + 6 ring + 5 more)
    # For a hexagonal lattice, the 12 points in a hexagonal cluster:
    # Center: (0,0)
    # Ring 1: 6 points at distance a (or b depending on lattice)
    # Ring 2: partial ring
    
    # Actually, let's use a fixed set of lattice indices for 12 points
    # Hexagonal arrangement: indices (i,j) where i,j are lattice coordinates
    # For a hex of radius 2: all (i,j) with max(|i|,|j|,|i+j|) <= 2 gives 19 points
    # For 12 points, use a hex of radius 1 (7 points) + 5 from radius 2
    
    # Let's use a specific set of 12 lattice points
    # Hexagonal lattice: point = i*e1 + j*e2
    # For a compact cluster of 12:
    lattice_indices = [
        (0, 0),      # center
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),  # ring 1 (6 points)
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2),  # 5 more from ring 2
    ]
    
    # Generate positions
    positions = np.zeros((n_inner, 2))
    for idx, (i, j) in enumerate(lattice_indices):
        positions[idx] = i * e1 + j * e2
    
    # Inner hexagons: all with angle 0 (or we could add a global inner angle)
    # Let's also try rotating inner hexagons by a fixed amount for better fit
    inner_angle = 30.0  # 30 degrees for better alignment with hexagonal symmetry
    
    inner_data = np.column_stack([positions, np.full(n_inner, inner_angle)])
    
    # Find optimal outer hexagon
    # The outer hexagon should be centered at the centroid of inner hexagons
    outer_center = np.mean(positions, axis=0)
    
    # Try a few outer angles to find the best fit
    best_s = np.inf
    best_outer_angle = 0.0
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        s = _compute_min_outer(inner_data, outer_center, outer_angle)
        if s < best_s:
            best_s = s
            best_outer_angle = outer_angle
    
    if not np.isfinite(best_s):
        return -np.inf
    
    # Validate
    outer_data = np.array([outer_center[0], outer_center[1], best_outer_angle])
    if not _validate(inner_data, outer_data, best_s):
        return -np.inf
    
    return 1.0 / best_s

def _optimize_lattice():
    """Optimize the lattice parameters using scipy."""
    from scipy.optimize import minimize
    
    best_value = -np.inf
    best_params = None
    
    # Multiple starting points
    starts = [
        (2.5, 2.5, 60, 0),
        (2.5, 2.5, 60, 30),
        (3.0, 3.0, 60, 0),
        (2.0, 2.0, 60, 0),
        (2.5, 2.5, 90, 0),
        (2.5, 2.5, 120, 0),
        (2.2, 2.2, 60, 15),
        (2.8, 2.8, 60, 45),
    ]
    
    bounds = [(1.5, 5.0), (1.5, 5.0), (30, 150), (0, 180)]
    
    for start in starts:
        try:
            res = minimize(
                lambda p: -_objective(p, None),
                start,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            val = -res.fun
            if val > best_value:
                best_value = val
                best_params = res.x
        except Exception:
            pass
    
    return best_value, best_params

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns:
        inner_hex_data: (12, 3) array of (x, y, angle_deg)
        outer_hex_data: (3,) array of (x, y, angle_deg)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # 580 second safety margin
    
    best_value = -np.inf
    best_result = None
    
    # Phase 1: Lattice-based optimization
    value, params = _optimize_lattice()
    elapsed = time.time() - start_time
    
    if value > best_value and params is not None:
        best_value = value
        # Reconstruct the result
        a, b, theta, phi = params
        e1 = a * np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
        e2 = b * np.array([np.cos(np.radians(phi + theta)), np.sin(np.radians(phi + theta))])
        
        lattice_indices = [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
            (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2),
        ]
        
        positions = np.zeros((12, 2))
        for idx, (i, j) in enumerate(lattice_indices):
            positions[idx] = i * e1 + j * e2
        
        inner_angle = 30.0
        inner_data = np.column_stack([positions, np.full(12, inner_angle)])
        
        outer_center = np.mean(positions, axis=0)
        
        best_s = np.inf
        best_outer_angle = 0.0
        for outer_angle in [0, 30, 60, 90, 120, 150]:
            s = _compute_min_outer(inner_data, outer_center, outer_angle)
            if s < best_s:
                best_s = s
                best_outer_angle = outer_angle
        
        outer_data = np.array([outer_center[0], outer_center[1], best_outer_angle])
        
        if _validate(inner_data, outer_data, best_s):
            best_result = (inner_data, outer_data, best_s)
    
    # Phase 2: Free-position refinement if time permits
    elapsed = time.time() - start_time
    if elapsed < 400 and best_result is not None:
        try:
            from scipy.optimize import minimize
            
            inner_data, outer_data, s = best_result
            
            # Flatten all parameters: 12*(x,y,angle) + outer(x,y,angle) + s
            # But we'll optimize positions and angles, then recompute s
            def objective_free(x):
                # x: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, ox,oy,oa]
                inner_x = x[0:36].reshape(12, 3)
                outer_x = x[36:39]
                
                # Compute min s for this configuration
                s_min = _compute_min_outer(inner_x, outer_x[:2], outer_x[2])
                if not np.isfinite(s_min) or s_min < np.sqrt(12) - 1e-9:
                    return 1e10
                
                outer_data_local = np.array([outer_x[0], outer_x[1], outer_x[2]])
                if not _validate(inner_x, outer_data_local, s_min):
                    return 1e10
                
                return s_min  # Minimize s, equivalent to maximizing 1/s
            
            x0 = np.concatenate([inner_data.flatten(), outer_data.flatten()])
            
            res = minimize(
                objective_free,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            
            if res.fun < 1e9:
                inner_new = res.x[0:36].reshape(12, 3)
                outer_new = res.x[36:39]
                s_new = _compute_min_outer(inner_new, outer_new[:2], outer_new[2])
                
                if _validate(inner_new, outer_new, s_new):
                    val_new = 1.0 / s_new
                    if val_new > best_value:
                        best_value = val_new
                        best_result = (inner_new, outer_new, s_new)
        except Exception:
            pass
    
    # Fallback: seed-like solution
    if best_result is None:
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
        outer_hex_side_length = 8.0
        best_result = (inner_hex_data, outer_hex_data, outer_hex_side_length)
    
    return best_result[0], best_result[1], best_result[2]

# EVOLVE-BLOCK-END
