# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _hex_normals(a_deg):
    """12 edge normals (unit vectors) for a hexagon of rotation a_deg."""
    a = np.radians(a_deg)
    # Normals are perpendicular to edges.
    # Vertex i is at angle a + 60i.
    # Edge i connects vertex i and i+1.
    # Midpoint angle is a + 60i + 30.
    # Normal angle is midpoint angle.
    edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    return np.vstack([normals, -normals])

def _separation_ok(v1, v2, n1, n2):
    """Check if two hexagons are separated."""
    for normals in (n1, n2):
        for j in range(normals.shape[0]):
            n = normals[j]
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True
    return False

def _hexagons_separated(v1, v2, a1, a2):
    n1 = _hex_normals(a1)
    n2 = _hex_normals(a2)
    return _separation_ok(v1, v2, n1, n2)

def _point_in_hexagon(p, outer_verts):
    """Check if point p is inside (or on boundary of) the outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        to_p = p - p1
        cross = edge[0] * to_p[1] - edge[1] * to_p[0]
        if cross < -1e-6:
            return False
    return True

def _all_inner_in_outer(inner_verts_list, outer_verts):
    """Check all vertices of all inner hexagons are inside the outer hexagon."""
    for verts in inner_verts_list:
        for p in verts:
            if not _point_in_hexagon(p, outer_verts):
                return False
    return True

def _compute_minimal_outer_hex(all_verts):
    """
    Compute the minimal enclosing regular hexagon for a set of points.
    Returns (center_x, center_y, angle_deg, side_length).
    """
    # Use convex hull to reduce points
    try:
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[hull.vertices]
    except Exception:
        hull_verts = all_verts
    
    if len(hull_verts) < 3:
        center = np.mean(hull_verts, axis=0) if len(hull_verts) > 0 else np.array([0.0, 0.0])
        if len(hull_verts) > 0:
            dists = np.linalg.norm(hull_verts - center, axis=1)
            r = np.max(dists)
            s = r / (np.sqrt(3) / 2)
            return (center[0], center[1], 0.0, s)
        return (0.0, 0.0, 0.0, np.sqrt(11.0))

    best_s = np.inf
    best_center = np.mean(hull_verts, axis=0)
    best_angle = 0.0
    
    centroid = np.mean(hull_verts, axis=0)
    
    # Search over angles 0 to 60 degrees
    # The optimal orientation of the enclosing hexagon is likely aligned with one of the hull edges or vertices.
    # We sample a fine grid of angles.
    angles = np.linspace(0, 60, 61)
    
    for a_deg in angles:
        a = np.radians(a_deg)
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        # Projections of hull verts onto normals
        proj = hull_verts @ normals.T  # (M, 6)
        max_proj = proj.max(axis=0)
        min_proj = proj.min(axis=0)
        
        # For a fixed angle, find optimal center C and apothem A.
        # Minimize A such that C @ n_i +/- A covers [min_proj_i, max_proj_i].
        # This is equivalent to minimizing max_i (max(C @ n_i - min_proj_i, max_proj_i - C @ n_i)).
        
        def obj_c(c):
            cp = c @ normals.T
            return np.max(np.maximum(cp - min_proj, max_proj - c_proj))
            
        # Use Nelder-Mead to find optimal center for this angle
        try:
            res = minimize(obj_c, centroid, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            A = res.fun
            C = res.x
        except Exception:
            C = centroid
            cp = C @ normals.T
            A = np.max(np.maximum(cp - min_proj, max_proj - c_proj))
            
        s = A / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
            best_center = C
            best_angle = a_deg
            
    return (best_center[0], best_center[1], best_angle, best_s)

def _build_hexagons_from_params(r1, a1, r2, a2):
    """
    Build the 11 hexagons based on symmetry-reduced parameters.
    r1: radius of inner ring (6 hexes)
    a1: angular offset of inner ring
    r2: radius of outer ring (4 hexes)
    a2: angular offset of outer ring
    All hexagons have side 1 and rotation 0 (aligned).
    """
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Inner Ring (6 hexes)
    for i in range(6):
        ang = np.deg2rad(60 * i + a1)
        positions.append([r1 * np.cos(ang), r1 * np.sin(ang)])
        angles.append(0.0)
        
    # Outer Ring (4 hexes)
    # To maintain symmetry and fill gaps, we place them at 45, 135, 225, 315 relative to a1 offset?
    # Actually, the "flower" pattern with 11 hexes usually has the outer 4 aligned with the gaps of the inner 6.
    # Inner 6 are at 0, 60, 120, 180, 240, 300 (plus offset a1).
    # Gaps are at 30, 90, 150, 210, 270, 330.
    # We pick 4 of these 6 gaps. Due to symmetry, picking 4 breaks 6-fold symmetry but keeps some.
    # A common tight packing for 11 is 1 center, 6 ring, and 4 in the "corners" or "edges" of the hexagon.
    # Let's try placing them at angles: a1 + 30, a1 + 150, a1 + 210, a1 + 330? 
    # Or simply a1 + 30 + 90*i? No, that's square symmetry.
    # Let's try a1 + 30, a1 + 120+30? No.
    # The 4 outer hexes in a minimal hexagon packing of 11 are typically placed at the vertices of a rhombus or similar.
    # Actually, the optimal packing of 11 unit hexagons in a hexagon is known to be the "flower" with 1 center, 6 ring, and 4 outer.
    # The 4 outer ones are usually placed in 4 of the 6 gaps.
    # Let's parameterize the angular positions of the 4 outer hexes relative to the inner ring.
    # Due to 6-fold symmetry of the inner ring, the outer ring positions can be defined by their angles.
    # Let's fix the outer ring angles to be 30, 150, 210, 330 degrees relative to the inner ring's 0-degree hex?
    # This corresponds to gaps at 30, 150, 210, 330.
    # Wait, 30 and 330 are symmetric. 150 and 210 are symmetric.
    # This configuration has D2 symmetry (or C2?).
    # Let's allow the outer ring angle offset a2 to shift the whole outer ring.
    # Base angles for outer 4: 30, 150, 210, 330.
    
    outer_base_angles = [30, 150, 210, 330]
    for ang_deg in outer_base_angles:
        ang = np.deg2rad(ang_deg + a2)
        positions.append([r2 * np.cos(ang), r2 * np.sin(ang)])
        angles.append(0.0)
        
    return np.array(positions), np.array(angles)

def _check_validity(positions, angles):
    """Check if the configuration is valid (no overlaps)."""
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in zip(positions[:,0], positions[:,1], angles)]
    for i in range(11):
        for j in range(i + 1, 11):
            if not _hexagons_separated(verts_list[i], verts_list[j], angles[i], angles[j]):
                return False
    return True

def _objective(params):
    """
    Objective function: returns side length of minimal enclosing hexagon.
    If invalid, returns infinity.
    """
    r1, a1, r2, a2 = params
    if r1 < 0 or r2 < 0:
        return np.inf
        
    positions, angles = _build_hexagons_from_params(r1, a1, r2, a2)
    
    if not _check_validity(positions, angles):
        return np.inf
        
    all_verts = np.vstack([_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in zip(positions[:,0], positions[:,1], angles)])
    ox, oy, oa, s = _compute_minimal_outer_hex(all_verts)
    
    # Verify containment
    outer_verts = _hex_vertices(ox, oy, s, oa)
    if not _all_inner_in_outer([_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in zip(positions[:,0], positions[:,1], angles)], outer_verts):
        # Expand s slightly
        for _ in range(10):
            s *= 1.01
            outer_verts = _hex_vertices(ox, oy, s, oa)
            if _all_inner_in_outer([_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in zip(positions[:,0], positions[:,1], angles)], outer_verts):
                break
        else:
            return np.inf
            
    return s

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Initial guess
    # Theoretical min distance for touching hexes (edge to edge) is sqrt(3) ~ 1.732.
    # For the inner ring, r1 should be around sqrt(3).
    # For the outer ring, r2 should be larger.
    
    # Grid search for coarse parameters
    best_params = None
    best_score = np.inf
    
    # r1 range: 1.7 to 2.0
    # a1 range: 0 to 60 (due to symmetry, 0-30 might suffice, but let's do 0-60)
    # r2 range: 2.5 to 3.5
    # a2 range: 0 to 60
    
    # Coarse grid
    r1_vals = np.linspace(1.7, 2.0, 20)
    a1_vals = np.linspace(0, 60, 12)
    r2_vals = np.linspace(2.5, 3.5, 20)
    a2_vals = np.linspace(0, 60, 12)
    
    # We can't iterate all 20*12*20*12 = 57600 combinations with expensive checks in the time limit easily if checks are slow.
    # Let's reduce resolution or use a smarter search.
    # Actually, _check_validity is O(N^2 * 12) which is fast. _compute_minimal_outer_hex is the bottleneck.
    # Let's do a coarser grid first.
    
    r1_vals = np.linspace(1.7, 2.0, 10)
    a1_vals = np.linspace(0, 30, 6) # Symmetry
    r2_vals = np.linspace(2.5, 3.5, 10)
    a2_vals = np.linspace(0, 30, 6)
    
    for r1 in r1_vals:
        if time.time() > deadline - 20:
            break
        for a1 in a1_vals:
            if time.time() > deadline - 20:
                break
            for r2 in r2_vals:
                if time.time() > deadline - 20:
                    break
                for a2 in a2_vals:
                    if time.time() > deadline - 20:
                        break
                    params = [r1, a1, r2, a2]
                    score = _objective(params)
                    if score < best_score:
                        best_score = score
                        best_params = params
                        
    # Local optimization using Nelder-Mead from the best coarse point
    if best_params is not None:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Final verification and construction
    if best_params is None:
        # Fallback to a known valid configuration if search failed
        # Use the physics-based result from the parent as a fallback? No, let's just use a safe default.
        # Safe default: r1=sqrt(3), a1=0, r2=2*sqrt(3), a2=0
        best_params = [np.sqrt(3), 0.0, 2*np.sqrt(3), 0.0]
        
    positions, angles = _build_hexagons_from_params(*best_params)
    
    # Re-center
    centroid = np.mean(positions, axis=0)
    positions -= centroid
    
    inner_hex_data = np.column_stack([positions, angles])
    
    all_verts = np.vstack([_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in zip(positions[:,0], positions[:,1], angles)])
    ox, oy, oa, s = _compute_minimal_outer_hex(all_verts)
    
    outer_hex_data = np.array([ox, oy, oa])
    
    # Final containment check and adjustment
    outer_verts = _hex_vertices(ox, oy, s, oa)
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in zip(positions[:,0], positions[:,1], angles)]
    
    if not _all_inner_in_outer(verts_list, outer_verts):
        for _ in range(50):
            s *= 1.01
            outer_verts = _hex_vertices(ox, oy, s, oa)
            if _all_inner_in_outer(verts_list, outer_verts):
                break
                
    s = max(s, np.sqrt(11.0))
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
