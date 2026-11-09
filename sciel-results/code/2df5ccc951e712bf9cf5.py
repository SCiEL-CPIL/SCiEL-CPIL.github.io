# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid(inner, outer, s, tol=1e-6, edge_tol=1e-8):
    """Check validity of a packing configuration."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < math.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons have unit edges
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > edge_tol:
                return False
    
    # Check separation between inner hexagons using SAT
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            
            normals = []
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n_norm = np.linalg.norm(n)
                    if n_norm > 1e-12:
                        n = n / n_norm
                        normals.append(n)
            
            separated = False
            for n in normals:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap > -tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                edge = p2 - p1
                rel = v - p1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _compute_min_s(inner, outer_angle_deg, outer_center):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    oang = np.deg2rad(outer_angle_deg)
    
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # n_k angles in radians
    n_angles = oang + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    # For each vertex v, compute max_k dot(v - C, n_k)
    dots = all_verts @ normals.T - outer_center @ normals.T  # shape (72, 6)
    max_dots = dots.max(axis=0)  # shape (6,)
    s_min = (2.0 / math.sqrt(3.0)) * max_dots.max()
    
    return s_min

def _rigid_cluster_packing(rotation_deg):
    """
    Rigid cluster: 12 hexagons in a fixed honeycomb pattern, rotated by rotation_deg.
    Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is scalar.
    """
    # Define a compact honeycomb arrangement of 12 hexagons
    # Using triangular lattice with spacing 2*sqrt(3) for touching hexagons
    d = 2 * math.sqrt(3)  # center-to-center distance for touching unit hexagons
    
    # Lattice coordinates (m, n) in triangular lattice basis
    # Pattern: 1 + 6 + 5 = 12 hexagons
    # Center: (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 5 selected hexagons for compactness
    
    lattice_coords = [
        (0, 0),       # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (-1, 2), (-2, 0), (0, -2)  # ring 2 (5 selected)
    ]
    
    # Convert lattice coords to Cartesian
    a1 = np.array([d, 0.0])
    a2 = np.array([d * math.cos(math.pi/3), d * math.sin(math.pi/3)])
    
    centers_local = []
    for (m, n) in lattice_coords:
        pos = m * a1 + n * a2
        centers_local.append(pos)
    centers_local = np.array(centers_local)
    
    # Center the cluster
    centroid = centers_local.mean(axis=0)
    centers_local = centers_local - centroid
    
    # Rotate the entire cluster
    rot = np.deg2rad(rotation_deg)
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    centers_rotated = centers_local @ R.T
    
    # Build inner array
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = centers_rotated[i, 0]
        inner[i, 1] = centers_rotated[i, 1]
        inner[i, 2] = rotation_deg  # all hexagons have same orientation
    
    # Compute minimum enclosing regular hexagon
    # Try multiple outer hexagon angles to find the minimum s
    best_s = float('inf')
    best_outer_angle = 0.0
    
    # The outer hexagon angle should be aligned with the cluster's symmetry
    # Try angles from 0 to 60 degrees in fine steps
    for oa in np.linspace(0, 60, 121):
        s = _compute_min_s(inner, oa, np.array([0.0, 0.0]))
        if s < best_s:
            best_s = s
            best_outer_angle = oa
    
    outer = np.array([0.0, 0.0, best_outer_angle])
    
    return inner, outer, best_s

def _rigid_cluster_objective(params):
    """Objective: minimize s. params: [rotation_deg]"""
    rotation_deg = params[0]
    inner, outer, s = _rigid_cluster_packing(rotation_deg)
    if not np.isfinite(s) or s < math.sqrt(12) - 1e-9:
        return s + 100.0
    if not _is_valid(inner, outer, s):
        return s + 100.0
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0
    
    best_state = {'inner': None, 'outer': None, 's': float('inf')}
    
    def _update_best(inner, outer, s):
        if s < best_state['s'] and _is_valid(inner, outer, s):
            best_state['inner'] = inner.copy()
            best_state['outer'] = outer.copy()
            best_state['s'] = s
    
    # Phase 1: Rigid cluster with rotation optimization
    # Try a range of rotations to find the best
    for rot in np.linspace(0, 60, 121):
        if time.time() > deadline - 30:
            break
        inner, outer, s = _rigid_cluster_packing(rot)
        _update_best(inner, outer, s)
    
    # Phase 2: Fine-tune with Nelder-Mead on the rotation parameter
    if time.time() < deadline - 60:
        try:
            from scipy.optimize import minimize
            # Start from the best rotation found in Phase 1
            # Reconstruct rotation from best_state
            if best_state['inner'] is not None:
                # The rotation is stored in inner[0, 2]
                initial_rot = best_state['inner'][0, 2]
            else:
                initial_rot = 0.0
            
            res = minimize(
                _rigid_cluster_objective,
                np.array([initial_rot]),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            inner, outer, s = _rigid_cluster_packing(res.x[0])
            _update_best(inner, outer, s)
        except Exception as e:
            pass
    
    # Phase 3: Try a few alternative cluster configurations
    if time.time() < deadline - 30:
        # Alternative: more symmetric arrangement
        d = 2 * math.sqrt(3)
        a1 = np.array([d, 0.0])
        a2 = np.array([d * math.cos(math.pi/3), d * math.sin(math.pi/3)])
        
        # Try different ring 2 selections
        alt_configs = [
            [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
             (2,0), (1,1), (-1,2), (-2,0), (0,-2)],
            [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
             (2,0), (1,1), (-1,2), (-2,0), (-1,-1)],
            [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
             (2,0), (1,1), (-1,2), (-2,0), (1,-2)],
        ]
        
        for config in alt_configs:
            if time.time() > deadline - 15:
                break
            
            centers_local = []
            for (m, n) in config:
                pos = m * a1 + n * a2
                centers_local.append(pos)
            centers_local = np.array(centers_local)
            
            centroid = centers_local.mean(axis=0)
            centers_local = centers_local - centroid
            
            for rot in np.linspace(0, 60, 61):
                if time.time() > deadline - 10:
                    break
                rot_rad = np.deg2rad(rot)
                c, s_val = math.cos(rot_rad), math.sin(rot_rad)
                R = np.array([[c, -s_val], [s_val, c]])
                centers_rotated = centers_local @ R.T
                
                inner = np.zeros((12, 3))
                for i in range(12):
                    inner[i, 0] = centers_rotated[i, 0]
                    inner[i, 1] = centers_rotated[i, 1]
                    inner[i, 2] = rot
                
                best_s = float('inf')
                best_outer_angle = 0.0
                for oa in np.linspace(0, 60, 61):
                    s_val = _compute_min_s(inner, oa, np.array([0.0, 0.0]))
                    if s_val < best_s:
                        best_s = s_val
                        best_outer_angle = oa
                
                outer = np.array([0.0, 0.0, best_outer_angle])
                _update_best(inner, outer, best_s)
    
    # Fallback: ensure we have a valid result
    if best_state['inner'] is None:
        # Use the initial rigid cluster at rotation 0
        inner, outer, s = _rigid_cluster_packing(0.0)
        return inner, outer, s
    
    return best_state['inner'], best_state['outer'], best_state['s']

# EVOLVE-BLOCK-END
