# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

_UNIT_COS = np.cos(np.deg2rad(60 * np.arange(6)))
_UNIT_SIN = np.sin(np.deg2rad(60 * np.arange(6)))
_UNIT_VERTS = np.column_stack([_UNIT_COS, _UNIT_SIN])  # (6,2)

def _hex_edge_normals(angle_deg):
    """Compute the 6 outward unit edge normals for a regular hexagon."""
    a = np.deg2rad(angle_deg)
    normals_angles = a + np.deg2rad(30.0) + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])

def _min_enclosing_hex_side(centers, inner_angles, outer_angle_deg=0.0, outer_center=None):
    """
    Find the minimum side length s of a regular hexagon that contains all inner hexagons.
    centers: (12, 2) array of inner hex centers
    inner_angles: (12,) array of inner hex angles
    outer_angle_deg: angle of outer hexagon
    outer_center: (2,) array, defaults to origin
    Returns s (scalar)
    """
    if outer_center is None:
        outer_center = np.array([0.0, 0.0])
    
    outer_normals = _hex_edge_normals(outer_angle_deg)  # (6, 2)
    
    max_required_s = 0.0
    for i in range(12):
        inner_verts = _hex_vertices(0, 0, 1.0, inner_angles[i])  # (6, 2)
        rel_center = centers[i] - outer_center
        
        for n in outer_normals:
            # Support function: farthest projection of inner hex in direction n
            support = np.max(inner_verts @ n)
            proj = rel_center @ n + support
            # For a regular hexagon with side s, inradius = s * sqrt(3) / 2
            # So proj <= inradius = s * sqrt(3) / 2
            required_s = proj / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max(0.0, max_required_s)

def _check_separation(hex1_data, hex2_data, tol=1e-6):
    """Separating axis test: return True if hexagons are separated (no overlap)."""
    v1 = _hex_vertices(hex1_data[0], hex1_data[1], 1.0, hex1_data[2])
    v2 = _hex_vertices(hex2_data[0], hex2_data[1], 1.0, hex2_data[2])
    
    n1 = _hex_edge_normals(hex1_data[2])
    n2 = _hex_edge_normals(hex2_data[2])
    
    for normals in [n1, n2]:
        for normal in normals:
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 + tol < min2 or max2 + tol < min1:
                return True
    
    return False

def _vertex_inside_outer(v, outer_verts, tol=1e-6):
    """Check if vertex v is inside or on the boundary of outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -tol:
            return False
    return True

def _validate(inner_hex_data, outer_hex_data, s):
    """Validate the packing. Return True if valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-8:
        return False
    if not np.isfinite(s):
        return False
    
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        inner_verts_list.append(verts)
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_hex_data[i], inner_hex_data[j]):
                return False
    
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    
    return True

def _make_honeycomb_343():
    """Create 12 hexagon centers in a 3-4-3 honeycomb arrangement."""
    sqrt3 = np.sqrt(3)
    col_spacing = 1.5
    row_spacing = sqrt3
    
    positions = []
    # 3-4-3 honeycomb: 3 cols, 4 rows, 3 cols pattern
    # Actually let's do a proper hexagonal cluster
    # Center at (0,0)
    # Ring 0: center
    # Ring 1: 6 around
    # Ring 2: partial - we need 12 total so center + 6 + 5 from ring 2
    
    # Better: use a 4x3 grid with hexagonal offset
    positions = []
    for col in range(4):
        for row in range(3):
            x = col * col_spacing
            if col % 2 == 1:
                y = (row + 0.5) * row_spacing
            else:
                y = row * row_spacing
            positions.append([x, y])
    
    positions = np.array(positions)
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _make_honeycomb_hex2():
    """Create 12 hexagon centers: center + 6 ring + 5 from next ring (hexagonal packing)."""
    sqrt3 = np.sqrt(3)
    positions = [[0.0, 0.0]]
    
    # Ring 1: 6 positions
    for i in range(6):
        angle = np.deg2rad(60 * i + 30)
        positions.append([sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)])
    
    # Ring 2: pick 5 positions (out of 12) to make 12 total
    # For hexagonal packing, ring 2 has 12 positions
    for i in range(5):
        angle = np.deg2rad(60 * i + 60)
        positions.append([2 * sqrt3 * np.cos(angle), 2 * sqrt3 * np.sin(angle)])
    
    positions = np.array(positions)
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _make_rect_grid(rows, cols, dx, dy):
    """Create a rectangular grid of positions with hexagonal offset."""
    positions = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2.0) * dx
            y = (r - (rows - 1) / 2.0) * dy
            if r % 2 == 1:
                x += dx / 2
            positions.append([x, y])
    positions = np.array(positions[:12])
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _objective_full(params, deadline=None):
    """
    Full objective: minimize outer hexagon side length with overlap penalties.
    params: [24 center coords, 12 inner angles, 1 outer angle, 2 outer center] = 39 params
    """
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    inner_pos = params[:24].reshape(12, 2)
    inner_angles = params[24:36]
    outer_angle = params[36]
    outer_center = params[37:39]
    
    s = _min_enclosing_hex_side(inner_pos, inner_angles, outer_angle, outer_center)
    
    # Overlap penalty using SAT
    penalty = 0.0
    for i in range(12):
        v1 = _hex_vertices(inner_pos[i, 0], inner_pos[i, 1], 1.0, inner_angles[i])
        for j in range(i + 1, 12):
            v2 = _hex_vertices(inner_pos[j, 0], inner_pos[j, 1], 1.0, inner_angles[j])
            
            n1 = _hex_edge_normals(inner_angles[i])
            n2 = _hex_edge_normals(inner_angles[j])
            
            min_gap = np.inf
            separated = False
            
            for normals in [n1, n2]:
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = proj1.min(), proj1.max()
                    min2, max2 = proj2.min(), proj2.max()
                    
                    if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                        separated = True
                        break
                    gap = min(min2 - max1, min1 - max2)
                    if gap < min_gap:
                        min_gap = gap
                if separated:
                    break
            
            if not separated and min_gap < 0:
                penalty += min_gap ** 2
    
    return s + 500.0 * penalty

def _objective_rigid(params, deadline=None):
    """
    Rigid cluster objective: optimize only rotation, scale, and outer angle.
    params: [rotation_deg, scale, outer_angle_deg, outer_cx, outer_cy] = 5 params
    """
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    rot_deg = params[0]
    scale = params[1]
    outer_angle = params[2]
    outer_cx, outer_cy = params[3], params[4]
    
    # Base positions (3-4-3 honeycomb)
    base_pos = _make_honeycomb_343()
    
    # Apply rotation and scale
    rot = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    centers = (R @ base_pos.T).T * scale
    inner_angles = np.full(12, rot_deg)
    outer_center = np.array([outer_cx, outer_cy])
    
    s = _min_enclosing_hex_side(centers, inner_angles, outer_angle, outer_center)
    
    # Check overlaps
    penalty = 0.0
    for i in range(12):
        v1 = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, inner_angles[i])
        for j in range(i + 1, 12):
            v2 = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, inner_angles[j])
            
            n1 = _hex_edge_normals(inner_angles[i])
            n2 = _hex_edge_normals(inner_angles[j])
            
            min_gap = np.inf
            separated = False
            
            for normals in [n1, n2]:
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = proj1.min(), proj1.max()
                    min2, max2 = proj2.min(), proj2.max()
                    
                    if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                        separated = True
                        break
                    gap = min(min2 - max1, min1 - max2)
                    if gap < min_gap:
                        min_gap = gap
                if separated:
                    break
            
            if not separated and min_gap < 0:
                penalty += min_gap ** 2
    
    return s + 500.0 * penalty

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, s)
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    
    best_s = np.inf
    best_params_full = None
    best_params_rigid = None
    
    # Phase 1: Rigid cluster optimization (fast, 5D)
    # Try multiple initial rotations and scales
    rigid_inits = []
    for rot in [0, 15, 30, 45, 60, 90, 120, 150, 180]:
        for scale in [0.95, 1.0, 1.05, 1.1, 1.15, 1.2]:
            for outer_angle in [0, 30, 60]:
                rigid_inits.append([rot, scale, outer_angle, 0.0, 0.0])
    
    for init in rigid_inits:
        if time.time() > deadline - 120:
            break
        
        try:
            result = minimize(_objective_rigid, init, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                     'adaptive': True})
            
            val = result.fun
            if val < best_s and val < 100:  # Filter out bad solutions
                best_s = val
                best_params_rigid = result.x
        except:
            pass
    
    # Phase 2: Full optimization from rigid solution
    if best_params_rigid is not None and time.time() < deadline - 60:
        # Extract solution from rigid params
        rot_deg = best_params_rigid[0]
        scale = best_params_rigid[1]
        outer_angle = best_params_rigid[2]
        outer_cx, outer_cy = best_params_rigid[3], best_params_rigid[4]
        
        base_pos = _make_honeycomb_343()
        rot = np.deg2rad(rot_deg)
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        centers = (R @ base_pos.T).T * scale
        
        # Build full params
        full_params = np.zeros(39)
        full_params[:24] = centers.flatten()
        full_params[24:36] = np.full(12, rot_deg)
        full_params[36] = outer_angle
        full_params[37:39] = [outer_cx, outer_cy]
        
        # Polish with full optimization
        try:
            result = minimize(_objective_full, full_params, method='Nelder-Mead',
                            options={'maxiter': 15000, 'xatol': 1e-11, 'fatol': 1e-13,
                                     'adaptive': True})
            
            val = result.fun
            if val < best_s and val < 100:
                best_s = val
                best_params_full = result.x
        except:
            pass
    
    # Phase 3: Full optimization from diverse starts
    if time.time() < deadline - 30:
        diverse_inits = []
        
        # From 3-4-3 honeycomb with various rotations
        base = _make_honeycomb_343()
        for rot in [0, 30, 60, 90, 120, 150]:
            rot_rad = np.deg2rad(rot)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            centers = (R @ base.T).T
            
            params = np.zeros(39)
            params[:24] = centers.flatten()
            params[24:36] = np.full(12, rot)
            params[36] = rot  # outer angle same as inner
            params[37:39] = [0.0, 0.0]
            diverse_inits.append(params)
        
        # From hexagonal cluster
        base2 = _make_honeycomb_hex2()
        for rot in [0, 30, 60]:
            rot_rad = np.deg2rad(rot)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            centers = (R @ base2.T).T
            
            params = np.zeros(39)
            params[:24] = centers.flatten()
            params[24:36] = np.full(12, rot)
            params[36] = rot
            params[37:39] = [0.0, 0.0]
            diverse_inits.append(params)
        
        # Random perturbations
        rng = np.random.default_rng(42)
        for seed in range(8):
            rng_local = np.random.default_rng(seed)
            base_pos = _make_honeycomb_343()
            base_pos += rng_local.normal(0, 0.3, size=(12, 2))
            
            params = np.zeros(39)
            params[:24] = base_pos.flatten()
            params[24:36] = np.zeros(12)
            params[36] = 0.0
            params[37:39] = [0.0, 0.0]
            diverse_inits.append(params)
        
        for init_params in diverse_inits:
            if time.time() > deadline - 15:
                break
            
            try:
                result = minimize(_objective_full, init_params, method='Nelder-Mead',
                                options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-13,
                                         'adaptive': True})
                
                val = result.fun
                if val < best_s and val < 100:
                    best_s = val
                    best_params_full = result.x
            except:
                pass
    
    # Phase 4: Final high-precision polish
    if best_params_full is not None and time.time() < deadline - 5:
        try:
            result = minimize(_objective_full, best_params_full, method='Nelder-Mead',
                            options={'maxiter': 30000, 'xatol': 1e-12, 'fatol': 1e-14,
                                     'adaptive': True})
            
            val = result.fun
            if val < best_s and val < 100:
                best_s = val
                best_params_full = result.x
        except:
            pass
    
    # Extract best solution
    if best_params_full is not None:
        params = best_params_full
        inner_pos = params[:24].reshape(12, 2)
        inner_angles = params[24:36]
        outer_angle = params[36]
        outer_center = params[37:39]
    elif best_params_rigid is not None:
        # Reconstruct from rigid params
        rot_deg = best_params_rigid[0]
        scale = best_params_rigid[1]
        outer_angle = best_params_rigid[2]
        outer_center = best_params_rigid[3:5]
        
        base_pos = _make_honeycomb_343()
        rot = np.deg2rad(rot_deg)
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        inner_pos = (R @ base_pos.T).T * scale
        inner_angles = np.full(12, rot_deg)
    else:
        # Fallback: simple honeycomb
        inner_pos = _make_honeycomb_343()
        inner_angles = np.zeros(12)
        outer_angle = 0.0
        outer_center = np.array([0.0, 0.0])
    
    # Build return arrays
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = inner_pos[i, 0]
        inner_hex_data[i, 1] = inner_pos[i, 1]
        inner_hex_data[i, 2] = inner_angles[i]
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], float(outer_angle)])
    
    # Compute final s
    s_final = _min_enclosing_hex_side(inner_pos, inner_angles, outer_angle, outer_center)
    
    # Validate
    if not _validate(inner_hex_data, outer_hex_data, s_final):
        # Fallback to a known valid configuration
        inner_hex_data = np.zeros((12, 3))
        positions_fb = _make_honeycomb_343()
        for i in range(12):
            inner_hex_data[i, 0] = positions_fb[i, 0]
            inner_hex_data[i, 1] = positions_fb[i, 1]
            inner_hex_data[i, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s_final = _min_enclosing_hex_side(positions_fb, np.zeros(12), 0.0, np.array([0.0, 0.0])) + 0.5
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
