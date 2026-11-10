import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner, outer_angle_deg):
    """
    Given 12 inner hexagons (12,3) array of (x, y, angle), find minimum outer hexagon
    side length centered at origin with given angle that contains all inner hexagons.
    """
    all_vertices = []
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        all_vertices.append(hv)
    
    all_vertices = np.vstack(all_vertices)  # (72, 2)
    
    theta = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        angle = theta + np.deg2rad(30.0) + np.deg2rad(60.0 * k)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)  # (6, 2)
    
    apothem = np.sqrt(3.0) / 2.0
    max_s = 0.0
    for v in all_vertices:
        for n in normals:
            val = np.dot(v, n)
            if val > 0:
                s_needed = val / apothem
                if s_needed > max_s:
                    max_s = s_needed
    
    max_s = max(max_s, np.sqrt(12.0))
    return max_s


def _check_separation(hv1, hv2, tol=1e-6):
    """Check if two hexagons are separated (non-overlapping) using SAT."""
    for i in range(6):
        edge = hv1[(i + 1) % 6] - hv1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        
        proj1 = hv1 @ normal
        proj2 = hv2 @ normal
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap1 = min2 - max1
        gap2 = min1 - max2
        if gap1 > tol or gap2 > tol:
            return True
    
    for i in range(6):
        edge = hv2[(i + 1) % 6] - hv2[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        
        proj1 = hv1 @ normal
        proj2 = hv2 @ normal
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap1 = min2 - max1
        gap2 = min1 - max2
        if gap1 > tol or gap2 > tol:
            return True
    
    return False


def _check_validity(inner, outer, s, tol=1e-6):
    """Validate packing; return True if valid."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12.0) - 1e-12:
        return False
    
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False
    
    # Check unit edge length
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for i in range(6):
            edge_len = np.linalg.norm(hv[(i + 1) % 6] - hv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check inner-inner separation
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts[i], inner_verts[j], tol):
                return False
    
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
                if cross < -tol:
                    return False
    
    return True


def _total_s(inner, outer_angle_deg):
    """Objective: minimize outer hexagon side length."""
    return _min_containing_hex_side(inner, outer_angle_deg)


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy: 
    1. Start with a hexagonal lattice arrangement
    2. Optimize inner positions and outer angle jointly
    3. Use multiple random restarts
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Leave 40s safety margin
    
    sqrt3 = np.sqrt(3.0)
    
    # Initial configuration: hexagonal cluster
    # Center + 6 neighbors + 5 more arranged symmetrically
    # For unit hexagons with angle 0 (flat-top), minimum center distance for non-overlap:
    # The Minkowski sum of two unit hexagons (same orientation) is a hexagon with side 2,
    # inradius = sqrt(3). So centers must be at distance >= sqrt(3) in all 6 normal directions.
    # For a hexagonal lattice with spacing d, adjacent centers are at distance d.
    # We need d >= 2*sqrt(3)/2 = sqrt(3) for same-orientation hexagons? 
    # Actually let me think again: two unit hexagons sharing an edge have center distance = 2*apothem = sqrt(3).
    # So for non-overlap with a small gap, use spacing slightly more than sqrt(3).
    
    # Hexagonal lattice positions (axial coordinates)
    # For 12 hexagons, use a compact arrangement:
    # Ring 0: center (0,0)
    # Ring 1: 6 hexagons at distance 1 (axial)
    # Ring 2: 5 hexagons at distance 2 (axial), chosen to fill gaps
    
    # Axial to Cartesian for flat-top hexagons:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # Wait, that's for pointy-top. For flat-top:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    # Let me use a simpler approach: just place them on a grid and verify
    
    # Flat-top hexagon: width = 2, height = sqrt(3)
    # Horizontal neighbors: center distance = 2
    # Vertical neighbors (offset): center distance = sqrt(3) in y, 1 in x
    
    # Let me try a 3-4-3-2 arrangement or similar
    
    # Configuration 1: Hexagonal cluster
    # Use pixel coordinates for flat-top hexagons
    # Adjacent horizontal: dx = 2, dy = 0
    # Adjacent diagonal: dx = 1, dy = sqrt(3)
    
    config1 = [
        (0, 0),      # center
        (2, 0), (-2, 0),  # horizontal neighbors
        (1, sqrt3), (-1, sqrt3), (1, -sqrt3), (-1, -sqrt3),  # diagonal
        (4, 0), (-4, 0),  # next horizontal
        (2, 2*sqrt3), (-2, 2*sqrt3), (2, -2*sqrt3), (-2, -2*sqrt3),  # next diagonal
    ]
    
    # But we need to verify these don't overlap. Let me use a larger spacing to be safe.
    # Use scale factor 1.05 for initial safety
    
    def make_config(spacing_scale=1.0):
        """Generate hexagonal cluster configuration."""
        s = spacing_scale
        positions = [
            (0, 0),
            (2*s, 0), (-2*s, 0),
            (1*s, sqrt3*s), (-1*s, sqrt3*s), (1*s, -sqrt3*s), (-1*s, -sqrt3*s),
            (4*s, 0), (-4*s, 0),
            (2*s, 2*sqrt3*s), (-2*s, 2*sqrt3*s), (2*s, -2*sqrt3*s), (-2*s, -2*sqrt3*s),
        ]
        return positions
    
    best_result = None
    best_s = float('inf')
    
    # Try multiple initial configurations and optimizations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Different spacing scales
        if trial == 0:
            spacing_scale = 1.0
        elif trial == 1:
            spacing_scale = 1.01
        elif trial == 2:
            spacing_scale = 0.99
        elif trial < 10:
            spacing_scale = 1.0 + (trial - 2) * 0.02
        else:
            spacing_scale = 1.0 + np.random.uniform(-0.05, 0.1)
        
        positions = make_config(spacing_scale)
        
        # Center the configuration
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx_mean = (min(xs) + max(xs)) / 2.0
        cy_mean = (min(ys) + max(ys)) / 2.0
        
        inner = np.array([[x - cx_mean, y - cy_mean, 0.0] for x, y in positions])
        
        # Verify no overlaps
        inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
        valid_separation = True
        for i in range(12):
            for j in range(i + 1, 12):
                if not _check_separation(inner_verts[i], inner_verts[j], 0.0):
                    valid_separation = False
                    break
            if not valid_separation:
                break
        
        if not valid_separation:
            continue
        
        # Optimize outer angle
        best_angle_for_this = 0.0
        best_s_for_this = float('inf')
        
        # Coarse search
        for angle in np.linspace(0, 60, 61):
            s_cand = _min_containing_hex_side(inner, angle)
            if s_cand < best_s_for_this:
                best_s_for_this = s_cand
                best_angle_for_this = angle
        
        # Fine search
        angle_range = np.linspace(best_angle_for_this - 1, best_angle_for_this + 1, 21)
        angle_range = angle_range % 60
        for angle in angle_range:
            s_cand = _min_containing_hex_side(inner, angle)
            if s_cand < best_s_for_this:
                best_s_for_this = s_cand
                best_angle_for_this = angle
        
        # Verify the result
        outer = np.array([0.0, 0.0, best_angle_for_this])
        if _check_validity(inner, outer, best_s_for_this):
            if best_s_for_this < best_s:
                best_s = best_s_for_this
                best_result = (inner.copy(), outer.copy(), best_s_for_this)
        
        # Now try to optimize inner positions
        if time.time() > deadline - 30:
            break
        
        # Parameterize: 24 positions + 12 angles + 1 outer angle = 37 params
        # But we can fix angles at 0 for simplicity and just optimize positions
        
        def objective(params):
            """Objective function: return outer hexagon side length."""
            # Extract positions (24) and angles (12)
            pos = params[:24].reshape(12, 2)
            angs = params[24:36]
            outer_angle = params[36]
            
            # Check separation quickly (if any overlap, return large penalty)
            for i in range(12):
                for j in range(i + 1, 12):
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist < sqrt3 * 0.99:  # Quick check
                        return 100.0 + dist
                    # More careful check with SAT
                    hv1 = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, angs[i])
                    hv2 = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, angs[j])
                    if not _check_separation(hv1, hv2, 0.0):
                        return 100.0 + dist
            
            inner_opt = np.column_stack([pos.flatten(), angs])
            return _min_containing_hex_side(inner_opt, outer_angle)
        
        # Initial parameters
        pos_init = inner[:, :2].flatten()
        ang_init = inner[:, 2]
        outer_ang_init = best_angle_for_this
        x0 = np.concatenate([pos_init, ang_init, [outer_ang_init]])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if result.fun < best_s_for_this and result.fun < 100:
                pos_opt = result.x[:24].reshape(12, 2)
                ang_opt = result.x[24:36]
                outer_ang_opt = result.x[36] % 60
                
                inner_opt = np.column_stack([pos_opt.flatten(), ang_opt])
                outer_opt = np.array([0.0, 0.0, outer_ang_opt])
                
                if _check_validity(inner_opt, outer_opt, result.fun):
                    if result.fun < best_s:
                        best_s = result.fun
                        best_result = (inner_opt.copy(), outer_opt.copy(), result.fun)
        except Exception:
            pass
    
    # If no valid result found, use a safe fallback
    if best_result is None:
        # Use a very conservative configuration
        positions = make_config(1.5)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx_mean = (min(xs) + max(xs)) / 2.0
        cy_mean = (min(ys) + max(ys)) / 2.0
        inner = np.array([[x - cx_mean, y - cy_mean, 0.0] for x, y in positions])
        
        # Find minimum containing hex
        best_angle = 0.0
        best_s = float('inf')
        for angle in np.linspace(0, 60, 121):
            s_cand = _min_containing_hex_side(inner, angle)
            if s_cand < best_s:
                best_s = s_cand
                best_angle = angle
        
        outer = np.array([0.0, 0.0, best_angle])
        # Add safety margin
        best_s *= 1.01
        
        return inner, outer, best_s
    
    return best_result
