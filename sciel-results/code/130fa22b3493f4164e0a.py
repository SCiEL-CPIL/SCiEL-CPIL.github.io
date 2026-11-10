import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner):
    """
    Compute the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all vertices of all inner hexagons.
    Uses the support function: for a regular hexagon with side s centered at origin,
    the distance from center to each edge is s * sqrt(3)/2.
    A point (x,y) is inside if for all 6 edge normals n_k, dot(n_k, p) <= s * sqrt(3)/2.
    So s >= max_k dot(n_k, p) / (sqrt(3)/2) for all points p.
    """
    # Edge normals for a regular hexagon with angle 0 (vertices at angles 0,60,120,180,240,300)
    # Edges are between consecutive vertices. The inward normal to edge k points toward center.
    # For a hexagon with vertices at angles 0,60,...,300, edge from v_k to v_{k+1} has 
    # outward normal at angle (60k + 30) degrees.
    # Inward normal = -outward normal, so inward normal angle = 60k + 210 degrees.
    # Actually, let's think differently.
    
    # A regular hexagon with side s, center at origin, "flat-top" orientation (angle 0 means 
    # vertex at angle 0) has vertices at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    
    # The 6 edges have outward normals at angles 30, 90, 150, 210, 270, 330 degrees.
    # For a point to be inside, dot(outward_normal, point) <= s * sqrt(3)/2 for all 6 normals.
    
    # Outward normal angles: 30 + 60*i for i=0..5
    normal_angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_s = 0.0
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for n in normals:
                val = np.dot(n, v)
                required_s = val / (np.sqrt(3.0) / 2.0)
                if required_s > max_s:
                    max_s = required_s
    return max_s


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
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
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


def _compute_min_s(inner, outer_center=np.array([0.0, 0.0]), outer_angle=0.0):
    """
    Compute the minimum side length s such that all inner hexagon vertices 
    are inside the outer hexagon with given center and angle.
    """
    # Shift inner hexagons so outer center is at origin
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= outer_center[0]
    inner_shifted[:, 1] -= outer_center[1]
    
    # For an outer hexagon with center at origin, side s, angle a_outer:
    # Vertices at angles a_outer + 60*i, i=0..5
    # Edges connect consecutive vertices. The outward normal to edge k is at angle 
    # a_outer + 60*k + 30 degrees.
    # A point p is inside if dot(outward_normal_k, p) <= s * sqrt(3)/2 for all k.
    # So s >= max_k dot(outward_normal_k, p) / (sqrt(3)/2) for all points p.
    
    a_outer = np.deg2rad(outer_angle)
    normal_angles = a_outer + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_s = 0.0
    for row in inner_shifted:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for n in normals:
                val = np.dot(n, v)
                required_s = val / (np.sqrt(3.0) / 2.0)
                if required_s > max_s:
                    max_s = required_s
    return max_s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Better initial guess: hexagonal arrangement with proper spacing
    # For unit hexagons with flat-top orientation, adjacent hexagons in a row are 
    # 3 units apart (2 * 1.5), and rows are sqrt(3) * 1.5 apart vertically.
    # Let's use a 3-4-3-3-4-3 pattern or similar.
    
    # Hexagon with side 1, flat-top: width = 3, height = sqrt(3)
    # Adjacent in same row: dx = 3
    # Adjacent in next row: dx = 1.5, dy = sqrt(3) * 1.5 = 2.598
    
    sqrt3 = np.sqrt(3.0)
    
    # Try a symmetric arrangement: center + 6 around + 6 more
    # Row-based: let's try 4-5-4 or 3-4-3-4-3 patterns
    
    # For 12 hexagons, a good arrangement is:
    # Row 0 (y=0): 4 hexagons at x = -4.5, -1.5, 1.5, 4.5
    # Row 1 (y=2.598): 4 hexagons at x = -3, 0, 3, 6 -> no, let me think more carefully
    
    # Actually, for flat-top hexagons:
    # - Same row: centers 3 apart
    # - Adjacent row: centers offset by 1.5 in x, sqrt(3)*1.5 in y
    
    # Let's use a hexagonal close-packed arrangement:
    # Center at origin
    # 6 neighbors at distance 3 in 6 directions (angles 0, 60, 120, 180, 240, 300)
    # Then 6 more at the "second shell" positions
    
    # First shell (6 hexagons): distance 3 from center
    # Angles: 0, 60, 120, 180, 240, 300
    # Positions: (3*cos(a), 3*sin(a)) for a in [0,60,120,180,240,300]
    
    # But wait - for flat-top hexagons, the "natural" neighbors are at:
    # (±3, 0), (±1.5, ±2.598)
    # That's 6 neighbors at distance 3.
    
    # Second shell: 6 more hexagons. These would be at:
    # (±4.5, 0), (±3, ±2.598), (0, ±5.196)
    # But that's 6 positions. Let's verify distances.
    
    # Actually, let me just use a known good initial configuration and let optimization refine it.
    
    # Symmetric 12-hexagon arrangement:
    inner = np.array([
        [0.0, 0.0, 0.0],           # Center
        [3.0, 0.0, 0.0],           # Right
        [-3.0, 0.0, 0.0],          # Left
        [1.5, 2.5980762, 0.0],     # Upper right
        [-1.5, 2.5980762, 0.0],    # Upper left
        [1.5, -2.5980762, 0.0],    # Lower right
        [-1.5, -2.5980762, 0.0],   # Lower left
        [4.5, 2.5980762, 0.0],     # Far upper right
        [-4.5, 2.5980762, 0.0],    # Far upper left
        [4.5, -2.5980762, 0.0],    # Far lower right
        [-4.5, -2.5980762, 0.0],   # Far lower left
        [0.0, 5.1961524, 0.0],     # Top
    ])
    
    outer = np.array([0.0, 0.0, 0.0])
    
    # Compute the minimum s needed
    s = _compute_min_s(inner, outer, 0.0)
    # Add a small margin
    s += 0.01
    
    # Validate
    valid, msg = _check_validity(inner, outer, s)
    if not valid:
        print(f"Initial guess invalid: {msg}")
        # Try larger s
        s += 1.0
        valid, msg = _check_validity(inner, outer, s)
        if not valid:
            # Fallback
            s = 20.0
            return inner, outer, s
    
    best_s = s
    best_inner = inner.copy()
    best_outer = outer.copy()
    print(f"Initial: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    # Optimize: only optimize inner positions (x, y, angle), compute s from them
    try:
        def objective(params):
            """
            params: 36 values: 12*(x, y, angle_deg)
            Returns: computed s + penalty for overlaps
            """
            inner_p = params.reshape(12, 3)
            
            # Compute s needed
            s_val = _compute_min_s(inner_p, np.array([0.0, 0.0]), 0.0)
            
            # Penalty for overlaps
            penalty = 0.0
            inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner_p]
            
            for i in range(12):
                for j in range(i + 1, 12):
                    v1 = inner_verts[i]
                    v2 = inner_verts[j]
                    # SAT check for overlap
                    edges = []
                    for k in range(6):
                        edges.append(v1[(k + 1) % 6] - v1[k])
                        edges.append(v2[(k + 1) % 6] - v2[k])
                    
                    separated = False
                    for edge in edges:
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-12:
                            continue
                        normal /= norm
                        
                        proj1 = v1 @ normal
                        proj2 = v2 @ normal
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        gap1 = min2 - max1
                        gap2 = min1 - max2
                        if gap1 > 1e-6 or gap2 > 1e-6:
                            separated = True
                            break
                    
                    if not separated:
                        # Overlapping - add penalty based on how much they overlap
                        # Use center distance as a proxy
                        d = np.linalg.norm(inner_p[i, :2] - inner_p[j, :2])
                        if d < 3.0:
                            penalty += (3.0 - d) ** 2 * 100.0
            
            return s_val + penalty
        
        x0 = inner.flatten()
        
        # Use Nelder-Mead for optimization
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-7, 'fatol': 1e-7,
                                   'adaptive': True})
        
        if result.fun < best_s:
            inner_opt = result.x.reshape(12, 3)
            s_opt = _compute_min_s(inner_opt, np.array([0.0, 0.0]), 0.0) + 0.001
            outer_opt = np.array([0.0, 0.0, 0.0])
            
            valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
            if valid:
                best_inner = inner_opt
                best_outer = outer_opt
                best_s = s_opt
                print(f"Optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
            else:
                print(f"Optimized result invalid: {msg}")
        
        if time.time() > deadline:
            return best_inner, best_outer, best_s
        
        # Second round of optimization with tighter tolerances
        if time.time() < deadline - 120:
            result2 = minimize(objective, result.x, method='Nelder-Mead',
                              options={'maxiter': 15000, 'xatol': 1e-8, 'fatol': 1e-8,
                                       'adaptive': True})
            
            if result2.fun < best_s:
                inner_opt = result2.x.reshape(12, 3)
                s_opt = _compute_min_s(inner_opt, np.array([0.0, 0.0]), 0.0) + 0.001
                outer_opt = np.array([0.0, 0.0, 0.0])
                
                valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
                if valid:
                    best_inner = inner_opt
                    best_outer = outer_opt
                    best_s = s_opt
                    print(f"Further optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Final validation
    valid, msg = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        print(f"Final result invalid: {msg}, using fallback")
        s = 20.0
        return best_inner, best_outer, s
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
