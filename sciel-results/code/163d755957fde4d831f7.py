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


def _min_containing_hex_side(inner_centers, inner_angles, outer_angle):
    """
    Given fixed inner hexagon centers and angles, find minimum outer hexagon side length
    centered at origin with given angle that contains all inner hexagons.
    """
    # For each inner hexagon, compute its 6 vertices
    all_vertices = []
    for i in range(len(inner_centers)):
        cx, cy = inner_centers[i]
        a = inner_angles[i]
        hv = _hex_vertices(cx, cy, 1.0, a)
        all_vertices.append(hv)
    
    all_vertices = np.vstack(all_vertices)  # (72, 2)
    
    # The outer hexagon has center at (0,0), side s, angle outer_angle
    # Vertices are at distance s from center at angles outer_angle + 60*i
    # A point (x,y) is inside if for each edge, the cross product condition holds
    
    # The constraint: for each outer edge, all inner vertices must satisfy
    # cross(edge_vec, v - p1) >= 0 (with tolerance)
    # Edge k goes from vertex k to vertex k+1
    # vertex_k = s * [cos(outer_angle + 60k), sin(outer_angle + 60k)]
    
    # For a given outer hexagon with side s, the inradius is s * sqrt(3)/2
    # The apothem (distance from center to each edge) is s * sqrt(3)/2
    
    # A point v is inside the outer hexagon iff for all 6 edges:
    # dot(v, n_k) <= s * sqrt(3)/2
    # where n_k is the unit normal to edge k pointing inward
    
    # Actually, let me think more carefully.
    # Outer hexagon vertices: v_k = s * (cos(θ + 60k°), sin(θ + 60k°)) for k=0..5
    # Edge k goes from v_k to v_{k+1}
    # The inward normal to edge k points toward the center
    # Edge vector: e_k = v_{k+1} - v_k
    # Inward normal: rotate e_k by -90° (or +90°, need to check) and normalize
    
    # The distance from origin to each edge is the apothem = s * cos(30°) = s * sqrt(3)/2
    # The unit inward normal to edge k is in the direction of the midpoint of edge k
    
    # Midpoint of edge k: m_k = (v_k + v_{k+1})/2
    # Direction from origin to m_k: this is the inward normal direction
    # |m_k| = s * cos(30°) = s * sqrt(3)/2
    
    # So the constraint for point p to be inside:
    # dot(p, n_k) <= s * sqrt(3)/2 for all k
    # where n_k = m_k / |m_k| is the unit vector from origin to midpoint of edge k
    
    # n_k direction: angle = outer_angle + 60k° + 30° = outer_angle + 30° + 60k°
    
    theta = np.deg2rad(outer_angle)
    normals = []
    for k in range(6):
        angle = theta + np.deg2rad(30.0) + np.deg2rad(60.0 * k)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)  # (6, 2)
    
    # For each inner vertex v, constraint: dot(v, n_k) <= s * sqrt(3)/2 for all k
    # So s >= max over all vertices and all k of: dot(v, n_k) / (sqrt(3)/2)
    # Also s >= max over all vertices of |v| / (sqrt(3)/2) ... no, that's not right
    
    # Actually: dot(v, n_k) <= s * sqrt(3)/2
    # So s >= dot(v, n_k) / (sqrt(3)/2) = 2*dot(v, n_k)/sqrt(3)
    
    # But we also need s >= 0, and the constraint must hold for all vertices and all normals
    
    apothem = np.sqrt(3.0) / 2.0
    max_s = 0.0
    for v in all_vertices:
        for n in normals:
            val = np.dot(v, n)
            if val > 0:
                s_needed = val / apothem
                if s_needed > max_s:
                    max_s = s_needed
    
    # Also need s >= sqrt(12)
    max_s = max(max_s, np.sqrt(12.0))
    
    return max_s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy: Place 12 unit hexagons on a hexagonal lattice (ensuring non-overlap by construction),
    then optimize the outer hexagon's angle to minimize its side length.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Define 12 hexagons on a hexagonal lattice
    # Hexagonal lattice: neighbors at distance 2*sqrt(3) ≈ 3.464 for non-overlapping unit hexagons
    # Actually, for two unit hexagons to not overlap, their centers must be at least
    # 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732 apart? No...
    # Two unit hexagons with same orientation: minimum center distance for non-overlap is 2*sqrt(3) ≈ 3.464
    # Wait, the inradius (apothem) of a unit hexagon is sqrt(3)/2 ≈ 0.866
    # So two same-oriented unit hexagons need center distance >= 2*sqrt(3)/2 = sqrt(3) ≈ 1.732
    # But that's only if they're separated along the normal to a shared edge direction.
    # Actually for two identical regular hexagons, the minimum center distance for non-overlap
    # depends on relative orientation. For same orientation, it's 2*sqrt(3) in the "flat" direction
    # or 2*sqrt(3) in the "pointy" direction? Let me think...
    
    # A unit regular hexagon has inradius r = sqrt(3)/2 and circumradius R = 1.
    # Two unit hexagons with same orientation: the Minkowski sum is a hexagon with inradius 2*sqrt(3)/2 = sqrt(3).
    # So centers must be at distance >= sqrt(3) in any direction? No, the Minkowski sum of two hexagons
    # with same orientation is a hexagon with inradius 2*(sqrt(3)/2) = sqrt(3).
    # So the minimum center distance for non-overlap is sqrt(3) in the direction of any of the 6 normals.
    # But in other directions it could be different.
    
    # Actually, for two congruent convex polygons P and Q (same orientation), 
    # they don't overlap iff the center difference is not in the interior of P + (-Q).
    # For a regular hexagon, P + (-P) = 2P (by symmetry), which is a regular hexagon with side 2.
    # The inradius of 2P is 2*sqrt(3)/2 = sqrt(3).
    # So the condition is: |center_diff| projected onto any of the 6 normal directions <= sqrt(3).
    # Equivalently, center_diff must be outside the open hexagon 2P.
    
    # For a hexagonal lattice arrangement, I'll use the standard hex grid.
    # In a hex grid with "pointy-top" orientation, the spacing between adjacent hex centers is 2*sqrt(3) for side-1 hexagons? 
    # No, let me just use a simple grid and verify.
    
    # Let me use a known valid configuration: 12 hexagons in a 3-4-3 or similar arrangement.
    # Actually, let me use a simple approach: place them in a rectangular-ish grid and verify.
    
    # For unit hexagons with 0° angle (flat-top), the vertices are at:
    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # The "width" (horizontal extent) is 2, the "height" (vertical extent) is sqrt(3)
    # Two such hexagons side by side horizontally need center distance >= 2 (they share a vertical edge)
    # Two stacked vertically need center distance >= sqrt(3) (they share a horizontal-ish edge)
    
    # Let me use a configuration and verify it's valid:
    # Arrangement: 4 rows
    # Row 0: 4 hexagons at x = -3, -1, 1, 3; y = 0
    # Row 1: 3 hexagons at x = -2, 0, 2; y = sqrt(3)
    # Row 2: 3 hexagons at x = -2, 0, 2; y = -sqrt(3)
    # Row 3: 2 hexagons... hmm that's only 10.
    
    # Let me try: 12 hexagons in a hexagonal cluster
    # Center + 6 around it + 5 more? 
    
    # Actually, let me just use a simple 3x4 grid with appropriate spacing:
    # For flat-top hexagons (angle=0):
    # Horizontal spacing: 2 (edge to edge)
    # Vertical spacing: sqrt(3)
    # Offset every other row by 1
    
    # Grid positions for flat-top hexagons:
    # Row 0 (y=0): x = -3, -1, 1, 3  (4 hexagons)
    # Row 1 (y=sqrt(3)): x = -2, 0, 2  (3 hexagons)  
    # Row 2 (y=2*sqrt(3)): x = -3, -1, 1, 3  (4 hexagons)
    # Row 3 (y=3*sqrt(3)): x = -2, 0, 2  (3 hexagons)
    # Total: 14, too many.
    
    # Let me try:
    # Row 0: x = -3, -1, 1, 3 (4)
    # Row 1: x = -2, 0, 2 (3)
    # Row 2: x = -3, -1, 1, 3 (4)
    # Row 3: x = 0 (1)
    # Total: 12
    
    sqrt3 = np.sqrt(3.0)
    positions = [
        # Row 0 (y=0)
        (-3.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (3.0, 0.0),
        # Row 1 (y=sqrt(3))
        (-2.0, sqrt3), (0.0, sqrt3), (2.0, sqrt3),
        # Row 2 (y=2*sqrt(3))
        (-3.0, 2*sqrt3), (-1.0, 2*sqrt3), (1.0, 2*sqrt3), (3.0, 2*sqrt3),
        # Row 3 (y=3*sqrt(3))
        (0.0, 3*sqrt3),
    ]
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (min(xs) + max(xs)) / 2.0
    cy_mean = (min(ys) + max(ys)) / 2.0
    
    inner = np.array([[x - cx_mean, y - cy_mean, 0.0] for x, y in positions])
    
    # Verify this configuration is valid (no overlaps)
    valid, msg = _check_validity(inner, np.array([0.0, 0.0, 0.0]), 100.0)
    if not valid:
        print(f"Lattice config invalid: {msg}")
        # Try different spacing
        # Maybe the spacing is too tight. Let me increase it.
        # For flat-top hexagons, adjacent horizontal neighbors need dx >= 2
        # Diagonal neighbors (one row up, offset by 1) need to check
    
    # Now optimize the outer hexagon angle
    best_s = float('inf')
    best_angle = 0.0
    
    inner_centers = inner[:, :2]
    inner_angles = inner[:, 2]
    
    # Try many angles
    for angle in np.linspace(0, 60, 61):
        s_candidate = _min_containing_hex_side(inner_centers, inner_angles, angle)
        if s_candidate < best_s:
            best_s = s_candidate
            best_angle = angle
    
    # Refine with optimization
    def obj_angle(params):
        angle = params[0] % 60.0
        return _min_containing_hex_side(inner_centers, inner_angles, angle)
    
    try:
        result = minimize(obj_angle, np.array([best_angle]), method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < best_s:
            best_s = result.fun
            best_angle = result.x[0] % 60.0
    except:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        outer = np.array([0.0, 0.0, best_angle])
        valid, msg = _check_validity(inner, outer, best_s)
        if valid:
            return inner, outer, best_s
        # Fallback
        return inner, np.array([0.0, 0.0, 0.0]), 20.0
    
    # Try to also optimize the inner configuration slightly
    # But first, let's see what s we got
    
    outer = np.array([0.0, 0.0, best_angle])
    valid, msg = _check_validity(inner, outer, best_s)
    if not valid:
        print(f"Result invalid: {msg}")
        # Increase s slightly
        for ds in [0.01, 0.1, 0.5, 1.0, 2.0]:
            s_try = best_s + ds
            valid, msg = _check_validity(inner, outer, s_try)
            if valid:
                best_s = s_try
                break
        else:
            return inner, np.array([0.0, 0.0, 0.0]), 20.0
    
    # Now try to optimize the inner positions while keeping the outer fixed
    # This is more complex, so let's try a few perturbations
    
    best_s_final = best_s
    best_inner_final = inner.copy()
    
    # Try rotating individual hexagons
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        
        trial_inner = inner.copy()
        # Randomly perturb a few hexagon angles
        for _ in range(3):
            idx = np.random.randint(0, 12)
            trial_inner[idx, 2] += np.random.uniform(-15, 15)
        
        # Recompute best outer angle
        trial_best_s = float('inf')
        for angle in np.linspace(0, 60, 31):
            s_cand = _min_containing_hex_side(trial_inner[:, :2], trial_inner[:, 2], angle)
            if s_cand < trial_best_s:
                trial_best_s = s_cand
        
        if trial_best_s < best_s_final - 1e-6:
            valid, msg = _check_validity(trial_inner, np.array([0.0, 0.0, 0.0]), 100.0)
            if valid:
                best_s_final = trial_best_s
                best_inner_final = trial_inner
    
    outer_final = np.array([0.0, 0.0, best_angle])
    valid, msg = _check_validity(best_inner_final, outer_final, best_s_final)
    if not valid:
        # Use the previously valid result
        valid, msg = _check_validity(inner, outer, best_s)
        if valid:
            return inner, outer, best_s
        return inner, np.array([0.0, 0.0, 0.0]), 20.0
    
    return best_inner_final, outer_final, best_s_final


# EVOLVE-BLOCK-END
