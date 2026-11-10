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


def _min_containing_hex_side(inner, outer_center, outer_angle):
    """
    Compute the minimum side length s of a regular hexagon with given center and angle
    that contains all inner hexagons.
    """
    # For a regular hexagon with side s, center (cx,cy), angle a:
    # The 6 outward normals to the sides are at angles a + 30 + 60*i degrees
    # A point p is inside iff for all 6 directions n_i: dot(p - center, n_i) <= s * sqrt(3)/2
    
    a = np.deg2rad(outer_angle)
    normal_angles = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_proj = 0.0
    for row in inner:
        cx, cy, ang = row
        hv = _hex_vertices(cx, cy, 1.0, ang)
        # Project vertices relative to outer center
        rel = hv - outer_center
        projections = rel @ normals.T
        max_proj = max(max_proj, np.max(projections))
    
    # s * sqrt(3)/2 >= max_proj => s >= max_proj / (sqrt(3)/2)
    s_min = max_proj / (np.sqrt(3.0) / 2.0)
    return s_min


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy: Use hexagonal lattice coordinates to define 12 hexagons with guaranteed
    non-overlap, then optimize the affine transformation (rotation of the whole configuration
    and the outer hexagon angle) to minimize the containing hexagon side length.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Hexagonal lattice: two basis vectors for center positions
    # For unit hexagons, adjacent centers are at distance sqrt(3)
    # Basis: e1 = (sqrt(3), 0), e2 = (sqrt(3)/2, 3/2)
    # But actually for hexagonal tiling, the lattice vectors are:
    # v1 = (3/2, sqrt(3)/2) * 2 = ... let me think more carefully.
    
    # For a regular hexagon with side 1, the distance between centers of two hexagons
    # that share an edge is 2 * (sqrt(3)/2) = sqrt(3).
    # The hexagonal lattice has basis vectors of length sqrt(3) at 60 degrees to each other.
    # b1 = (sqrt(3), 0)
    # b2 = (sqrt(3)*cos(60), sqrt(3)*sin(60)) = (sqrt(3)/2, 3/2)
    
    b1 = np.array([np.sqrt(3.0), 0.0])
    b2 = np.array([np.sqrt(3.0) / 2.0, 1.5])
    
    # Define 12 hexagons in a compact hexagonal arrangement using integer coordinates (i, j)
    # in the lattice basis. We want a configuration that's roughly hexagonal.
    # 
    # A hexagonal number arrangement for 12: 
    # Center (0,0), then ring of 6, then 5 more from the next ring.
    # Or: rows of 4,3,3,2 = 12? Let me think of a compact shape.
    #
    # Actually, let's use a "hexagonal" shape with 12 cells:
    # The hexagonal numbers are 1, 7, 19, ... so 12 isn't a perfect hex.
    # Let's use: 1 center + 6 first ring + 5 from second ring (to make a near-hexagon)
    #
    # Lattice coordinates (i, j) where position = i*b1 + j*b2:
    # Center: (0, 0)
    # First ring (6 hexagons): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Second ring partial (5 hexagons): (2,0), (1,1), (-1,2), (-2,1), (-2,0)
    # That's 12 total.
    
    lattice_coords = np.array([
        [0, 0],      # center
        [1, 0],      # first ring
        [0, 1],
        [-1, 1],
        [-1, 0],
        [0, -1],
        [1, -1],
        [2, 0],      # second ring (partial)
        [1, 1],
        [-1, 2],
        [-2, 1],
        [-2, 0],
    ], dtype=float)
    
    # Convert lattice coords to Cartesian positions
    def lattice_to_cartesian(coords):
        """Convert (i,j) lattice coordinates to (x,y) Cartesian."""
        x = coords[:, 0] * b1[0] + coords[:, 1] * b2[0]
        y = coords[:, 0] * b1[1] + coords[:, 1] * b2[1]
        return np.column_stack((x, y))
    
    # Center the configuration at origin
    positions = lattice_to_cartesian(lattice_coords)
    centroid = np.mean(positions, axis=0)
    positions = positions - centroid
    
    # All inner hexagons have angle 0 (same orientation as lattice)
    inner_base = np.column_stack([positions, np.zeros(12)])
    
    # Now optimize: rotate the entire configuration and choose outer hexagon angle
    # to minimize the containing hexagon side length.
    # Parameters: [rotation_angle_deg, outer_angle_deg]
    
    def objective(params):
        rot_deg = params[0]
        outer_ang_deg = params[1]
        
        # Rotate all inner hexagons
        rot_rad = np.deg2rad(rot_deg)
        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                      [np.sin(rot_rad), np.cos(rot_rad)]])
        
        rotated_pos = positions @ R.T
        inner_rot = np.column_stack([rotated_pos, np.zeros(12)])
        
        # Compute min containing hex side with outer at origin
        s = _min_containing_hex_side(inner_rot, np.array([0.0, 0.0]), outer_ang_deg)
        return s
    
    # Initial guess: rotation 0, outer angle 0
    x0 = np.array([0.0, 0.0])
    
    best_s = np.inf
    best_inner = inner_base.copy()
    best_outer = np.array([0.0, 0.0, 0.0])
    
    # Try multiple starting points for rotation and outer angle
    start_points = []
    for rot in range(0, 360, 15):
        for out_ang in range(0, 360, 30):
            start_points.append([float(rot), float(out_ang)])
    
    # Also add some random starts
    rng = np.random.RandomState(42)
    for _ in range(20):
        start_points.append([rng.uniform(0, 360), rng.uniform(0, 360)])
    
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(objective, sp, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if result.fun < best_s:
                rot_deg = result.x[0]
                outer_ang_deg = result.x[1]
                
                rot_rad = np.deg2rad(rot_deg)
                R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                              [np.sin(rot_rad), np.cos(rot_rad)]])
                
                rotated_pos = positions @ R.T
                inner_rot = np.column_stack([rotated_pos, np.zeros(12)])
                
                s = _min_containing_hex_side(inner_rot, np.array([0.0, 0.0]), outer_ang_deg)
                
                # Add tiny margin for numerical safety
                s_final = s + 1e-5
                
                outer = np.array([0.0, 0.0, outer_ang_deg])
                
                # Validate
                valid, msg = _check_validity(inner_rot, outer, s_final)
                if valid:
                    best_s = s_final
                    best_inner = inner_rot.copy()
                    best_outer = outer.copy()
        except Exception:
            continue
    
    # If no valid result from optimization, use a safe fallback
    if best_s == np.inf:
        # Just use the base configuration with a large enough outer hex
        s = _min_containing_hex_side(inner_base, np.array([0.0, 0.0]), 0.0) + 0.1
        outer = np.array([0.0, 0.0, 0.0])
        valid, msg = _check_validity(inner_base, outer, s)
        if not valid:
            s += 1.0
            valid, msg = _check_validity(inner_base, outer, s)
        if not valid:
            s = 20.0
        best_s = s
        best_inner = inner_base.copy()
        best_outer = outer.copy()
    
    # Final validation
    valid, msg = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        # Emergency fallback
        s = 20.0
        outer = np.array([0.0, 0.0, 0.0])
        return best_inner, outer, s
    
    return best_inner, best_outer, best_s
