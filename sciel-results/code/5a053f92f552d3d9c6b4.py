# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts[i, 0] = cx + d * math.cos(theta)
        verts[i, 1] = cy + d * math.sin(theta)
    return verts

def _check_separation(hex1, hex2):
    """Check if two hexagons (each (6,2) arrays) are separated by SAT."""
    for verts in (hex1, hex2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            # normal to edge
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            proj1 = hex1 @ n
            proj2 = hex2 @ n
            if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                return True
    return False

def _check_inside(hex_verts, outer_verts):
    """Check if all vertices of hex are inside outer hexagon."""
    for v in hex_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _is_valid(inner_data, outer_data, s):
    """Full validity check."""
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(v)
        # Check inside outer
        if not _check_inside(v, outer_verts):
            return False
    
    # Check pairwise separation
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Start with the seed arrangement
    inner = np.array([
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
        [0, -4, 0]
    ])
    outer = np.array([0, 0, 0])
    s = 8.0
    
    # Try to find a tighter packing by optimizing the outer hexagon size
    # First, let's try to center the inner hexagons and find the minimal enclosing hexagon
    
    # Compute the minimal enclosing regular hexagon for a given set of points
    def _min_enclosing_hexagon(points, angle_deg=0):
        """Find minimal side length s such that all points fit in a hexagon of side s centered at origin."""
        a = math.radians(angle_deg)
        # The hexagon has vertices at angles a + 60*i
        # A point (x,y) is inside if for all edges, it's on the correct side
        # For a regular hexagon centered at origin with side s and angle a:
        # The distance from center to each edge (apothem) is s * sqrt(3)/2
        # The edges are at angles a + 30 + 60*i (perpendicular to vertex directions)
        # Actually, for a regular hexagon with vertices at angles a+60i, the edges are perpendicular
        # to directions a+30+60i
        # The half-width in direction theta is s * sqrt(3)/2 / cos(angle_diff) where angle_diff is 
        # the angle between theta and the nearest edge normal... 
        
        # Simpler: for each edge, the constraint is that the projection of the point onto the 
        # edge normal must be <= s * sqrt(3)/2
        # Edge normals are at angles a + 30 + 60*i for i=0..5
        apothem = s * math.sqrt(3) / 2
        normals = []
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        # For each point, the required s is such that for all normals n:
        # point @ n <= s * sqrt(3)/2
        # So s >= max over all points and normals of (point @ n) / (sqrt(3)/2)
        # But we also need the negative direction: point @ (-n) <= s*sqrt(3)/2
        # i.e., point @ n >= -s*sqrt(3)/2
        # So s >= max over all points and normals of |point @ n| / (sqrt(3)/2)
        
        max_proj = 0
        for p in points:
            for n in normals:
                proj = abs(p @ n)
                max_proj = max(max_proj, proj)
        return max_proj / (math.sqrt(3) / 2)
    
    # Try different angles for the outer hexagon and different centering
    best_s = s
    best_inner = inner.copy()
    best_outer = outer.copy()
    
    # Try to optimize: shift inner hexagons and rotate outer
    # The inner hexagons are fixed relative to each other (all at angle 0)
    # We can shift them and rotate the outer hexagon
    
    # Get the center of mass of inner hexagons
    com = np.mean(inner[:, :2], axis=0)
    
    # Try various outer hexagon angles and shifts
    for angle_deg in np.arange(0, 60, 5):
        # Center the inner hexagons at origin
        shifted = inner.copy()
        shifted[:, :2] -= com
        
        # Find minimal s for this angle
        s_candidate = _min_enclosing_hexagon(shifted[:, :2], angle_deg)
        
        if s_candidate < best_s:
            best_s = s_candidate
            best_inner = shifted
            best_outer = np.array([0, 0, angle_deg])
    
    # Also try without recentering (keep original positions)
    for angle_deg in np.arange(0, 60, 5):
        s_candidate = _min_enclosing_hexagon(inner[:, :2], angle_deg)
        if s_candidate < best_s:
            best_s = s_candidate
            best_inner = inner.copy()
            best_outer = np.array([0, 0, angle_deg])
    
    # Now try local optimization: perturb inner hexagon positions slightly
    # and see if we can reduce s further
    current_inner = best_inner.copy()
    current_outer = best_outer.copy()
    current_s = best_s
    
    # Local search: try small perturbations
    np.random.seed(42)
    for iteration in range(2000):
        if time.time() > deadline - 10:
            break
        
        # Try a small random perturbation of one hexagon
        idx = np.random.randint(0, 12)
        dx = np.random.uniform(-0.1, 0.1)
        dy = np.random.uniform(-0.1, 0.1)
        
        trial_inner = current_inner.copy()
        trial_inner[idx, 0] += dx
        trial_inner[idx, 1] += dy
        
        # Check if still separated from others
        valid = True
        trial_verts = _hex_vertices(trial_inner[idx, 0], trial_inner[idx, 1], 1.0, trial_inner[idx, 2])
        for j in range(12):
            if j == idx:
                continue
            other_verts = _hex_vertices(trial_inner[j, 0], trial_inner[j, 1], 1.0, trial_inner[j, 2])
            if not _check_separation(trial_verts, other_verts):
                valid = False
                break
        
        if valid:
            # Check if it fits in the current outer hexagon
            outer_verts = _hex_vertices(current_outer[0], current_outer[1], current_s, current_outer[2])
            if _check_inside(trial_verts, outer_verts):
                current_inner = trial_inner
        
        # Also try rotating the outer hexagon slightly
        if iteration % 10 == 0:
            angle_offset = np.random.uniform(-1, 1)
            trial_angle = current_outer[2] + angle_offset
            # Recompute minimal s for this angle
            com = np.mean(current_inner[:, :2], axis=0)
            shifted = current_inner.copy()
            shifted[:, :2] -= com
            s_candidate = _min_enclosing_hexagon(shifted[:, :2], trial_angle)
            if s_candidate < current_s:
                # Check validity
                trial_outer = np.array([0, 0, trial_angle])
                if _is_valid(shifted, trial_outer, s_candidate):
                    current_inner = shifted
                    current_outer = trial_outer
                    current_s = s_candidate
    
    # Final validation
    if not _is_valid(current_inner, current_outer, current_s):
        # Fall back to seed
        current_inner = inner
        current_outer = outer
        current_s = s
    
    return current_inner, current_outer, current_s

# EVOLVE-BLOCK-END
