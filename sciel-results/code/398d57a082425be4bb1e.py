# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts.append((cx + d * math.cos(theta), cy + d * math.sin(theta)))
    return np.array(verts)

def _min_outer_side_for_inner(inner_centers, inner_angles, outer_center, outer_angle):
    """
    Compute the minimum side length of the outer hexagon (centered at outer_center, with given angle)
    such that all inner hexagons fit inside.
    We use the support function approach: for each direction, find the max distance from center.
    For a regular hexagon with side s, the support function in direction theta is s * f(theta)
    where f depends on the hexagon orientation.
    
    Actually, a simpler approach: for each inner vertex, check which outer edge it's closest to.
    But the cleanest way: the outer hexagon with side s has inradius = s * sqrt(3)/2.
    The maximum distance from center to any point inside is the circumradius = s.
    But we need all points to be inside, so we need the max over all inner vertices of their
    'hexagonal distance' from the outer center.
    
    For a regular hexagon with side s and orientation angle a, a point p is inside if for all 6 edges,
    the point is on the correct side. The distance from center to each edge (inradius) is s*sqrt(3)/2.
    
    Let me use a different approach: binary search on s, or compute directly.
    For a hexagon with center (0,0), side s, angle a, the constraint for point p is:
    For each edge i, dot(p, n_i) <= s * sqrt(3)/2, where n_i is the unit normal to edge i.
    The normals are at angles a + 30 + 60*i degrees (pointing outward from center to edge midpoint).
    Wait, let me think again.
    
    Vertices are at angles a + 60*i. Edge i connects vertex i and vertex i+1.
    The outward normal to edge i points in the direction of the midpoint of the edge, which is at angle a + 30 + 60*i.
    The distance from center to edge is s * cos(30 deg) = s * sqrt(3)/2.
    
    So for point p relative to outer center, the constraint is:
    dot(p, n_i) <= s * sqrt(3)/2 for all i=0..5
    where n_i = (cos(a + 30 + 60*i), sin(a + 30 + 60*i))
    
    So s >= max_i dot(p, n_i) / (sqrt(3)/2) = max_i dot(p, n_i) * 2/sqrt(3)
    And we need this for ALL inner vertices.
    """
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        normals.append((math.cos(theta), math.sin(theta)))
    normals = np.array(normals)
    
    max_s = 0.0
    for icx, icy, iang in inner_centers:
        verts = _hex_vertices(icx, icy, 1.0, iang)
        for v in verts:
            p = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            dots = normals @ p
            s_needed = np.max(dots) * 2.0 / math.sqrt(3)
            if s_needed > max_s:
                max_s = s_needed
    return max_s

def _sat_gap(verts1, verts2):
    """
    Separating Axis Test: return the minimum gap between two convex polygons.
    If they overlap, return negative value (penetration depth).
    If separated, return positive value (distance).
    """
    min_gap = float('inf')
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                continue
            normal = normal / norm_len
            
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap: if intervals overlap, gap is negative
            gap = max(min2 - max1, min1 - max2)
            if gap < min_gap:
                min_gap = gap
    return min_gap

def _check_validity(inner, outer, s, tol=1e-8):
    """Check if the packing is valid. Returns (is_valid, error_msg)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < math.sqrt(12) - 1e-9:
        return False, "s too small"
    
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False, "outer vertices not distinct"
    
    # Check inner hexagons don't overlap
    inner_verts = []
    for i in range(12):
        inner_verts.append(_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]))
    
    for i in range(12):
        for j in range(i+1, 12):
            gap = _sat_gap(inner_verts[i], inner_verts[j])
            if gap < -1e-6:
                return False, f"inner {i} and {j} overlap"
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        for v in inner_verts[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    return False, f"inner {i} vertex outside outer"
    
    return True, "ok"

def _compute_objective(inner, outer, s):
    """Compute the objective: 1/s, but penalize invalid configurations."""
    is_valid, msg = _check_validity(inner, outer, s)
    if not is_valid:
        return -1.0
    return 1.0 / s

def _objective_from_params(params):
    """
    params: flat array of 12*3 + 3 = 39 values
    First 36: 12 inner hexagons (x, y, angle_deg) each
    Last 3: outer hexagon (x, y, angle_deg)
    We compute s from the inner configuration.
    """
    inner = params[:36].reshape(12, 3)
    outer = params[36:39]
    s = _min_outer_side_for_inner(inner, inner[:,2], (outer[0], outer[1]), outer[2])
    # Ensure s is at least sqrt(12)
    s = max(s, math.sqrt(12))
    return -s  # We minimize s, so negative for maximization

def _sa_optimize(initial_params, time_budget=550.0):
    """
    Simulated annealing optimization.
    """
    start_time = time.time()
    n_params = 39
    T = 1.0
    T_min = 1e-6
    alpha = 0.9995
    best_params = initial_params.copy()
    best_score = -_objective_from_params(best_params)  # score = s, we minimize
    current_params = initial_params.copy()
    current_score = -_objective_from_params(current_params)
    
    # Precompute initial
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_budget:
        # Propose new solution
        step_size = T * 0.1
        new_params = current_params + rng.normal(0, step_size, size=n_params)
        new_score = -_objective_from_params(new_params)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < math.exp(-delta / max(T, T_min)):
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        
        T = max(T * alpha, T_min)
        
        # Reset temperature occasionally
        if T <= T_min * 1.1:
            T = T_min * 10
    
    return best_params, -best_score

def _seed_honeycomb():
    """
    Create a honeycomb-like seed arrangement of 12 unit hexagons.
    Honeycomb lattice: hexagons at positions (i * 3, j * 3*sqrt(3)/2) with offsets.
    Actually, for unit hexagons (side 1), the center-to-center distance in a honeycomb
    is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3) in one direction, and 3 in the other? 
    No. Let me think.
    
    A regular hexagon with side 1 has inradius sqrt(3)/2 and circumradius 1.
    In a honeycomb tiling, adjacent hexagons share edges. The distance between centers
    of adjacent hexagons is 2 * inradius = sqrt(3).
    
    The honeycomb lattice vectors: 
    v1 = (3, 0) -- wait, no.
    Actually, if hexagons have flat tops (angle 0), vertices at angles 0,60,120,180,240,300.
    The neighbors are at distance sqrt(3) in directions 30, 90, 150, 210, 270, 330 degrees.
    So lattice vectors: v1 = sqrt(3) * (cos30, sin30) = (1.5, sqrt(3)/2)
    v2 = sqrt(3) * (cos90, sin90) = (0, sqrt(3))
    
    Let me place 12 hexagons in a honeycomb pattern.
    A 3x4 arrangement or similar.
    """
    hexes = []
    # Use a hexagonal grid layout
    # Row-based: rows offset by half
    # Let's do a 3x4 grid in hexagonal coordinates
    dx = 3.0  # horizontal spacing for flat-top hexagons
    dy = math.sqrt(3)  # vertical spacing
    
    positions = []
    for row in range(4):
        for col in range(3):
            x = col * dx + (row % 2) * dx/2
            y = row * dy
            positions.append((x, y, 0.0))
    
    # Center them
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    
    inner = np.array([[p[0]-cx, p[1]-cy, p[2]] for p in positions])
    outer = np.array([0.0, 0.0, 0.0])
    
    params = np.concatenate([inner.flatten(), outer])
    return params

def _seed_compact():
    """
    A more compact seed: 12 hexagons in a tighter arrangement.
    Try a 2x6 or 3x4 with smaller spacing (allowing slight overlap initially, SA will fix).
    """
    positions = []
    dx = 2.9  # slightly less than 3 to encourage compaction
    dy = math.sqrt(3) * 0.95
    
    for row in range(4):
        for col in range(3):
            x = col * dx + (row % 2) * dx/2
            y = row * dy
            positions.append((x, y, 0.0))
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    
    inner = np.array([[p[0]-cx, p[1]-cy, p[2]] for p in positions])
    outer = np.array([0.0, 0.0, 0.0])
    
    params = np.concatenate([inner.flatten(), outer])
    return params

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    deadline = time.time() + 620.0  # 10s safety margin
    
    # Generate seeds
    seeds = [_seed_honeycomb(), _seed_compact()]
    
    best_params = None
    best_s = float('inf')
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Optimize
        params, s = _sa_optimize(seed, time_budget=min(280.0, deadline - time.time() - 10))
        
        if s < best_s:
            best_s = s
            best_params = params.copy()
    
    if best_params is None:
        # Fallback to seed
        best_params = seeds[0]
        best_s = -_objective_from_params(best_params)
    
    # Extract final configuration
    inner = best_params[:36].reshape(12, 3)
    outer = best_params[36:39]
    s = _min_outer_side_for_inner(inner, inner[:,2], (outer[0], outer[1]), outer[2])
    s = max(s, math.sqrt(12))
    
    # Verify validity
    is_valid, msg = _check_validity(inner, outer, s)
    if not is_valid:
        # Try to fix by using the seed configuration
        inner = seeds[0][:36].reshape(12, 3)
        outer = seeds[0][36:39]
        s = _min_outer_side_for_inner(inner, inner[:,2], (outer[0], outer[1]), outer[2])
        s = max(s, math.sqrt(12))
    
    return inner, outer, s

# EVOLVE-BLOCK-END
