import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _get_axes(hex_verts):
    """Get the 6 edge normals (axes) of a hexagon."""
    axes = []
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        # Normal is perpendicular to edge
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
        axes.append(normal)
    return np.array(axes)

def _sat_overlap(v1, v2):
    """
    Check if two convex polygons overlap using Separating Axis Theorem.
    Returns True if they overlap (or touch), False if separated.
    """
    axes = np.vstack([_get_axes(v1), _get_axes(v2)])
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def _min_dist_polygons(v1, v2):
    """
    Calculate minimum distance between two convex polygons.
    Returns 0 if they overlap.
    """
    if _sat_overlap(v1, v2):
        return 0.0
        
    min_dist = np.inf
    # Distance from each vertex of v1 to edges of v2
    n1, n2 = len(v1), len(v2)
    
    def point_segment_dist(p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    for p in v1:
        for i in range(n2):
            d = point_segment_dist(p, v2[i], v2[(i+1)%n2])
            if d < min_dist:
                min_dist = d
                
    for p in v2:
        for i in range(n1):
            d = point_segment_dist(p, v1[i], v1[(i+1)%n1])
            if d < min_dist:
                min_dist = d
                
    return min_dist

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check containment: all inner vertices must be inside outer
    # We can use a point-in-polygon test or check against half-planes.
    # Since outer is convex, checking against its edge normals is robust.
    outer_axes = _get_axes(outer_verts)
    # Note: _get_axes returns outward normals if vertices are CCW.
    # Let's ensure orientation. Standard math: CCW.
    # For a convex polygon defined by CCW vertices, the interior is to the left of each edge.
    # So dot(v - v_i, normal) >= 0? 
    # Let's just use the projection method.
    # For each axis (normal to an edge), the projection of the inner point must be 
    # within the range of the outer polygon's projection on that axis.
    
    for iv in inner_verts:
        for v in iv:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                # Cross product to check side
                cross = (b[0]-a[0])*(v[1]-a[1]) - (b[1]-a[1])*(v[0]-a[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
                
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                return False, "overlap"
                
    return True, "valid"

def _objective(x, outer_angle):
    """
    Objective function for optimization.
    x: 3N vector of inner hexagon parameters (x1, y1, a1, x2, y2, a2, ...)
    outer_angle: angle of the outer hexagon (fixed during optimization step)
    Returns: (value, grad) -- but Nelder-Mead doesn't need grad.
    We return a scalar value.
    
    Strategy: 
    1. Compute the minimum outer side length required to contain all inner hexagons.
    2. Compute the maximum overlap between any pair of inner hexagons.
    3. Return a penalty if overlap > 0, otherwise the side length.
    
    To make this differentiable-ish for NM, we can return:
    max(0, max_overlap) * LARGE_PENALTY + side_length
    
    However, NM works on values. A "feasibility first" approach:
    If any overlap exists, return a huge value proportional to the overlap.
    This pushes the simplex to expand the cluster until it's valid, then minimizes size.
    """
    params = x.reshape(N, 3)
    centers = params[:, :2]
    angles = params[:, 2]
    
    inner_verts = []
    for i in range(N):
        v = _hex_vertices(centers[i], 1.0, angles[i])
        inner_verts.append(v)
        
    # 1. Check overlaps
    max_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # If they overlap, distance is 0. We want a measure of "how much" they overlap.
            # SAT doesn't give depth easily. 
            # Alternative: Use the minimum distance. If dist < 0 (overlap), we need depth.
            # For simplicity and speed, if SAT says overlap, we return a large penalty.
            # But NM needs gradients (differences in value).
            # Let's use a soft penalty based on vertex distances if they are close, 
            # or just a binary penalty which NM can handle if it's smooth enough.
            # Actually, let's compute the minimum distance. If < 0, it's overlap.
            # Since _min_dist_polygons returns 0 for overlap, it's not smooth.
            
            # Better: Compute the maximum distance of any vertex of i inside j? No.
            # Let's stick to the "Expand until valid" logic.
            # If SAT detects overlap, we return a value that encourages separation.
            # How to quantify separation needed?
            # We can use the minimum distance between vertices of i and edges of j, etc.
            # But if they intersect, it's complex.
            
            # Heuristic: If SAT says overlap, return a penalty based on the sum of squared distances of vertices of i to the centroid of j? No.
            
            # Let's use a different metric: 
            # For each pair, if they overlap, calculate the "penetration depth" approximation.
            # This is hard.
            
            # Alternative Strategy: 
            # The objective is the outer side length.
            # Constraints: No overlap.
            # We can use a penalty method:
            # Penalty = sum(max(0, -dist_ij)^2) where dist_ij is the minimum distance.
            # But _min_dist_polygons returns 0 for overlap.
            
            # Let's modify _min_dist_polygons to return a negative value for overlap?
            # Or better, let's just use the "Vertex-In-Polygon" check for a soft penalty.
            # If a vertex of hex i is inside hex j, that's a clear violation.
            # But edges can cross without vertices inside.
            
            # Given time constraints, let's use a robust but potentially non-smooth penalty:
            # If SAT says overlap, return 1000 + sum of squared distances of vertices of i to vertices of j?
            # This is noisy.
            
            # Let's try a different angle: 
            # Optimize positions such that the minimum distance between any two hexagons is >= 0.
            # We can define the objective as:
            # Side_Length + LAMBDA * max(0, -min_dist_global)^2
            # But min_dist is 0 for overlap.
            
            # Let's use the "Expansion" idea:
            # If the configuration is invalid, the "effective side length" is undefined/infinite.
            # Instead, let's return the side length required to contain the *expanded* hexagons?
            
            # Simplest robust approach for NM:
            # Value = Side_Length
            # If Invalid: Value += 100 * (Number of Overlaps) + 10 * (Sum of Vertex Distances to Centroids of Overlapping Pairs)
            # This creates a landscape where invalid states are high, and valid states are low.
            
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                # Calculate a penalty
                # Distance between centroids
                d_centroids = np.linalg.norm(centers[i] - centers[j])
                # Smaller distance = worse
                max_overlap = max(max_overlap, 10.0 - d_centroids)
                
    if max_overlap > 0:
        return max_overlap * 100.0 + _compute_side_length(inner_verts, outer_angle)
    else:
        return _compute_side_length(inner_verts, outer_angle)

def _compute_side_length(inner_verts, outer_angle):
    """
    Compute the minimum side length of the outer hexagon (centered at 0,0) 
    to contain all inner hexagons, given the outer angle.
    """
    max_proj = 0.0
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        # Project all vertices of all inner hexagons onto this normal
        # The outer hexagon boundary is at distance S * sqrt(3) / 2 from center.
        # So we need S * sqrt(3) / 2 >= max_projection
        for v in inner_verts:
            projs = np.dot(v, normal)
            max_proj = max(max_proj, np.max(projs))
            
    return max_proj * 2.0 / np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # We will try a few outer angles
    outer_angles = [0.0, 30.0]
    
    # Generate initial guesses
    initial_guesses = []
    
    # Guess 1: Hexagonal ring (Center + 6 + 5)
    pts = [[0,0,0]]
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle), 0.0])
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle), 0.0])
        if len(pts) == 12:
            break
    initial_guesses.append(np.array(pts).flatten())
    
    # Guess 2: Random
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.uniform(-3, 3, (N, 3))
        # Ensure angles are 0 or 30 or 60
        pts[:, 2] = np.random.choice([0, 30, 60, 90, 120, 150], N)
        initial_guesses.append(pts.flatten())
        
    # Guess 3: Triangular lattice
    lattice_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            lattice_pts.append([x, y, 0.0])
    lattice_pts = np.array(lattice_pts)
    
    # Greedy pick 12 points with min distance
    picked = []
    sorted_idx = np.argsort(np.sum(lattice_pts**2, axis=1))
    for idx in sorted_idx:
        p = lattice_pts[idx]
        valid = True
        for q in picked:
            if np.linalg.norm(p[:2] - q[:2]) < np.sqrt(3) * 0.99:
                valid = False
                break
        if valid:
            picked.append(p)
            if len(picked) == 12:
                break
    if len(picked) == 12:
        initial_guesses.append(np.array(picked).flatten())

    for outer_angle in outer_angles:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        for guess in initial_guesses:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            try:
                # Use Nelder-Mead
                # x0 is 3N
                res = minimize(
                    _objective,
                    guess,
                    args=(outer_angle,),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                
                x_opt = res.x
                params_opt = x_opt.reshape(N, 3)
                
                # Verify validity and compute exact side length
                inner_hexes = [[params_opt[i][0], params_opt[i][1], params_opt[i][2]] for i in range(N)]
                
                # Compute side length
                side = _compute_side_length([_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes], outer_angle)
                
                is_valid, reason = _check_validity(inner_hexes, side, (0.0, 0.0), outer_angle)
                
                if is_valid and side < best_side:
                    best_side = side
                    best_params = inner_hexes
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle
                    
            except Exception as e:
                pass
                
    if best_params is not None:
        return (best_params, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
