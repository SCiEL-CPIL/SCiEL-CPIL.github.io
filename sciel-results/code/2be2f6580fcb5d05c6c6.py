# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s and angle a_deg.
    Vertices are at angles a + 60*i degrees.
    """
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.deg2rad(60 * np.arange(6))
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack((xs, ys))

def _min_outer_side_for_inner(inner_hexes):
    """
    inner_hexes: array of shape (11, 3) (x, y, angle_deg). Unit side.
    Returns the minimum side length S of a regular hexagon centered at origin
    that contains all inner hexagons, allowing rotation of the outer hexagon.
    """
    # Collect all vertices
    all_verts = []
    for i in range(inner_hexes.shape[0]):
        v = _hex_vertices(inner_hexes[i, 0], inner_hexes[i, 1], 1.0, inner_hexes[i, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # We want to minimize d(apothem) such that |proj_n(p)| <= d for all p, n
    # n_k are normals at alpha + 30, alpha + 150, alpha + 270
    # d(alpha) = max_k max_p | proj_{n_k(alpha)}(p) |
    # S = 2 * d / sqrt(3)
    
    # Sample alpha in [0, 60)
    # Due to symmetry of the set of 3 axes (separated by 120), the function d(alpha)
    # might have period 60?
    # Let's just sample finely.
    alphas = np.linspace(0, 60, 360, endpoint=False)
    min_d = np.inf
    
    # Precompute projections for efficiency?
    # For each alpha, we need 3 normals.
    # Normal angles: alpha + 30, alpha + 150, alpha + 270 (in degrees)
    
    for a in alphas:
        a_rad = np.deg2rad(a)
        # Normal directions in radians
        n_angles = a_rad + np.deg2rad([30, 150, 270])
        
        # Project all vertices onto these 3 axes
        # proj = x * cos(n) + y * sin(n)
        # We can do this vectorized
        # n_angles shape (3,)
        # all_verts shape (N, 2)
        
        cos_n = np.cos(n_angles) # (3,)
        sin_n = np.sin(n_angles) # (3,)
        
        # Projections: (N, 3)
        projs = all_verts[:, 0:1] * cos_n[None, :] + all_verts[:, 1:2] * sin_n[None, :]
        
        # Absolute values
        abs_projs = np.abs(projs)
        
        # Max over vertices for each axis
        max_per_axis = np.max(abs_projs, axis=0) # (3,)
        
        # Max over axes
        current_d = np.max(max_per_axis)
        
        if current_d < min_d:
            min_d = current_d
            
    S = 2.0 * min_d / np.sqrt(3.0)
    return S

def _gap_between_hexes(h1, h2):
    """
    h1, h2: arrays of shape (6, 2) vertices.
    Returns the minimum distance between the two hexagons.
    If they overlap, returns a negative value (penalty).
    Uses Separating Axis Theorem.
    """
    v1 = h1
    v2 = h2
    
    # Get edge normals for both hexagons
    def get_normals(verts):
        normals = []
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            # Normal is perpendicular to edge
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-9:
                n = n / n_norm
                normals.append(n)
        return np.array(normals)
        
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    
    all_normals = np.vstack((n1, n2))
    
    min_gap = np.inf
    
    for n in all_normals:
        # Project v1 and v2 onto n
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check separation
        # Gap is distance between intervals [min1, max1] and [min2, max2]
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            # Overlap on this axis
            gap = 0.0 # Actually, if they overlap on ALL axes, they intersect.
            # But if they are separated on ANY axis, they are disjoint.
            # So if gap is 0 here, it's not a separating axis.
            # We are looking for the MINIMUM gap over all axes?
            # No, the "gap" between two convex polygons is 0 if they intersect or touch.
            # If they are disjoint, the distance is the minimum over separating axes of the gap.
            # Wait.
            # If they are disjoint, there exists at least one separating axis.
            # The distance between them is the minimum gap over all axes that are separating?
            # Actually, the distance between two disjoint convex polygons is the minimum gap
            # over the axes perpendicular to the edges.
            # If they overlap, the "gap" is negative.
            
            # For optimization, we want to know if they are separated by > 1e-6.
            # If they overlap on any axis, they are NOT separated by that axis.
            # They are separated if AND ONLY IF they are separated on ALL axes? NO.
            # They are separated if there EXISTS at least one axis where they are separated.
            
            # So, to check if they are validly separated:
            # We need to find if there is ANY axis where the gap > 1e-6.
            # If yes, they are separated.
            # If no, they are touching or overlapping.
            
            # For the penalty function, we want to maximize the "minimum gap".
            # But "minimum gap" is ill-defined if they overlap.
            
            # Let's define:
            # If they are separated on any axis, the distance is min(gaps on separating axes).
            # If they are not separated on any axis (i.e. they overlap), distance is -infinity?
            # Or we can just check:
            # Are they separated on ANY axis with gap > 1e-6?
            # If yes, valid.
            # If no, invalid.
            
            # To make it a continuous objective:
            # Let max_gap = max(gap_i) over all axes i.
            # If max_gap > 1e-6, they are separated.
            # The "safety margin" is max_gap.
            # We want max_gap to be large.
            # So we can use max_gap as the measure of separation.
            
            # If they overlap, all gaps are <= 0 (or 0 if we clip).
            # Actually, if they overlap, the intervals overlap, so the "gap" calculated as above is 0 (if we use max(0, ...)).
            # But we want to penalize overlap.
            
            # Let's calculate the gap on each axis.
            # gap_i = min(0, min2 - max1, min1 - max2) ? No.
            # If max1 < min2, gap = min2 - max1 (positive).
            # If max2 < min1, gap = min1 - max2 (positive).
            # Else, gap = -min(max1 - min2, max2 - min1)? No, that's depth of intersection.
            
            # Let's use:
            # gap_i = min(0, min2 - max1, min1 - max2) is wrong.
            
            # Standard SAT distance:
            # If separated, distance is gap.
            # If intersecting, distance is negative?
            
            # Let's just check validity directly in the objective.
            # If any pair has max_gap <= 1e-6, return penalty.
            
            # So in this loop, we just need to track the MAX gap found.
            # If max_gap > 1e-6, the pair is OK.
            
            pass

    # Re-do logic:
    # For a pair, calculate gap on all 12 axes.
    # If max(gaps) > 1e-6, they are separated.
    # The "quality" of separation is max(gaps).
    # We want to minimize outer side, so we just need validity.
    
    # Let's return the max gap found.
    # If max_gap > 1e-6, it's valid.
    
    # I will implement this properly in the function below.
    return 0.0 # Placeholder

def _check_pair_separation(h1, h2):
    """
    Returns the maximum gap between h1 and h2 over all 12 edge normals.
    If they are separated, this will be > 0.
    If they overlap, this will be <= 0 (actually, we return the max gap, which might be negative if we define it that way, 
    but for SAT, if they overlap, there is no separating axis, so all "gaps" are <= 0 in the sense of separation).
    Actually, if they overlap, the intervals overlap, so the calculated gap (min2-max1) would be negative.
    So max_gap will be the least negative (closest to 0) or positive.
    """
    v1 = h1
    v2 = h2
    
    def get_normals(verts):
        normals = []
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-9:
                n = n / n_norm
                normals.append(n)
        return np.array(normals)
        
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    
    all_normals = np.vstack((n1, n2))
    
    max_gap = -np.inf
    
    for n in all_normals:
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Gap on this axis
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            # Overlap
            gap = 0.0 # Or negative? 
            # If we use 0, then if they overlap on all axes, max_gap will be 0.
            # If they are separated on one axis, max_gap > 0.
            # This works for validity check: valid if max_gap > 1e-6.
            
        if gap > max_gap:
            max_gap = gap
            
    return max_gap

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 550.0 # 550s limit to be safe
    
    n = 11
    
    # Objective function for optimization
    def objective(params):
        if time.time() > deadline:
            return 1e9
            
        hexes = params.reshape(n, 3)
        
        # Check separation
        min_sep = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                v1 = _hex_vertices(hexes[i, 0], hexes[i, 1], 1.0, hexes[i, 2])
                v2 = _hex_vertices(hexes[j, 0], hexes[j, 1], 1.0, hexes[j, 2])
                gap = _check_pair_separation(v1, v2)
                if gap < min_sep:
                    min_sep = gap
        
        if min_sep < 1e-6:
            # Penalty for invalid separation
            return 10.0 + (1e-6 - min_sep) * 100.0
            
        # Calculate min outer side
        S = _min_outer_side_for_inner(hexes)
        return S

    # Initial guesses
    # 1. Seed grid
    seed_grid = np.array([
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
    ])
    
    # 2. Symmetric flower + 4
    # Center
    # 6 around
    # 4 in gaps?
    # Let's try a dense random start
    np.random.seed(42)
    
    best_S = np.inf
    best_hexes = seed_grid.copy()
    
    # Try seed grid
    S0 = objective(seed_grid.flatten())
    if S0 < best_S:
        best_S = S0
        best_hexes = seed_grid.copy()
        
    # Run optimization from multiple starts
    starts = [seed_grid.flatten()]
    
    # Add random starts
    for _ in range(5):
        # Random positions in a circle of radius 5
        angles = np.random.uniform(0, 2*np.pi, n)
        radii = np.random.uniform(0, 4, n)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        as_ = np.random.uniform(0, 60, n)
        rand_start = np.column_stack((xs, ys, as_)).flatten()
        starts.append(rand_start)
        
        # Add a "flower" start
        flower = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [1, np.sqrt(3), 0],
            [-1, np.sqrt(3), 0],
            [-2, 0, 0],
            [-1, -np.sqrt(3), 0],
            [1, -np.sqrt(3), 0],
            [3, np.sqrt(3), 0],
            [0, 2*np.sqrt(3), 0],
            [-3, np.sqrt(3), 0],
            [0, -2*np.sqrt(3), 0],
        ])
        # Perturb it
        perturbed = flower + np.random.normal(0, 0.1, flower.shape)
        starts.append(perturbed.flatten())

    for start in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000, 'maxfev': 2000})
            
            if res.fun < best_S:
                best_S = res.fun
                best_hexes = res.x.reshape(n, 3)
                
        except Exception as e:
            pass
            
    # Final check and cleanup
    # Ensure angles are in [0, 60)
    best_hexes[:, 2] = best_hexes[:, 2] % 60.0
    
    # Compute final S
    final_S = _min_outer_side_for_inner(best_hexes)
    
    # Construct return values
    inner = best_hexes
    outer = np.array([0.0, 0.0, 0.0]) # Centered at origin, angle 0 is fine as we optimized for min S with variable angle, 
                                      # but the returned outer hexagon needs to be the one that contains them.
                                      # The problem asks for (x, y, angle_deg) of the outer hexagon.
                                      # We found the min S assuming we can rotate.
                                      # We should return the optimal rotation angle as well.
                                      # Let's re-find the optimal angle for the best hexes.
    
    # Re-calc optimal angle
    all_verts = []
    for i in range(n):
        v = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    alphas = np.linspace(0, 60, 360, endpoint=False)
    min_d = np.inf
    best_alpha = 0
    
    for a in alphas:
        a_rad = np.deg2rad(a)
        n_angles = a_rad + np.deg2rad([30, 150, 270])
        cos_n = np.cos(n_angles)
        sin_n = np.sin(n_angles)
        projs = all_verts[:, 0:1] * cos_n[None, :] + all_verts[:, 1:2] * sin_n[None, :]
        abs_projs = np.abs(projs)
        max_per_axis = np.max(abs_projs, axis=0)
        current_d = np.max(max_per_axis)
        if current_d < min_d:
            min_d = current_d
            best_alpha = a
            
    outer_angle = best_alpha
    outer = np.array([0.0, 0.0, outer_angle])
    side = 2.0 * min_d / np.sqrt(3.0)
    
    return inner, outer, side

# EVOLVE-BLOCK-END
