# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, linprog

def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _hex_normals(a_deg):
    """12 edge normals (unit vectors) for a hexagon of rotation a_deg."""
    a = np.radians(a_deg)
    edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    return np.vstack([normals, -normals])

def _separation_ok(v1, v2, n1, n2):
    """Check if two hexagons are separated."""
    for normals in (n1, n2):
        for j in range(normals.shape[0]):
            n = normals[j]
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True
    return False

def _hexagons_separated(v1, v2, a1, a2):
    n1 = _hex_normals(a1)
    n2 = _hex_normals(a2)
    return _separation_ok(v1, v2, n1, n2)

def _point_in_hexagon(p, outer_verts):
    """Check if point p is inside (or on boundary of) the outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        to_p = p - p1
        cross = edge[0] * to_p[1] - edge[1] * to_p[0]
        if cross < -1e-6:
            return False
    return True

def _all_inner_in_outer(inner_verts_list, outer_verts):
    """Check all vertices of all inner hexagons are inside the outer hexagon."""
    for verts in inner_verts_list:
        for p in verts:
            if not _point_in_hexagon(p, outer_verts):
                return False
    return True

def _compute_minimal_outer_hex_lp(all_verts):
    """
    Compute the minimal enclosing regular hexagon for a set of points using Linear Programming.
    Returns (center_x, center_y, angle_deg, side_length).
    """
    # The optimal orientation of the enclosing hexagon is likely aligned with one of the hull edges or vertices.
    # We will sample angles. For each angle, the problem of finding the minimal side length s and center (cx, cy)
    # such that all points are inside the hexagon is a Linear Program.
    
    # Variables: x = [cx, cy, s]
    # Constraints: For each point p_i and each of the 6 outward normals n_j of the hexagon:
    # (p_i - c) . n_j <= apothem = s * sqrt(3) / 2
    # => -n_j[0]*cx - n_j[1]*cy - (sqrt(3)/2)*s <= -p_i . n_j
    # Also s >= 0.
    
    best_s = np.inf
    best_center = np.mean(all_verts, axis=0)
    best_angle = 0.0
    
    # Sample angles. The symmetry of the hexagon is 60 degrees.
    # The optimal angle is often related to the geometry of the points.
    # We'll sample 0 to 60 degrees with a reasonable resolution.
    # Since the LP is fast, we can afford a finer grid.
    angles = np.linspace(0, 60, 61)
    
    apothem_coeff = np.sqrt(3) / 2.0
    
    for a_deg in angles:
        a = np.radians(a_deg)
        # Outward normals for the 6 edges
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)]) # (6, 2)
        
        # Construct A_ub and b_ub for linprog
        # We want to minimize s.
        # Objective: c = [0, 0, 1]
        c_obj = np.array([0.0, 0.0, 1.0])
        
        A_ub_rows = []
        b_ub_rows = []
        
        # For each point i and each normal j:
        # (p_i - c) . n_j <= s * apothem_coeff
        # p_i . n_j - c . n_j <= s * apothem_coeff
        # -c . n_j - s * apothem_coeff <= -p_i . n_j
        # -n_j[0]*cx - n_j[1]*cy - apothem_coeff*s <= -p_i . n_j
        
        for i in range(len(all_verts)):
            p = all_verts[i]
            for j in range(6):
                n = normals[j]
                # Coefficients for cx, cy, s
                row = np.array([-n[0], -n[1], -apothem_coeff])
                A_ub_rows.append(row)
                b_ub_rows.append(-p @ n)
        
        if not A_ub_rows:
            continue
            
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
        
        # Bounds: cx, cy are unbounded (or large), s >= 0
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0.0, np.inf)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                s_val = res.x[2]
                if s_val < best_s:
                    best_s = s_val
                    best_center = res.x[:2]
                    best_angle = a_deg
        except Exception:
            pass
            
    return (best_center[0], best_center[1], best_angle, best_s)

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Initial configuration: Hexagonal lattice
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3)
    # 4 more at distance 2*sqrt(3) in specific directions?
    
    np.random.seed(42)
    
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1: 6 hexagons
    r1 = np.sqrt(3) * 1.1 # Slightly larger to avoid initial overlap in optimization
    for i in range(6):
        ang = np.deg2rad(60 * i)
        positions.append([r1 * np.cos(ang), r1 * np.sin(ang)])
        angles.append(0.0)
        
    # Ring 2: 4 hexagons to make 11 total
    r2 = 2 * np.sqrt(3) * 1.1
    for i in range(4):
        ang = np.deg2rad(60 * i + 30)
        positions.append([r2 * np.cos(ang), r2 * np.sin(ang)])
        angles.append(0.0)
        
    positions = np.array(positions, dtype=float)
    angles = np.array(angles, dtype=float)
    
    # Optimization Strategy:
    # We want to pack the hexagons tightly.
    # We can formulate this as minimizing the maximum "overlap" or maximizing the minimum "clearance".
    # However, exact non-overlap is a non-convex constraint.
    # A common heuristic is to minimize the sum of pairwise distances, or minimize the bounding box size,
    # while enforcing a minimum distance penalty if they are too close.
    # Given the small number of hexagons (11), we can use a smooth penalty function for overlap.
    
    # Let's define a function that computes a "cost" for the current configuration.
    # Cost = Sum of squared overlaps (approximated) + Regularization to keep them centered.
    # Better: Minimize the diameter of the cluster (max distance from centroid) while pushing apart overlaps.
    
    def cost_function(pos_flat):
        pos = pos_flat.reshape(11, 2)
        # Centering penalty to prevent drift
        centroid = np.mean(pos, axis=0)
        centering_cost = np.sum((pos - centroid)**2) * 0.01
        
        # Overlap penalty
        # For each pair, if distance < d_min, add penalty
        # d_min for touching hexagons (same orientation) is sqrt(3).
        # We use a smooth approximation of the "max(0, d_min - dist)^2"
        d_min = np.sqrt(3)
        overlap_cost = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                dist = np.linalg.norm(pos[j] - pos[i])
                if dist < d_min:
                    # Quadratic penalty for violation
                    overlap_cost += (d_min - dist)**2 * 10.0
                # Also penalize if they are very close to ensure they don't just touch
                # Actually, we want them to be as close as possible without overlapping.
                # So we want to minimize the total spread, but the overlap penalty pushes them apart.
                # To pack tightly, we should minimize the bounding box or radius.
        
        # Radius penalty: minimize max distance from centroid
        # Using a smooth approximation: sum(exp(dist^2 / T)) or just max dist
        # Let's use sum of squared distances from centroid as a proxy for compactness
        # But we need to balance this with overlap.
        # Actually, if we just minimize sum of pairwise distances, they will clump.
        # The overlap penalty will push them apart to the boundary of non-overlap.
        # So: Objective = Sum_{i<j} dist(i,j) + Large * Overlap_Penalty
        
        total_dist = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                total_dist += np.linalg.norm(pos[j] - pos[i])
                
        return total_dist + overlap_cost + centering_cost

    # Initial guess
    x0 = positions.flatten()
    
    # Run optimization
    # Use L-BFGS-B for efficiency
    # We might need to run this for a few iterations or restarts if it gets stuck in a local minimum.
    
    best_cost = np.inf
    best_pos = positions.copy()
    
    # Multiple restarts with slightly perturbed initial positions
    for restart in range(5):
        if time.time() > deadline - 100: # Leave time for LP
            break
            
        if restart > 0:
            # Perturb initial positions
            x0_perturbed = x0 + np.random.randn(22) * 0.1
        else:
            x0_perturbed = x0
            
        try:
            res = minimize(cost_function, x0_perturbed, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-8})
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_pos = res.x.reshape(11, 2)
        except Exception:
            continue
            
    # Use the best position found
    positions = best_pos
    
    # Re-center
    centroid = np.mean(positions, axis=0)
    positions -= centroid
    
    # Now we have a candidate packing.
    # We need to verify it is valid (non-overlapping).
    # If the optimizer left any overlaps, we must resolve them.
    # The penalty was strong, so hopefully it's valid.
    # Let's do a final exact check and repair if necessary.
    
    hexagons = [(positions[i][0], positions[i][1], angles[i]) for i in range(11)]
    
    # Repair loop: if any overlap, push apart
    for iter in range(50):
        if time.time() > deadline - 10:
            break
            
        verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons]
        collision_found = False
        
        for i in range(11):
            for j in range(i + 1, 11):
                if not _hexagons_separated(verts_list[i], verts_list[j], hexagons[i][2], hexagons[j][2]):
                    # Collision!
                    collision_found = True
                    r = positions[j] - positions[i]
                    dist = np.linalg.norm(r)
                    if dist < 1e-6:
                        r = np.random.randn(2)
                        dist = np.linalg.norm(r)
                    f_dir = r / dist
                    # Push apart by a small amount
                    push = 0.005
                    positions[i] -= f_dir * push
                    positions[j] += f_dir * push
                    hexagons[i] = (positions[i][0], positions[i][1], angles[i])
                    hexagons[j] = (positions[j][0], positions[j][1], angles[j])
                    
        if not collision_found:
            break
            
    # Re-center again after repair
    centroid = np.mean(positions, axis=0)
    positions -= centroid
    hexagons = [(positions[i][0], positions[i][1], angles[i]) for i in range(11)]
    
    inner_hex_data = np.array(hexagons)
    
    # Compute minimal enclosing hexagon using LP
    all_verts = np.vstack([_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons])
    ox, oy, oa, s = _compute_minimal_outer_hex_lp(all_verts)
    
    outer_hex_data = np.array([ox, oy, oa])
    
    # Verify containment
    outer_verts = _hex_vertices(ox, oy, s, oa)
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons]
    
    if not _all_inner_in_outer(verts_list, outer_verts):
        # Expand s slightly to ensure validity
        # This might happen if LP had numerical issues or tolerance mismatch
        for _ in range(20):
            s *= 1.001
            outer_verts = _hex_vertices(ox, oy, s, oa)
            if _all_inner_in_outer(verts_list, outer_verts):
                break
                
    # Ensure s >= sqrt(11)
    s = max(s, np.sqrt(11.0))
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
