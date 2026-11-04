import numpy as np
import time
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _min_dist_polygons(v1, v2):
    """Compute minimum distance between two convex polygons using SAT."""
    # If they intersect, distance is 0 (or negative if we considered penetration, but we want separation)
    # For optimization, we want a smooth penalty for overlap.
    # SAT gives the minimum separation distance along the separating axis.
    
    min_sep = float('inf')
    
    # Check all edges of both polygons as potential separating axes
    for i in range(len(v1)):
        edge = v1[(i+1) % len(v1)] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        
        # Project v1
        proj1 = np.dot(v1, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        
        # Project v2
        proj2 = np.dot(v2, normal)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2:
            sep = min2 - max1
        elif max2 < min1:
            sep = min1 - max2
        else:
            # Overlap in this direction
            # The "penetration" depth is the overlap amount
            # For distance, if they overlap, the minimum distance is 0 (touching) or negative (intersecting)
            # We treat intersection as a penalty.
            overlap = min(max1, max2) - max(min1, min2)
            # If we want a smooth function that penalizes overlap, we can return -overlap
            # But standard distance is 0 if intersecting.
            # Let's use a continuous approximation: if overlap, return -overlap * factor
            # This creates a gradient pushing them apart.
            return -overlap # Return negative distance to indicate penetration
            
        if sep < min_sep:
            min_sep = sep
            
    # Also check if one is inside the other? 
    # For convex polygons, if no separating axis is found, they intersect.
    # The loop above returns -overlap if any axis shows overlap.
    # If the loop finishes without returning, it means no axis showed overlap?
    # Wait, if they intersect, EVERY axis will show overlap.
    # So the first axis checked will return -overlap.
    # This is correct for a penalty function.
    
    return min_sep

def _objective(params):
    """
    Objective function to minimize.
    params: [x1, y1, ..., x12, y12, global_rotation]
    """
    if time.time() - START_TIME > TIME_LIMIT:
        return 1e10 # Timeout penalty
        
    # Unpack parameters
    # First 24 params are centers (12 hexes * 2)
    # Last param is global rotation in radians
    centers = params[:24].reshape(12, 2)
    global_rot = params[24]
    
    # Topology: 
    # Hex 0: Center (0,0) relative to cluster center? 
    # Let's fix the cluster center at (0,0) for the outer hexagon.
    # The centers are absolute coordinates.
    
    # We need to ensure disjointness.
    # Penalty for overlap
    overlap_penalty = 0.0
    verts = []
    for i in range(12):
        # All inner hexagons have the same orientation relative to the cluster?
        # The task says "each inner hexagon may translate and rotate".
        # However, the strategy says "pre-computing the optimal relative positions... via a fixed... topology".
        # And "reducing the non-convex landscape to a smooth function of rotation".
        # This implies we fix the relative orientation of inner hexes to be the same, 
        # and we only optimize the global rotation of the whole cluster.
        # Or, we optimize individual rotations?
        # "1D constrained optimization problem over a single global rotation angle"
        # This implies the relative positions and orientations are fixed by the topology, 
        # and we only search the global rotation?
        # "pre-computing the optimal relative positions... via a fixed, high-density honeycomb cluster topology"
        # This suggests the relative positions are fixed (e.g., standard honeycomb spacing).
        # But if positions are fixed, the packing is rigid.
        # "minimizing the outer side length"
        # If the cluster is rigid, the side length is fixed.
        # Unless... the "optimal relative positions" are found by a separate step?
        # Or the "1D optimization" is over the rotation, and the positions are allowed to relax?
        
        # Let's re-read: "Formulate the packing as a 1D constrained optimization problem over a single global rotation angle by pre-computing the optimal relative positions... thereby reducing the non-convex landscape to a smooth function of rotation"
        
        # This implies:
        # 1. We have a specific topology (e.g. which hex touches which).
        # 2. We assume they are in contact (distance = 2*apothem = sqrt(3) or similar).
        # 3. We solve for the relative positions given the topology and contact constraints.
        # 4. This gives a set of relative positions.
        # 5. We then rotate this rigid cluster by angle theta.
        # 6. We compute the bounding hexagon side length for this rotated cluster.
        # 7. We minimize this side length over theta.
        
        # However, "pre-computing" suggests we do this outside the main optimization loop?
        # Or is the "1D optimization" the main loop?
        
        # Let's try a slightly more flexible interpretation:
        # The topology defines the graph of contacts.
        # We assume all contacts are tight.
        # We solve for the positions using a local relaxation (like a spring system) 
        # where springs are constrained to length L = sqrt(3) (for side 1 hexes, distance between centers of touching hexes is 2*sqrt(3)/2? No.
        # Distance between centers of two touching unit hexagons is 2 * apothem = 2 * (sqrt(3)/2) * 1 = sqrt(3).
        # Wait, apothem is sqrt(3)/2. Distance from center to edge is sqrt(3)/2.
        # Two hexagons touching edge-to-edge have center distance 2 * sqrt(3)/2 = sqrt(3).
        
        # So, we define a graph of 12 nodes.
        # We run a few steps of force-directed layout with fixed edge lengths to get the "optimal relative positions".
        # Then we freeze them.
        # Then we optimize the global rotation.
        
        # But the prompt says "1D constrained optimization... over a single global rotation angle".
        # This strongly implies the positions are FIXED.
        
        # Let's implement this:
        # 1. Define topology.
        # 2. Relax positions to satisfy contact constraints (once, at start).
        # 3. In the objective function, just rotate the fixed positions and compute bounding box.
        
        # BUT, the code provided in the prompt is a "seed" that does simulation.
        # My new strategy is different.
        
        # Let's refine the "Pre-compute" part.
        # I will define the topology in `construct` and do the relaxation there.
        # Then `construct` will run the 1D search.
        
        # For now, this objective function is a placeholder. 
        # The real logic will be in construct.
        return 0.0

def _compute_bounding_side(centers, global_rot, outer_rot=0.0):
    """
    Compute the minimum side length of a regular hexagon (with rotation outer_rot)
    that contains all unit hexagons at `centers` (rotated by global_rot).
    """
    # Rotate centers by global_rot
    c, s = np.cos(global_rot), np.sin(global_rot)
    rot_mat = np.array([[c, -s], [s, c]])
    rotated_centers = centers @ rot_mat.T
    
    # The inner hexagons are also rotated.
    # Since they are regular hexagons, rotating the center and the shape by the same amount
    # is equivalent to just rotating the center if the shape is symmetric?
    # No, the shape orientation matters for the bounding box.
    # But if all inner hexagons have the SAME orientation relative to the cluster,
    # and we rotate the whole cluster, they all rotate together.
    # Let's assume all inner hexagons have angle 0 in the cluster frame.
    # So in global frame, their angle is global_rot.
    
    max_proj = 0.0
    # Outer hexagon normals
    # The outer hexagon is fixed at (0,0) with rotation outer_rot.
    # We want to find the minimum side L such that all vertices of inner hexes are inside.
    # This is equivalent to: for each of the 6 normals of the outer hex,
    # the max projection of any inner vertex onto that normal must be <= L * cos(0) = L?
    # No. The distance from center to edge (apothem) of a hexagon with side L is L * sqrt(3) / 2.
    # So the constraint is: max_proj <= L * sqrt(3) / 2.
    # Therefore L = max_proj * 2 / sqrt(3).
    
    for k in range(6):
        angle = np.deg2rad(outer_rot + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(12):
            center = rotated_centers[i]
            # Vertices of the inner hexagon
            # Center is `center`, angle is `global_rot`
            # Vertices are center + 1 * [cos(global_rot + j*60), sin(global_rot + j*60)]
            # We need the max projection of these vertices onto `normal`.
            # proj_v = dot(center + v_rel, normal) = dot(center, normal) + dot(v_rel, normal)
            # Max over j of dot(v_rel, normal).
            # v_rel are the vertices of a unit hexagon at angle 0 (relative to its center) rotated by global_rot?
            # Yes, if the hexagon is aligned with the cluster.
            
            # The set of vectors v_rel is fixed for a unit hexagon.
            # Let's precompute the support function of a unit hexagon.
            # h(u) = max_{v in Hex} dot(v, u)
            # For a regular hexagon with side 1, the support function is the apothem in direction u?
            # No, it depends on the angle.
            
            # Let's just compute it explicitly for safety.
            theta = global_rot
            angles = theta + np.arange(6) * (np.pi / 3)
            rel_verts = np.array([
                [np.cos(a), np.sin(a)] for a in angles
            ])
            
            proj_rel = np.dot(rel_verts, normal)
            max_rel = np.max(proj_rel)
            
            proj_center = np.dot(center, normal)
            
            total_proj = proj_center + max_rel
            
            if total_proj > max_proj:
                max_proj = total_proj
                
    return max_proj * 2.0 / np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    # 1. Define Topology
    # 12 Hexagons:
    # 0: Center
    # 1-6: Ring 1 (touching 0)
    # 7-11: Ring 2 (touching 1,2,3,4,5,6 respectively, and some others)
    # Standard Honeycomb Cluster:
    # Center (0)
    # Ring 1: 6 hexes at distance sqrt(3) from center.
    # Ring 2: 6 hexes at distance 2*sqrt(3) from center? 
    # In a honeycomb lattice, the next ring has 6 hexes.
    # Total 1 + 6 + 6 = 13. We have 12.
    # So we take the center, ring 1 (6), and 5 from ring 2?
    # Or maybe a different topology?
    # Let's try: Center + 6 Ring 1 + 5 Ring 2.
    
    # Graph edges (i, j) means hex i and j are touching (distance sqrt(3))
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
        # Ring 2 connections
        # Hex 7 touches 1 and 2?
        # In a perfect lattice, the hexes in ring 2 are located at the "corners" of the ring 1 hexes.
        # Let's place them roughly.
        (1, 7), (2, 7), (7, 2), # 7 is between 1 and 2?
        (2, 8), (3, 8),
        (3, 9), (4, 9),
        (4, 10), (5, 10),
        (5, 11), (6, 11),
        # And connections between ring 2 hexes?
        # (7, 8)? No, they are separated by 2?
        # Let's just use the edges to relax positions.
    ]
    
    # Initial positions for relaxation
    pos = np.zeros((12, 2))
    # Place Ring 1
    for i in range(1, 7):
        angle = np.pi / 3 * (i - 1)
        pos[i] = np.array([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
        
    # Place Ring 2 (roughly)
    # Hex 7 is opposite to 0 relative to edge 1-2?
    # Angle of 1 is 0, Angle of 2 is 60. Midpoint angle 30.
    # Distance from center is 2 * sqrt(3)?
    # Let's estimate:
    # 7 touches 1 and 2.
    # 1 is at (sqrt(3), 0). 2 is at (sqrt(3)/2, 3/2).
    # 7 is at distance sqrt(3) from both.
    # Intersection of two circles.
    # Let's just use a heuristic placement for initialization.
    ring2_angles = [np.pi/6 + i * np.pi/3 for i in range(6)]
    for i in range(7, 13):
        idx = i - 7
        angle = ring2_angles[idx]
        pos[i] = np.array([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)])
        
    # We only have 12 hexes (0-11).
    # So indices 7, 8, 9, 10, 11 are the 5 ring 2 hexes.
    # Let's adjust edges to match indices 0-11.
    # Edges:
    # 0 touches 1-6
    # 1-6 form a ring
    # 7 touches 1, 2
    # 8 touches 2, 3
    # 9 touches 3, 4
    # 10 touches 4, 5
    # 11 touches 5, 6
    
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
        (1, 7), (2, 7),
        (2, 8), (3, 8),
        (3, 9), (4, 9),
        (4, 10), (5, 10),
        (5, 11), (6, 11)
    ]
    
    # 2. Relax Positions (Fixed Topology Optimization)
    # We want to find positions such that dist(i,j) = sqrt(3) for all edges.
    # This is a distance geometry problem.
    # We can use a simple gradient descent on the error sum (dist - L)^2.
    
    L = np.sqrt(3)
    current_pos = pos.copy()
    
    # Optimization of relative positions
    # We fix hex 0 at (0,0) and fix the angle of hex 1 on the x-axis to break rotational symmetry.
    # Actually, we just want the shape.
    
    def pos_error(p):
        p = p.reshape(12, 2)
        err = 0.0
        for i, j in edges:
            d = np.linalg.norm(p[i] - p[j])
            err += (d - L) ** 2
        return err
    
    # Use scipy minimize to relax the topology
    # We optimize all 24 coordinates.
    # To prevent drift, we can fix p[0] = 0 and p[1] = (L, 0).
    # Let's just run it and then center the result.
    
    if time.time() - start_time < TIME_LIMIT - 5:
        try:
            res = minimize(pos_error, current_pos.flatten(), method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success or res.fun < pos_error(current_pos.flatten()):
                current_pos = res.x.reshape(12, 2)
        except Exception:
            pass
            
    # Center the cluster
    centroid = np.mean(current_pos, axis=0)
    current_pos -= centroid
    
    # 3. 1D Optimization over Global Rotation
    # The cluster is now rigid.
    # We rotate it by theta.
    # We compute the bounding side length.
    # We minimize this.
    
    best_side = float('inf')
    best_theta = 0.0
    
    # Search space for theta: 0 to 60 degrees (periodicity of hexagon)
    # Actually, the bounding hexagon has 60 degree symmetry.
    # The inner cluster has 60 degree symmetry? 
    # Our topology (missing one ring 2 hex) breaks the 6-fold symmetry.
    # It has 2-fold symmetry? No, it's asymmetric.
    # So we search 0 to 360.
    
    # Use Golden Section Search or just a fine grid + local refine.
    # Grid: 0 to 360 with step 0.1 deg.
    
    # Precompute support function of unit hexagon for speed?
    # _compute_bounding_side is fast enough for a few hundred evaluations.
    
    # Let's use a coarse grid first, then refine.
    
    thetas = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    
    for theta in thetas:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        side = _compute_bounding_side(current_pos, theta)
        if side < best_side:
            best_side = side
            best_theta = theta
            
    # Refine around best_theta
    # Nelder-Mead on 1D? Or just finer grid.
    # Let's do a few steps of local search.
    
    current_best_theta = best_theta
    current_best_side = best_side
    
    for _ in range(20): # Refinement steps
        if time.time() - start_time > TIME_LIMIT - 2:
            break
        # Try small perturbations
        improved = False
        for delta in [0.01, -0.01, 0.001, -0.001]:
            new_theta = current_best_theta + delta
            side = _compute_bounding_side(current_pos, new_theta)
            if side < current_best_side:
                current_best_side = side
                current_best_theta = new_theta
                improved = True
        if not improved:
            break
            
    best_theta = current_best_theta
    best_side = current_best_side
    
    # 4. Construct Output
    # Inner hexagons: centers are current_pos rotated by best_theta.
    # Angles are best_theta (in degrees).
    
    c, s = np.cos(best_theta), np.sin(best_theta)
    rot_mat = np.array([[c, -s], [s, c]])
    final_centers = current_pos @ rot_mat.T
    
    inner_hexes = []
    for i in range(12):
        inner_hexes.append([final_centers[i][0], final_centers[i][1], np.rad2deg(best_theta)])
        
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = 0.0 # We assumed outer hexagon is aligned with axes for the projection calc.
    # Wait, _compute_bounding_side assumed outer_rot=0.
    # If we change outer_rot, the required side might change?
    # No, the minimum side length is invariant to the rotation of the outer hexagon,
    # because we are free to choose the outer hexagon's rotation to minimize the side length.
    # Actually, the problem asks to return the outer hexagon's rotation.
    # We can set outer_angle = 0.0 and the side length is the minimum possible.
    # Or we can return the angle that achieved it.
    # Since we minimized over global_rot of the INNER cluster relative to a FIXED outer hexagon,
    # the resulting packing is optimal for that outer orientation.
    # So we return outer_angle = 0.0.
    
    return (inner_hexes, outer_center, outer_side, outer_angle)

# Fallback if something goes wrong
# The function above is the main logic.
# Ensure no exceptions escape.

def construct_safe():
    try:
        return construct()
    except Exception:
        # Fallback: Linear arrangement
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

# The evaluator expects `construct`.
# We replace `construct` with the safe version?
# No, we should just make `construct` robust.

# Let's wrap the main logic in try-except inside construct.

def construct():
    start_time = time.time()
    try:
        # ... (move logic here) ...
        
        # 1. Define Topology
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
            (1, 7), (2, 7),
            (2, 8), (3, 8),
            (3, 9), (4, 9),
            (4, 10), (5, 10),
            (5, 11), (6, 11)
        ]
        
        pos = np.zeros((12, 2))
        for i in range(1, 7):
            angle = np.pi / 3 * (i - 1)
            pos[i] = np.array([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
            
        ring2_angles = [np.pi/6 + i * np.pi/3 for i in range(6)]
        for i in range(7, 13):
            idx = i - 7
            angle = ring2_angles[idx]
            pos[i] = np.array([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)])
            
        current_pos = pos.copy()
        L = np.sqrt(3)
        
        def pos_error(p):
            p = p.reshape(12, 2)
            err = 0.0
            for i, j in edges:
                d = np.linalg.norm(p[i] - p[j])
                err += (d - L) ** 2
            return err
        
        if time.time() - start_time < TIME_LIMIT - 5:
            try:
                res = minimize(pos_error, current_pos.flatten(), method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.success or res.fun < pos_error(current_pos.flatten()):
                    current_pos = res.x.reshape(12, 2)
            except Exception:
                pass
                
        centroid = np.mean(current_pos, axis=0)
        current_pos -= centroid
        
        best_side = float('inf')
        best_theta = 0.0
        
        thetas = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        
        for theta in thetas:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            side = _compute_bounding_side(current_pos, theta)
            if side < best_side:
                best_side = side
                best_theta = theta
                
        current_best_theta = best_theta
        current_best_side = best_side
        
        for _ in range(20):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            improved = False
            for delta in [0.01, -0.01, 0.001, -0.001]:
                new_theta = current_best_theta + delta
                side = _compute_bounding_side(current_pos, new_theta)
                if side < current_best_side:
                    current_best_side = side
                    current_best_theta = new_theta
                    improved = True
            if not improved:
                break
                
        best_theta = current_best_theta
        best_side = current_best_side
        
        c, s = np.cos(best_theta), np.sin(best_theta)
        rot_mat = np.array([[c, -s], [s, c]])
        final_centers = current_pos @ rot_mat.T
        
        inner_hexes = []
        for i in range(12):
            inner_hexes.append([final_centers[i][0], final_centers[i][1], np.rad2deg(best_theta)])
            
        outer_center = (0.0, 0.0)
        outer_side = best_side
        outer_angle = 0.0
        
        return (inner_hexes, outer_center, outer_side, outer_angle)
        
    except Exception:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
