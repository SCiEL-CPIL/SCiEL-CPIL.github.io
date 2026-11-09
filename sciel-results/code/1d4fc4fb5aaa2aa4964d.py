# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _compute_min_enclosing_hex_size(vertices, outer_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon with given orientation
    that encloses all given vertices.
    vertices: (N, 2) array of points.
    outer_angle_deg: orientation of the outer hexagon.
    """
    a = np.radians(outer_angle_deg)
    # Normals of the outer hexagon edges
    # The edges are perpendicular to the normals.
    # Vertices of outer hexagon are at angles a + 60i.
    # The outward normals are at angles a + 60i + 30? No.
    # Let's derive.
    # Vertex i is at angle $\theta_i = a + i \pi/3$.
    # Edge i connects vertex i and i+1.
    # Midpoint of edge i is at angle $\theta_i + \pi/6$.
    # The normal to the edge points in the direction of the midpoint angle.
    # So normal angle $\phi_i = a + i \pi/3 + \pi/6 = a + (2i+1)\pi/6$.
    
    normal_angles = a + np.pi/6 + np.pi/3 * np.arange(6)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Project all vertices onto these 6 normals
    # proj (6, N)
    projs = np.dot(normals, vertices.T)
    
    # For each normal, find the max projection (furthest point in that direction)
    # The hexagon must extend at least this far.
    # The distance from center to the edge (apothem) must be >= max_proj.
    # s = 2 * apothem / sqrt(3)
    max_projs = np.max(projs, axis=1)
    
    # Also consider the center? The problem implies the outer hexagon is centered at (0,0) 
    # after shifting? 
    # "after shifting all centers by minus the outer center"
    # The evaluator shifts the inner hexagons so their mean center is (0,0)? 
    # No, "after shifting all centers by minus the outer center".
    # The outer center is returned as part of the tuple.
    # The evaluator likely computes the outer center, shifts everything, and checks.
    # To minimize s, we should center the cluster optimally.
    # The minimal enclosing hexagon of a set of points is not necessarily centered at the origin.
    # However, the task returns (inner, outer, s).
    # outer is (x, y, angle).
    # The evaluator will likely use the provided outer center.
    # So we should return the optimal center for the outer hexagon as well.
    
    # For a fixed orientation, the minimal enclosing regular hexagon is not trivially centered at the centroid.
    # But for a highly symmetric cluster, the centroid is a good approximation.
    # Let's assume the outer hexagon is centered at the centroid of the inner hexagons' centers.
    # Then we just need to ensure all vertices are inside.
    
    # If we center at centroid C, the vertices relative to C are v - C.
    # We need max((v-C) . n) <= apothem for all normals n.
    # And max((v-C) . (-n)) <= apothem for all normals n.
    # So apothem >= max( max(v.n - C.n), max(-v.n + C.n) )
    # apothem >= max( max(v.n), -min(v.n) ) - C.n ? No.
    # Let's just compute the required apothem for a hexagon centered at C.
    # For each normal n:
    #   max_dist = max( (v - C) . n for v in V )
    #   min_dist = min( (v - C) . n for v in V )
    #   required_apothem_n = max(max_dist, -min_dist)
    #   s_n = 2 * required_apothem_n / sqrt(3)
    # s = max(s_n)
    
    # We will try to optimize the center C as well?
    # For a fixed orientation, the optimal C is the intersection of the "middle" lines.
    # This is a linear programming problem.
    # Given the time constraint, let's assume the centroid is close to optimal.
    # Or better, let's find the center that minimizes s for a fixed orientation.
    # This is the Chebyshev center of the intersection of strips.
    # Actually, let's just use the centroid for now and see if it's good enough.
    # The cluster is rigid, so its shape is fixed.
    
    centroid = np.mean(vertices, axis=0)
    rel_verts = vertices - centroid
    
    max_projs_rel = np.max(np.dot(normals, rel_verts.T), axis=1)
    min_projs_rel = np.min(np.dot(normals, rel_verts.T), axis=1)
    
    required_apothems = np.maximum(max_projs_rel, -min_projs_rel)
    max_apothem = np.max(required_apothems)
    
    s = 2.0 * max_apothem / np.sqrt(3)
    
    return s, centroid

def _generate_rigid_clusters():
    """
    Generate several rigid cluster configurations of 12 hexagons.
    Returns a list of arrays, each of shape (12, 3) [x, y, angle].
    """
    clusters = []
    
    L = np.sqrt(3) # Distance between centers of touching hexagons
    
    def make_cluster(coords):
        """
        coords: list of (q, r) axial coordinates.
        """
        pts = []
        for q, r in coords:
            x = L * (q + r / 2.0)
            y = L * r * np.sqrt(3) / 2.0
            # Angle of the hexagon in the lattice is 0
            pts.append([x, y, 0.0])
        
        pts = np.array(pts)
        # Center the cluster
        center = np.mean(pts[:, :2], axis=0)
        pts[:, 0] -= center[0]
        pts[:, 1] -= center[1]
        return pts

    # Cluster 1: 7-hex flower (center + 6 neighbors) + 5 more
    # Axial coords for flower: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Add 5 more attached to the outer ring.
    # Let's attach them to (1,0), (0,1), (-1,1), (-1,0), (0,-1)
    # (1,0) -> (2,0)
    # (0,1) -> (0,2)
    # (-1,1) -> (-2,1)
    # (-1,0) -> (-2,0)
    # (0,-1) -> (0,-2)
    coords1 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (0,2), (-2,1), (-2,0), (0,-2)
    ]
    clusters.append(make_cluster(coords1))

    # Cluster 2: 3-4-3-2 arrangement? 
    # Let's try a more rectangular block.
    # Row 0: 4 hexes
    # Row 1: 4 hexes
    # Row 2: 3 hexes
    # Row 3: 1 hex? No, 12 total.
    # Let's do 3 rows of 4?
    # (0,0) to (3,0)
    # (0,1) to (3,1)
    # (0,2) to (3,2)
    coords2 = []
    for r in range(3):
        for q in range(4):
            coords2.append((q, r))
    clusters.append(make_cluster(coords2))

    # Cluster 3: Compact Diamond
    # Center (0,0)
    # Ring 1: 6
    # Ring 2: Pick 5 from the 12 neighbors of the ring 1?
    # Let's just pick a random compact set from a larger lattice and prune?
    # Or define manually.
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1) -> 5 more
    coords3 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    clusters.append(make_cluster(coords3))

    # Cluster 4: "Pentagon" like shape?
    # Just a line of 6, and 6 attached?
    coords4 = []
    for q in range(6):
        coords4.append((q, 0))
    for q in range(3):
        coords4.append((q, 1))
    for q in range(3):
        coords4.append((q+1, -1))
    clusters.append(make_cluster(coords4))

    return clusters

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620.0
    
    clusters = _generate_rigid_clusters()
    
    best_s = float('inf')
    best_inner = None
    best_outer_angle = 0.0
    best_center = np.array([0.0, 0.0])
    
    # We will sweep the relative rotation between the cluster and the outer hexagon.
    # Since the outer hexagon has 6-fold symmetry, the period is 60 degrees.
    # The cluster might have its own symmetry, but let's assume 60 deg sweep is safe.
    # Actually, the outer hexagon orientation is a free parameter.
    # Let's fix the cluster at angle 0 and sweep the outer hexagon angle from 0 to 30 deg.
    # (0 to 30 is enough due to 2-fold symmetry of the hexagon shape? No, 6-fold. 0 to 30 covers half of 60).
    
    for cluster in clusters:
        # Pre-compute vertices of the cluster at angle 0
        # This is static for this cluster iteration
        centers = cluster[:, :2]
        angles = cluster[:, 2]
        
        all_verts = np.zeros((12, 6, 2))
        for i in range(12):
            all_verts[i] = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        
        # Flatten vertices for fast projection
        # We need to rotate these vertices for each candidate outer angle?
        # No, rotating the cluster is equivalent to rotating the outer hexagon in the opposite direction.
        # So we can keep the cluster fixed and just change the outer hexagon's normals.
        
        flat_verts = all_verts.reshape(72, 2)
        
        # Sweep outer angle
        # Step size 0.1 deg initially, then refine
        n_steps = 300 # 0 to 30 deg in 0.1 deg steps
        angles_deg = np.linspace(0, 30, n_steps)
        
        min_s_for_cluster = float('inf')
        best_angle_for_cluster = 0.0
        
        for a_deg in angles_deg:
            if time.time() > deadline - 60.0:
                break
            
            s, centroid = _compute_min_enclosing_hex_size(flat_verts, a_deg)
            
            if s < min_s_for_cluster:
                min_s_for_cluster = s
                best_angle_for_cluster = a_deg
        
        if min_s_for_cluster < best_s:
            best_s = min_s_for_cluster
            best_outer_angle = best_angle_for_cluster
            best_inner = cluster.copy()
            # Note: best_inner is in the frame of the cluster.
            # The outer hexagon is at (0,0) in this frame?
            # _compute_min_enclosing_hex_size returns centroid of the vertices.
            # The evaluator expects the outer center.
            # If we return inner relative to outer center (0,0), we should shift inner by -centroid.
            # Let's store the centroid to shift later.
            # Actually, let's just return the inner positions relative to the optimal center.
            # The optimal center is the centroid returned by the function.
            # So we need to subtract that centroid from the inner positions.
            
            # Wait, _compute_min_enclosing_hex_size computes s assuming the hexagon is centered at the centroid of the vertices.
            # So the inner positions should be shifted by -centroid.
            # But we need the centroid value.
            # Let's re-calculate for the best angle to get the centroid.
            s_check, centroid_best = _compute_min_enclosing_hex_size(flat_verts, best_angle_for_cluster)
            best_inner_shifted = cluster.copy()
            best_inner_shifted[:, 0] -= centroid_best[0]
            best_inner_shifted[:, 1] -= centroid_best[1]
            best_inner = best_inner_shifted
            best_center = centroid_best # Not used in return, outer is (0,0, angle)

    # Refinement: Golden Section Search on the best cluster's angle
    if best_inner is not None:
        cluster_idx = 0 # We need to know which cluster was best.
        # Let's just re-run the sweep with finer steps on the best cluster found?
        # Or save the best cluster index.
        
        # For simplicity, let's do a fine sweep on ALL clusters again if time permits,
        # but only around the best angles found?
        # No, let's just refine the global best.
        
        # We need the unshifted cluster to rotate? No, the cluster is fixed.
        # We just need to refine the outer angle.
        
        # Let's find which cluster was the best.
        # We didn't store it. Let's assume the first one or re-evaluate?
        # Let's store the best cluster data.
        
        # Refined approach:
        # Keep track of best_cluster_data.
        
    # Let's restart the loop with tracking.
    
    best_s = float('inf')
    best_inner = None
    best_outer_angle = 0.0
    best_cluster_data = None
    
    for cluster in clusters:
        centers = cluster[:, :2]
        angles = cluster[:, 2]
        
        all_verts = np.zeros((12, 6, 2))
        for i in range(12):
            all_verts[i] = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        flat_verts = all_verts.reshape(72, 2)
        
        # Coarse sweep
        n_steps = 600 # 0 to 30 deg, 0.05 deg steps
        angles_deg = np.linspace(0, 30, n_steps)
        
        min_s_for_cluster = float('inf')
        best_angle_for_cluster = 0.0
        
        for a_deg in angles_deg:
            if time.time() > deadline - 100.0:
                break
            
            s, centroid = _compute_min_enclosing_hex_size(flat_verts, a_deg)
            
            if s < min_s_for_cluster:
                min_s_for_cluster = s
                best_angle_for_cluster = a_deg
        
        if min_s_for_cluster < best_s:
            best_s = min_s_for_cluster
            best_outer_angle = best_angle_for_cluster
            best_cluster_data = flat_verts
            # Store the cluster itself for shifting
            best_cluster_raw = cluster.copy()
            
    # Fine sweep on the best cluster
    if best_cluster_data is not None:
        # Golden section search or fine linear sweep around best_outer_angle
        # Range: best_outer_angle - 0.1 to best_outer_angle + 0.1
        # Step: 0.001
        
        a_low = max(0.0, best_outer_angle - 0.1)
        a_high = min(30.0, best_outer_angle + 0.1)
        
        n_fine = 200
        fine_angles = np.linspace(a_low, a_high, n_fine)
        
        for a_deg in fine_angles:
            if time.time() > deadline - 50.0:
                break
            
            s, centroid = _compute_min_enclosing_hex_size(best_cluster_data, a_deg)
            
            if s < best_s:
                best_s = s
                best_outer_angle = a_deg
                best_centroid = centroid
        
        # Final shift
        s_final, centroid_final = _compute_min_enclosing_hex_size(best_cluster_data, best_outer_angle)
        
        # We need the raw cluster to shift
        # We didn't store best_cluster_raw in the second loop properly.
        # Let's re-generate the best cluster?
        # We can just use the first cluster if it was the best?
        # No, let's store the cluster index.
        
    # Let's rewrite the main loop to store the cluster index.
    
    best_s = float('inf')
    best_inner = None
    best_outer_angle = 0.0
    best_cluster_idx = 0
    
    for idx, cluster in enumerate(clusters):
        centers = cluster[:, :2]
        angles = cluster[:, 2]
        
        all_verts = np.zeros((12, 6, 2))
        for i in range(12):
            all_verts[i] = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        flat_verts = all_verts.reshape(72, 2)
        
        # Coarse sweep
        n_steps = 600
        angles_deg = np.linspace(0, 30, n_steps)
        
        min_s_for_cluster = float('inf')
        best_angle_for_cluster = 0.0
        
        for a_deg in angles_deg:
            if time.time() > deadline - 100.0:
                break
            
            s, centroid = _compute_min_enclosing_hex_size(flat_verts, a_deg)
            
            if s < min_s_for_cluster:
                min_s_for_cluster = s
                best_angle_for_cluster = a_deg
        
        if min_s_for_cluster < best_s:
            best_s = min_s_for_cluster
            best_outer_angle = best_angle_for_cluster
            best_cluster_idx = idx
            best_flat_verts = flat_verts
            
    # Fine sweep
    if best_flat_verts is not None:
        a_low = max(0.0, best_outer_angle - 0.1)
        a_high = min(30.0, best_outer_angle + 0.1)
        n_fine = 200
        fine_angles = np.linspace(a_low, a_high, n_fine)
        
        for a_deg in fine_angles:
            if time.time() > deadline - 50.0:
                break
            
            s, centroid = _compute_min_enclosing_hex_size(best_flat_verts, a_deg)
            
            if s < best_s:
                best_s = s
                best_outer_angle = a_deg
                best_centroid = centroid
        
        # Final shift
        s_final, centroid_final = _compute_min_enclosing_hex_size(best_flat_verts, best_outer_angle)
        
        cluster = clusters[best_cluster_idx]
        best_inner = cluster.copy()
        best_inner[:, 0] -= centroid_final[0]
        best_inner[:, 1] -= centroid_final[1]
        
        # Outer hexagon center is (0,0) in the shifted frame
        outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
        
        return best_inner, outer_hex_data, float(best_s)
        
    # Fallback
    cluster = clusters[0]
    return cluster, np.array([0.0, 0.0, 0.0]), float(np.sqrt(12))

# EVOLVE-BLOCK-END
