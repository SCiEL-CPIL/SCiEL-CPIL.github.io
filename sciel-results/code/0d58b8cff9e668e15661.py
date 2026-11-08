import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def hex_verts(cx, cy, s, rot_deg):
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_dist_polygons_fast(v1, v2):
        # Fast approximate min distance for convex polygons.
        # Returns 0 if intersecting, else min distance.
        # Uses vertex-to-edge distance.
        min_d = float('inf')
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        for i in range(6):
            p1 = v2[i]
            for j in range(6):
                p2 = v1[j]
                p3 = v1[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        return min_d

    def get_cluster_verts(cluster_params, s):
        # cluster_params: (cx, cy, rot_deg)
        cx, cy, rot = cluster_params
        verts = []
        # 6 hexagons in a cluster: 1 center, 5 surrounding? No, 6 in a ring or 2x3?
        # Let's use a 2x3 grid of hexagons (side 1) as the cluster shape.
        # Hexagon side 1. Distance between centers in a grid:
        # dx = 1.5, dy = sqrt(3) for offset rows.
        # Row 0: (0, 0), (1.5, sqrt(3)/2) -- wait, standard hex grid.
        # Let's use a compact cluster: 1 center, 6 neighbors? That's 7.
        # We need 6.
        # A "hexagon of hexagons" of size 2 has 7 hexagons.
        # A cluster of 6 could be a "trapezoid" or a "ring of 5 + 1".
        # Let's try a 2x3 grid.
        # Coordinates for 2x3 grid of unit hexagons (flat top):
        # (0,0), (1.5, sqrt(3)/2)
        # (0, sqrt(3)), (1.5, 3*sqrt(3)/2)
        # (0, 2*sqrt(3)), (1.5, 5*sqrt(3)/2)
        # This is 6 hexagons.
        offsets = np.array([
            [0.0, 0.0],
            [1.5, np.sqrt(3)/2],
            [0.0, np.sqrt(3)],
            [1.5, 3*np.sqrt(3)/2],
            [0.0, 2*np.sqrt(3)],
            [1.5, 5*np.sqrt(3)/2]
        ])
        
        # Center of mass of these offsets
        com = np.mean(offsets, axis=0)
        offsets = offsets - com
        
        rot_rad = np.deg2rad(rot)
        rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                            [np.sin(rot_rad), np.cos(rot_rad)]])
        
        rotated_offsets = (offsets @ rot_mat.T)
        
        for off in rotated_offsets:
            v = hex_verts(cx + off[0], cy + off[1], s, rot)
            verts.append(v)
        return verts

    def evaluate_violation(params, outer_s):
        # params: [cx1, cy1, rot1, cx2, cy2, rot2]
        # Two clusters of 6 hexagons
        cluster1_params = (params[0], params[1], params[2])
        cluster2_params = (params[3], params[4], params[5])
        
        verts1 = get_cluster_verts(cluster1_params, 1.0)
        verts2 = get_cluster_verts(cluster2_params, 1.0)
        
        all_verts = verts1 + verts2
        
        violation = 0.0
        r = outer_s * np.sqrt(3) / 2.0
        normals = np.deg2rad(30 + np.arange(6)*60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        # Containment violation
        for v in all_verts:
            dists = np.dot(v, n_vecs.T)
            out = np.maximum(0, dists - r)
            violation += np.sum(out)
        
        # Overlap violation (all pairs)
        # 12 hexagons, 66 pairs.
        for i in range(12):
            for j in range(i+1, 12):
                d = min_dist_polygons_fast(all_verts[i], all_verts[j])
                if d < 1e-4:
                    # Penalty for overlap
                    # Since min_dist is 0 for overlap, we need a better metric or just assume no overlap if we are careful.
                    # However, for optimization, we need a gradient.
                    # The min_dist_polygons_fast is not differentiable at 0.
                    # We can use a soft penalty: if d < 0 (overlap), penalty is large.
                    # But min_dist returns 0 for overlap.
                    # Let's use a different check for optimization:
                    # We can approximate overlap by checking if distance between centers is < 2*sqrt(3)/2 * some factor?
                    # No, rotations make it hard.
                    # Let's rely on the fact that the cluster internal structure is fixed and non-overlapping.
                    # We only need to check inter-cluster overlaps and intra-cluster overlaps are 0.
                    # Intra-cluster: fixed, non-overlapping.
                    # Inter-cluster: 6x6 = 36 pairs.
                    # We can just penalize if d < 0? But d is 0.
                    # Let's use a "penetration depth" approximation.
                    # For convex polygons, if they intersect, we can compute the minimum translation vector? Too slow.
                    # Let's just use a large penalty if d < 1e-3.
                    violation += 100.0 * (1e-3 - d)**2 if d < 1e-3 else 0.0
                    # Actually, if d=0, (1e-3 - 0)^2 * 100 = 1e-4. Small.
                    # We need a stronger penalty.
                    # Let's check center distance as a proxy for quick rejection in optimization.
                    # But we need it in the objective.
                    # Let's try: if d < 0.1, penalty = (0.1 - d)^2 * 1000
                    if d < 0.1:
                        violation += (0.1 - d)**2 * 1000.0
        return violation

    def objective(params):
        return evaluate_violation(params, current_outer_s)

    def get_initial_params():
        # Cluster 1 at (-1.5, 0), rot 0
        # Cluster 2 at (1.5, 0), rot 0
        # They might overlap initially if close.
        # 2x3 grid width is approx 1.5 + 0.75 = 2.25?
        # Let's place them further apart.
        return np.array([-2.0, 0.0, 0.0, 2.0, 0.0, 0.0])

    LB = 3.0
    UB = 10.0
    best_solution = None
    best_s = float('inf')
    
    max_iter_lbfgs = 200
    tol = 1e-6
    
    for step in range(30):
        if time.time() > deadline:
            break
        mid_s = (LB + UB) / 2.0
        current_outer_s = mid_s
        found_valid = False
        best_violation = float('inf')
        
        for restart in range(5):
            if time.time() > deadline:
                break
            x0 = get_initial_params()
            # Add some noise
            x0 += np.random.normal(0, 0.2, x0.shape)
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': max_iter_lbfgs, 'ftol': 1e-8, 'gtol': 1e-8})
                if res.fun < best_violation:
                    best_violation = res.fun
                    best_params = res.x
                if res.fun < 1e-4:
                    found_valid = True
                    best_params = res.x
                    break
            except Exception:
                pass
        
        if found_valid:
            UB = mid_s
            best_solution = best_params.copy()
            best_s = mid_s
        else:
            LB = mid_s
        
    if best_solution is None:
        # Fallback: return a valid but loose configuration
        # Just place them in a big hexagon
        inner = []
        for i in range(12):
            x = (i % 4) * 2.5 - 3.75
            y = (i // 4) * 2.2 - 2.2
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    # Extract positions for output
    # We need to return 12 inner hexagons: (x, y, angle)
    cluster1_params = (best_solution[0], best_solution[1], best_solution[2])
    cluster2_params = (best_solution[3], best_solution[4], best_solution[5])
    
    verts1 = get_cluster_verts(cluster1_params, 1.0)
    verts2 = get_cluster_verts(cluster2_params, 1.0)
    
    inner = []
    all_verts = verts1 + verts2
    for v in all_verts:
        # Center is the mean of vertices
        cx, cy = np.mean(v, axis=0)
        # Angle is the cluster rotation
        # All hexagons in a cluster have the same rotation
        if len(inner) < 6:
            rot = cluster1_params[2]
        else:
            rot = cluster2_params[2]
        inner.append([cx, cy, rot])
        
    return inner, (0.0, 0.0), best_s, 0.0
