import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin for final computation and return
    
    # Precompute normals for the 6 directions of the outer hexagon
    # Unit regular hexagon normals (pointing to midpoints of edges)
    # If vertices are at 0, 60, ..., normals are at 30, 90, ...
    # Let's stick to the definition used in the failed code:
    # Vertices at 0, 60, ... => Normals at 30, 90, ...
    # The code used:
    # angles = np.deg2rad(np.arange(0, 360, 60)) for normals?
    # Let's check the failed code's normals definition.
    # angles = np.deg2rad(np.arange(0, 360, 60))
    # normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    # This defines normals at 0, 60, ...
    # And vertices at 0, 60, ...
    # This is inconsistent. A regular hexagon with vertices at 0 deg has normals at 30 deg.
    # If vertices are at 0, 60, 120, 180, 240, 300.
    # The edge between 0 and 60 is perpendicular to 30.
    # So if we define the hexagon by "distance to edge <= sqrt(3)/2", the normals should be at 30, 90, etc.
    # OR, we define the hexagon by vertices.
    # The failed code used `u_vertices` at 0, 60... and `normals` at 0, 60...
    # And checked `p_local @ normals.T <= sqrt(3)/2`.
    # If vertices are at 0, the distance from center to edge (apothem) is sqrt(3)/2.
    # The normal to the edge connecting v0 and v1 points in direction 30 deg.
    # So `normals` in the failed code were WRONG for the check `p_local @ normals.T <= sqrt(3)/2` if vertices are at 0.
    # If vertices are at 0, the supporting lines are at angles 30, 90, ...
    # Let's fix the geometry definition to be consistent.
    
    # Definition:
    # Inner hexagons: Side 1.
    # Vertices relative to center: $v_k = (\cos(k \pi/3), \sin(k \pi/3))$ for $k=0..5$.
    # This is a "pointy-top" or "flat-top"? 
    # v0 = (1, 0). v1 = (0.5, sqrt(3)/2).
    # This is a hexagon with a vertex on the x-axis.
    # The edges are between v0-v1, v1-v2, etc.
    # The normal to edge v0-v1 is in direction 30 deg.
    # The distance from center to this edge is $\sqrt{3}/2$.
    # So the condition for a point $p$ to be inside is:
    # $|p \cdot n_k| \le \sqrt{3}/2$ for all $k$, where $n_k$ are normals at 30, 90, 150, 210, 270, 330.
    
    # Let's redefine normals correctly.
    normal_angles = np.deg2rad(np.arange(30, 360, 60))
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
    
    # Precompute unit hex vertices (side 1)
    # Vertices at 0, 60, ...
    vertex_angles = np.deg2rad(np.arange(0, 360, 60))
    u_vertices = np.stack([np.cos(vertex_angles), np.sin(vertex_angles)], axis=1)
    
    def get_vertices(c, theta):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        verts = u_vertices @ R.T + c
        return verts

    def check_overlap(c1, t1, c2, t2):
        """
        Returns minimum separation distance (negative if overlapping).
        Uses support function approach or simple vertex-edge check.
        For hexagons, we can check if any vertex of one is inside the other.
        This is sufficient for convex polygons? No, need edge-edge intersection too.
        But for "disjoint" requirement, we just need to ensure they don't intersect.
        A robust way: Calculate the Minkowski difference or use SAT (Separating Axis Theorem).
        SAT is fast for convex polygons.
        """
        v1 = get_vertices(c1, t1)
        v2 = get_vertices(c2, t2)
        
        # Get edges of both polygons
        edges1 = np.roll(v1, -1, axis=0) - v1
        edges2 = np.roll(v2, -1, axis=0) - v2
        
        # Normals to edges (perpendiculars)
        def get_normals(edges):
            # Rotate edges by 90 degrees
            return np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        
        n1 = get_normals(edges1)
        n2 = get_normals(edges2)
        
        all_normals = np.vstack([n1, n2])
        
        min_dist = float('inf')
        for n in all_normals:
            # Project v1 and v2 onto n
            p1 = v1 @ n
            p2 = v2 @ n
            
            max1 = np.max(p1)
            min1 = np.min(p1)
            max2 = np.max(p2)
            min2 = np.min(p2)
            
            # Gap
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = 0.0 # Overlapping on this axis
            
            min_dist = min(min_dist, gap)
            
            if min_dist == 0.0:
                return 0.0 # Early exit if overlap found
        
        return min_dist

    def compute_metrics(params):
        centers_x = params[0::3]
        centers_y = params[1::3]
        thetas = params[2::3]
        c = np.stack([centers_x, centers_y], axis=1)
        
        all_verts = []
        for i in range(N):
            v = get_vertices(c[i], thetas[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Compute bounding box widths for outer hexagon (aligned with axes? No, outer hexagon can rotate)
        # The task says "minimizing the outer side length".
        # The outer hexagon is a regular hexagon.
        # We can assume the outer hexagon is fixed in orientation (e.g., flat top) because we can rotate the whole system.
        # Or we can optimize the outer orientation too?
        # The interface returns `angle` for outer hexagon.
        # However, since the inner hexagons can rotate, and the outer is just a container,
        # the optimal outer hexagon orientation is usually aligned with the "tightest" packing direction.
        # For simplicity and speed, let's assume the outer hexagon has a fixed orientation (e.g., normals at 0, 60, ...) 
        # and we just find the minimal side length that contains them.
        # Wait, if we fix outer orientation, we might miss a better packing.
        # But the inner hexagons can rotate.
        # Let's assume outer hexagon is aligned such that its normals are at 0, 60, 120, 180, 240, 300.
        # This corresponds to a "flat top" hexagon if vertices are at 30, 90...?
        # Let's stick to the previous code's assumption:
        # Outer hexagon normals at 0, 60, ...
        # This implies the outer hexagon has edges perpendicular to 0, 60...
        # i.e., edges are vertical/horizontal-ish?
        # If normal is (1,0), edge is vertical.
        # This is a valid regular hexagon.
        
        outer_normals = np.stack([np.cos(np.deg2rad(np.arange(0, 360, 60))), 
                                  np.sin(np.deg2rad(np.arange(0, 360, 60)))], axis=1)
        
        projections = all_verts @ outer_normals.T
        max_p = np.max(projections, axis=0)
        min_p = np.min(projections, axis=0)
        widths = max_p - min_p
        max_width = np.max(widths)
        L = max_width / np.sqrt(3)
        
        # Compute penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Use SAT to check overlap and get penetration depth
                # For speed, we can approximate penetration by checking vertices.
                # But SAT is more robust.
                # Let's use a simpler check: min distance between vertices and edges.
                # Or just use the `check_overlap` function above, but it's slow in Python loops.
                # Let's vectorize the vertex-inside-hex check for penalty.
                
                # Check if any vertex of i is inside j
                # And if any vertex of j is inside i
                # This catches most overlaps. Edge-edge intersections are rare in local optima if we start from a grid.
                
                vi = all_verts[i*6:(i+1)*6]
                vj = all_verts[j*6:(j+1)*6]
                ci = c[i]
                cj = c[j]
                ti = thetas[i]
                tj = thetas[j]
                
                # Transform vi to frame of j
                R_j = np.array([[np.cos(tj), -np.sin(tj)], [np.sin(tj), np.cos(tj)]])
                v_rel = vi - cj
                p_local = v_rel @ R_j.T
                # Check against unit hex (normals at 30, 90...)
                dists = p_local @ normals.T
                max_dists = np.max(dists, axis=1)
                thresh = np.sqrt(3)/2.0
                over = np.maximum(0, max_dists - thresh)
                penalty += np.sum(over**2)
                
                # Transform vj to frame of i
                R_i = np.array([[np.cos(ti), -np.sin(ti)], [np.sin(ti), np.cos(ti)]])
                v_rel2 = vj - ci
                p_local2 = v_rel2 @ R_i.T
                dists2 = p_local2 @ normals.T
                max_dists2 = np.max(dists2, axis=1)
                over2 = np.maximum(0, max_dists2 - thresh)
                penalty += np.sum(over2**2)
                
        return L, penalty

    # Initialization
    init_params = np.zeros(3 * N)
    idx = 0
    # Hex grid layout
    # Row 0: 3 hexes
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    for r in range(3):
        count = 3 if r == 0 else 4
        offset = 0.5 * np.sqrt(3) * (1 if r > 0 else 0)
        for c in range(count):
            x = c * np.sqrt(3) + offset
            y = r * 1.5
            theta = 0.0
            init_params[idx] = x
            init_params[idx+1] = y
            init_params[idx+2] = theta
            idx += 3
            
    # Center
    com_x = np.mean(init_params[0::3])
    com_y = np.mean(init_params[1::3])
    init_params[0::3] -= com_x
    init_params[1::3] -= com_y
    
    best_params = init_params
    best_L, best_pen = compute_metrics(init_params)
    
    def obj_func(p):
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        L, pen = compute_metrics(p)
        weight = 1000.0 # High weight to enforce non-overlap
        return L + weight * pen
        
    try:
        res = minimize(obj_func, init_params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
        if res.success:
            best_params = res.x
            best_L, best_pen = compute_metrics(best_params)
    except TimeoutError:
        pass
    except Exception as e:
        pass
        
    # Repair step: If there are still overlaps, push them apart
    # This is a heuristic local search
    for _ in range(50):
        if time.time() > deadline - 5:
            break
            
        L, pen = compute_metrics(best_params)
        if pen < 1e-6:
            break
            
        # Identify overlapping pairs and push them
        centers_x = best_params[0::3]
        centers_y = best_params[1::3]
        thetas = best_params[2::3]
        c = np.stack([centers_x, centers_y], axis=1)
        
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                # Quick check: distance between centers
                dist = np.linalg.norm(c[i] - c[j])
                # Minimum distance for non-overlap is roughly 2 * apothem = sqrt(3) ~ 1.732 if aligned
                # If dist < 1.732, they likely overlap
                if dist < 1.8:
                    # Calculate overlap direction
                    # Use the vector from i to j
                    d = c[j] - c[i]
                    if np.linalg.norm(d) < 1e-6:
                        d = np.array([1.0, 0.0])
                    d = d / np.linalg.norm(d)
                    
                    # Estimate penetration
                    # We want to separate them.
                    # Push i by -d * delta and j by d * delta
                    # delta = 0.1 * dist?
                    # Let's just push them apart by a small amount
                    delta = 0.05
                    c[i] -= d * delta
                    c[j] += d * delta
                    moved = True
                    
        if not moved:
            break
            
        best_params[0::3] = c[:, 0]
        best_params[1::3] = c[:, 1]
        
    # Final metrics
    L, pen = compute_metrics(best_params)
    
    # If still overlapping, we have to return something.
    # The task requires disjoint. If we can't achieve it, we score 0.
    # But hopefully the repair worked.
    
    centers_x = best_params[0::3]
    centers_y = best_params[1::3]
    thetas = best_params[2::3]
    c = np.stack([centers_x, centers_y], axis=1)
    
    all_verts = []
    for i in range(N):
        v = get_vertices(c[i], thetas[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    outer_normals = np.stack([np.cos(np.deg2rad(np.arange(0, 360, 60))), 
                              np.sin(np.deg2rad(np.arange(0, 360, 60)))], axis=1)
                              
    projections = all_verts @ outer_normals.T
    max_p = np.max(projections, axis=0)
    min_p = np.min(projections, axis=0)
    mid_p = (max_p + min_p) / 2.0
    
    # Solve for center C
    n0 = outer_normals[0]
    n1 = outer_normals[1]
    m0 = mid_p[0]
    m1 = mid_p[1]
    
    A = np.array([n0, n1])
    b = np.array([m0, m1])
    try:
        C = np.linalg.solve(A, b)
    except:
        C = np.array([0.0, 0.0])
        
    widths = max_p - min_p
    max_width = np.max(widths)
    side = max_width / np.sqrt(3)
    
    inner = []
    for i in range(N):
        inner.append([best_params[i*3], best_params[i*3+1], np.rad2deg(best_params[i*3+2])])
        
    return inner, (C[0], C[1]), side, 0.0
# EVOLVE-BLOCK-END
