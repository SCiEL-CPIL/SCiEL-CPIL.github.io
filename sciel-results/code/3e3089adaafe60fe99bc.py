"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_distance_polygons(p1, p2):
        """
        Compute the minimum distance between two convex polygons.
        Returns 0 if they overlap or touch.
        Uses a simple vertex-to-edge distance check, which is sufficient for 
        gradient estimation if we smooth it, but here we use a robust SAT-based 
        check for validity and a smooth approximation for optimization.
        For the gradient step, we will use a soft penalty based on center distance 
        and vertex separation.
        """
        # For optimization, we need a differentiable metric.
        # A common approach is to use the minimum distance between any pair of vertices/edges.
        # However, min() is not differentiable. 
        # We will use a smooth approximation: -log-sum-exp of negative squared distances? 
        # Or simpler: penalty = sum(max(0, overlap_depth)^2).
        # Let's stick to a robust SAT check for the final result, 
        # and for the objective function, use a smooth penalty based on 
        # the minimum distance between all vertex pairs of the two hexagons.
        
        # Distance between all pairs of vertices is O(36).
        # min_dist = min ||v1_i - v2_j||
        # This is not differentiable at min, but we can use a soft-min.
        # soft_min(x) = -1/b * log(sum(exp(-b*x_i)))
        
        dists = []
        for v1 in p1:
            for v2 in p2:
                d = np.linalg.norm(v1 - v2)
                dists.append(d)
        dists = np.array(dists)
        b = 10.0 # Sharpness of the soft-min
        soft_min_dist = -1.0 / b * np.log(np.sum(np.exp(-b * dists)))
        return soft_min_dist

    def objective(params):
        """
        Minimize the outer hexagon side length S.
        params: [cx1, cy1, a1, cx2, cy2, a2, ...]
        """
        n = len(params) // 3
        vertices_list = []
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        # Calculate required outer side length S
        # The outer hexagon is centered at (0,0) with angle 0 for now.
        # S is determined by the maximum projection of all inner vertices onto the 6 normals.
        apothem = np.sqrt(3) / 2
        S = 0.0
        for verts in vertices_list:
            # Project onto 6 normals of the container hexagon
            # Normals are at angles pi/6 + k*pi/3
            for k in range(6):
                theta_n = np.pi / 6 + k * np.pi / 3
                cos_n = np.cos(theta_n)
                sin_n = np.sin(theta_n)
                # Projection of all vertices in this hexagon
                projs = verts[:, 0] * cos_n + verts[:, 1] * sin_n
                max_proj = np.max(np.abs(projs))
                S = max(S, max_proj)
        
        S = S / apothem
        
        # Penalty for overlaps
        # We want the inner hexagons to be disjoint.
        # If they overlap, the minimum distance between their boundaries is negative.
        # We approximate this by checking the minimum distance between vertices.
        # If min_dist < 0, they are overlapping.
        # We use a smooth penalty: sum of squared overlaps.
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # Quick rejection based on center distance
                c1 = np.array([params[3*i], params[3*i+1]])
                c2 = np.array([params[3*j], params[3*j+1]])
                dist_centers = np.linalg.norm(c1 - c2)
                if dist_centers < 2.5: # Threshold for potential overlap
                    d = min_distance_polygons(vertices_list[i], vertices_list[j])
                    # If d is small, they are close or overlapping.
                    # We want d >= 0. 
                    # Penalty if d < 0 is not possible with min distance of vertices if they are disjoint.
                    # Wait, min_distance_polygons returns distance between vertices.
                    # If polygons overlap, min distance between vertices can be > 0? No, if they overlap, 
                    # some vertex of one might be inside the other, but min distance between *vertices* 
                    # is not the correct metric for overlap depth.
                    
                    # Better: Use SAT to check overlap, but SAT is non-diff.
                    # Alternative: Use the fact that for convex polygons, 
                    # they are disjoint iff there exists a separating axis.
                    # For optimization, a common trick is to minimize the sum of squared 
                    # penetration depths along a set of candidate axes.
                    
                    # Let's use a simpler, robust penalty:
                    # Calculate the minimum distance between the two sets of vertices.
                    # If they are disjoint, this is > 0.
                    # If they overlap, this metric fails to give a negative value.
                    
                    # Let's use a different metric:
                    # Project both polygons onto the normal of each edge of both polygons.
                    # If any projection intervals do not overlap, they are disjoint.
                    # If they overlap on all axes, they are overlapping.
                    # The "overlap depth" on an axis is the amount by which the intervals overlap.
                    # We want the minimum overlap depth over all axes to be >= 0? No, if they are disjoint, 
                    # at least one axis has 0 overlap (separation).
                    # So we want to maximize the minimum separation?
                    # Or minimize the maximum overlap?
                    
                    # Let's compute the overlap depth for each of the 12 axes (6 from poly i, 6 from poly j).
                    # overlap_depth = min(max1, max2) - max(min1, min2)
                    # If overlap_depth < 0, they are separated on this axis.
                    # If overlap_depth >= 0 for all axes, they overlap.
                    # We want to push them apart. So we want to make at least one axis have negative overlap.
                    # We can penalize the sum of positive overlap depths.
                    
                    p1 = vertices_list[i]
                    p2 = vertices_list[j]
                    overlap_penalty = 0.0
                    for poly in [p1, p2]:
                        n_pts = len(poly)
                        for k in range(n_pts):
                            p1v = poly[k]
                            p2v = poly[(k + 1) % n_pts]
                            edge = p2v - p1v
                            normal = np.array([-edge[1], edge[0]])
                            # Normalize normal
                            norm_len = np.linalg.norm(normal)
                            if norm_len > 1e-9:
                                normal /= norm_len
                            
                            proj1 = p1 @ normal
                            proj2 = p2 @ normal
                            min1, max1 = proj1.min(), proj1.max()
                            min2, max2 = proj2.min(), proj2.max()
                            
                            # Calculate overlap
                            # Intervals [min1, max1] and [min2, max2]
                            # Overlap = min(max1, max2) - max(min1, min2)
                            ov = min(max1, max2) - max(min1, min2)
                            if ov > 0:
                                # They overlap on this axis
                                overlap_penalty += ov ** 2
                                
                    penalty += overlap_penalty
        
        return S + 100.0 * penalty

    # Initial guess: Hexagonal lattice
    # Center at (0,0)
    # 6 around it, 4 more?
    # Standard 11-hex packing:
    # 1 in center
    # 6 in first ring
    # 4 in second ring? No, that's 11.
    # Positions:
    # (0,0)
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # The next ring would have 12. We need 4 more.
    # Maybe a different arrangement?
    # Or just start with a random tight packing.
    
    params = np.zeros(3 * N)
    # Center
    params[0:3] = [0, 0, 0]
    
    # Ring 1
    r1 = np.sqrt(3)
    for k in range(6):
        theta = k * np.pi / 3
        cx = r1 * np.cos(theta)
        cy = r1 * np.sin(theta)
        idx = 1 + k
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        
    # Ring 2 (partial)
    # We need 4 more.
    # Place them in the "corners" of the hexagon formed by ring 1.
    # The corners are at distance 2*sqrt(3) from center?
    # Or just add 4 more at the end of the hexagon axes?
    # Let's place them at (2*sqrt(3), 0), (-2*sqrt(3), 0), (sqrt(3), 3), (-sqrt(3), -3)?
    # This might not be optimal, but it's a valid start.
    r2 = 2 * np.sqrt(3)
    for k in range(4):
        # Place at 0, 90, 180, 270 degrees? No, that's square.
        # Place at the vertices of the hexagon?
        # Hexagon vertices are at angles 0, 60, ...
        # Let's place them at angles 30, 150, 210, 330?
        # Distance from center to vertex of unit hex is 1.
        # Distance from center to vertex of ring 1 hex is sqrt(3)+1? No.
        # Let's just use a random perturbation of a tight grid.
        pass
        
    # Better initialization:
    # Use a known tight packing or random.
    # Let's use a random tight cluster.
    np.random.seed(42)
    params_init = np.zeros(3 * N)
    # Place them in a circle
    for i in range(N):
        if i == 0:
            params_init[0:3] = [0, 0, 0]
        else:
            angle = (i - 1) * 2 * np.pi / (N - 1)
            r = 1.7 # Approximate distance for touching
            params_init[3 * i] = r * np.cos(angle)
            params_init[3 * i + 1] = r * np.sin(angle)
            params_init[3 * i + 2] = 0

    best_params = params_init
    best_score = objective(params_init)

    # Optimize
    # L-BFGS-B is good for this
    res = minimize(objective, best_params, method='L-BFGS-B', 
                   options={'maxiter': 500, 'ftol': 1e-10})
    
    if res.fun < best_score:
        best_params = res.x
        best_score = res.fun

    # Refine with Powell to handle non-differentiable bits better if needed
    # But L-BFGS-B should be good enough if the penalty is smooth.
    
    # Extract results
    inner = []
    vertices_list = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
        vertices_list.append(hex_vertices(cx, cy, angle))

    # Find best outer container
    # The objective assumed outer at (0,0) rot 0.
    # We can refine the outer container position and rotation.
    def find_best_container(vertices_list):
        def cont_obj(params_c):
            ocx, ocy, oangle = params_c
            cos_a = np.cos(oangle)
            sin_a = np.sin(oangle)
            S = 0.0
            apothem = np.sqrt(3) / 2
            for verts in vertices_list:
                for v in verts:
                    lx = v[0] - ocx
                    ly = v[1] - ocy
                    rx = lx * cos_a + ly * sin_a
                    ry = -lx * sin_a + ly * cos_a
                    for k in range(6):
                        theta_n = np.pi / 6 + k * np.pi / 3
                        proj = rx * np.cos(theta_n) + ry * np.sin(theta_n)
                        S = max(S, abs(proj))
            return S / apothem

        res_c = minimize(cont_obj, [0, 0, 0], method='Powell', options={'maxiter': 100, 'maxfev': 500})
        return res_c.x, res_c.fun

    cont_params, S = find_best_container(vertices_list)
    ocx, ocy, oangle = cont_params
    oangle_deg = np.degrees(oangle)

    return inner, (ocx, ocy), S, oangle_deg
# EVOLVE-BLOCK-END
