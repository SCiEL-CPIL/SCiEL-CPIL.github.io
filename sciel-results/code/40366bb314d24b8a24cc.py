# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S(vertices_list):
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = v[0] * np.cos(theta_n) + v[1] * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem

    def min_distance(v1, v2):
        """
        Calculate the minimum distance between two convex polygons.
        Returns negative distance if they overlap.
        Uses SAT to check overlap and then finds min distance if not overlapping.
        For simplicity and speed in optimization, we can use a rough estimate:
        dist = min_dist_between_points - sum_inradii? No, that's for circles.
        
        Better: Use the separating axis theorem to find the minimum separation.
        If no separating axis, they overlap.
        To get a smooth penalty, we can calculate the minimum distance between
        any edge of v1 and any edge of v2, and vertex to edge.
        """
        # This is computationally expensive. 
        # Alternative: Use a softer metric.
        # Let's stick to the current SAT check but make the penalty quadratic and smoother.
        # If we can't easily get smooth distance, we use a heuristic.
        # However, for 11 hexagons, the number of pairs is 55.
        # SAT is O(12) per pair. 55 * 12 = 660 ops. Very fast.
        
        # To get a distance value for penalty:
        # If they overlap, we want a negative value proportional to penetration depth.
        # Penetration depth is the minimum shift needed to separate them.
        # In SAT, for each axis, we find the overlap interval.
        # The minimum overlap length over all axes is the penetration depth.
        
        min_overlap = np.inf
        overlapping = True
        
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                # Normalize normal? Not strictly necessary for SAT overlap check,
                # but for depth calculation, we need the projection width.
                norm_len = np.linalg.norm(normal)
                if norm_len == 0: continue
                normal_unit = normal / norm_len
                
                proj1 = v1 @ normal_unit
                proj2 = v2 @ normal_unit
                
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                
                # Check separation
                if max1 < min2 or max2 < min1:
                    overlapping = False
                    break
                
                # Calculate overlap on this axis
                overlap = min(max1, max2) - max(min1, min2)
                if overlap < min_overlap:
                    min_overlap = overlap
        
        if overlapping:
            return -min_overlap
        else:
            # If not overlapping, we need the actual distance to make the gradient smooth?
            # Or we can just return 0.
            # If we return 0, the optimizer might not know how to push them apart
            # if they are close but not overlapping.
            # But the objective is to minimize S. S increases if they move apart.
            # So the optimizer will try to keep them close.
            # If they are not overlapping, penalty is 0.
            return 0.0

    def objective(params):
        n = len(params) // 3
        vertices_list = []
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S(vertices_list)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = min_distance(vertices_list[i], vertices_list[j])
                if dist < 0:
                    # Quadratic penalty for overlap
                    penalty += 100.0 * dist * dist
        
        return S + penalty

    # Initial guess: Hexagonal lattice
    # We want to pack them tightly.
    # Center at 0,0.
    # Ring 1: 6 hexagons at distance sqrt(3).
    # Ring 2: We need 4 more.
    # Place them at gaps?
    
    params = np.zeros(3 * N)
    params[0:3] = [0, 0, 0]
    idx = 1
    for k in range(6):
        theta = k * np.pi / 3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1
    # 4 more hexagons. Place them further out, but optimized.
    # Start them at radius 2*sqrt(3)
    for k in range(4):
        theta = k * np.pi / 3
        cx = 2 * np.sqrt(3) * np.cos(theta)
        cy = 2 * np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1

    best_params = params
    best_score = objective(params)
    
    # Multi-start optimization
    # Use Nelder-Mead for robustness
    for _ in range(20):
        # Perturb initial guess
        p = params.copy()
        if _ > 0:
            p += np.random.normal(0, 0.5, size=p.size)
        
        res = minimize(objective, p, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x

    opt_params = best_params
    inner = []
    for i in range(N):
        cx = opt_params[3 * i]
        cy = opt_params[3 * i + 1]
        angle = np.degrees(opt_params[3 * i + 2])
        inner.append([cx, cy, angle])
    
    # Compute the final S
    vertices_list = []
    for i in range(N):
        cx = opt_params[3 * i]
        cy = opt_params[3 * i + 1]
        angle = opt_params[3 * i + 2]
        verts = hex_vertices(cx, cy, angle)
        vertices_list.append(verts)
    S = compute_S(vertices_list)
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
