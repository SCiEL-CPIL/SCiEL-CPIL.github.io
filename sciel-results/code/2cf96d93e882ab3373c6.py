import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    deadline = time.time() + 145.0
    
    # Hexagon vertices for side 1, centered at origin, flat-top (vertex at angle 0)
    # This matches the seed's base_verts: angles 0, 60, ...
    base_verts = np.array([
        [math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))] 
        for i in range(6)
    ])
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        R = np.array([[ca, -sa], [sa, ca]])
        return cx + side * (base_verts @ R.T)[:, 0] # Wait, base_verts is (6,2). 
        # Correct transformation:
        # v' = R * (v * side) + c
        # vertices = (side * base_verts) @ R.T + c
        # Let's do it properly.
        # base_verts shape (6,2)
        # R shape (2,2)
        # rotated = base_verts @ R.T  -> (6,2)
        # scaled = rotated * side
        # translated = scaled + [cx, cy]
        verts = (base_verts @ R.T) * side + np.array([cx, cy])
        return verts

    def compute_outer_side_smooth(inner_params, beta):
        # inner_params: flat array [x0, y0, a0, x1, y1, a1, ...]
        # Returns smooth approximation of max projection
        # We project onto 6 directions.
        # The required side S is max over 6 directions of (max_proj - min_proj) / (2 * apothem_factor)?
        # No, the container is centered at (0,0) in our local optimization? 
        # The task allows the outer hexagon to have a center (x,y). 
        # However, we can fix the outer center to (0,0) by centering the inner config?
        # The score is the side length. Translation of the whole system doesn't change size.
        # So we can assume outer center is (0,0).
        
        S = 0.0
        apothem_factor = math.sqrt(3) / 2.0
        
        # 6 directions for a regular hexagon centered at origin, aligned with axes?
        # If outer hex is rotated by phi, normals are at phi + 30 + 60k.
        # We also optimize the outer rotation phi.
        # Let's add phi to the parameters.
        # Actually, the current program optimizes inner positions/angles and outer center/angle.
        # Since translation is invariant, we can fix outer center to (0,0).
        # We need to optimize outer angle phi.
        
        # Let's say inner_params contains N*3 params for inner hexes, and 1 param for outer angle.
        # Total 34 params.
        
        phi = inner_params[-1]
        
        # Directions
        angles = phi + math.radians(30 + 60 * np.arange(6))
        normals = np.array([np.cos(angles), np.sin(angles)]).T # (6,2)
        
        # For each direction, find max projection of all inner hex vertices
        # To make it smooth, we use log-sum-exp on the projections?
        # max_proj = logsumexp(projections * beta) / beta
        
        # We need vertices of all inner hexes.
        # This is expensive to compute in a pure Python loop inside the objective.
        # But N=11 is small.
        
        max_projs = np.zeros(6)
        
        for i in range(N):
            x = inner_params[i*3]
            y = inner_params[i*3+1]
            a = inner_params[i*3+2]
            verts = get_hex_vertices(x, y, 1.0, a)
            
            # Project all 6 vertices onto the 6 normals
            # proj shape (6, 6) -> rows are vertices, cols are normals
            projs = verts @ normals.T
            
            # For each normal (column), find max projection
            # Use smooth max
            # max_proj_j = logsumexp(projs[:, j] * beta) / beta
            # We can do this for all 6 normals at once.
            # projs * beta -> (6,6)
            # logsumexp along axis 0
            # scipy.special.logsumexp
            
            from scipy.special import logsumexp
            smooth_max = logsumexp(projs * beta, axis=0) / beta
            
            # Combine with existing max_projs? 
            # We need global max over all hexes for each direction.
            # max(global_max_j) = max(max(hex1_j), max(hex2_j), ...)
            # This is not simply logsumexp of the smooth maxes.
            # But max(a, b) approx logsumexp(a,b)/beta.
            # So we can accumulate.
            
            # However, logsumexp is associative in a loose sense if we just take the max of the set of all projections.
            # Better: collect all projections into one big array?
            # 11 hexes * 6 verts = 66 points.
            # Project 66 points onto 6 normals.
            # Then smooth max over 66 points for each normal.
            
            # Let's refactor to collect all vertices first.
            pass
        
        # Refactor: Collect all vertices
        all_verts = []
        for i in range(N):
            x = inner_params[i*3]
            y = inner_params[i*3+1]
            a = inner_params[i*3+2]
            verts = get_hex_vertices(x, y, 1.0, a)
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts) # (66, 2)
        
        # Project onto normals
        projs = all_verts @ normals.T # (66, 6)
        
        # Smooth max over axis 0 for each of the 6 directions
        from scipy.special import logsumexp
        max_projs = logsumexp(projs * beta, axis=0) / beta # (6,)
        
        # The container must contain all these points.
        # For a hexagon centered at 0 with side S and rotation phi:
        # The distance from center to side in direction theta is S * apothem_factor.
        # So we need max_proj <= S * apothem_factor for all 6 directions?
        # Wait, the projection of the container boundary in direction n is S * apothem_factor.
        # So S = max(max_projs) / apothem_factor.
        
        S = np.max(max_projs) / apothem_factor
        return S

    def compute_overlap_penalty(inner_params, beta):
        # Penalty for overlapping inner hexagons.
        # For each pair, compute min distance between edges? Or just check vertex-in-polygon?
        # Differentiable penalty is hard.
        # Alternative: Use a grid-based penalty? Or just use a hard constraint in the optimizer?
        # L-BFGS-B doesn't support non-smooth constraints well.
        # Let's use a penalty based on the minimum distance between convex hulls?
        # Or simpler: For each pair, if they overlap, add penalty.
        # We can approximate overlap by checking if any vertex of A is inside B or vice versa.
        # This is non-differentiable.
        
        # Better approach: Use the "separation axis" theorem.
        # Two convex polygons are disjoint if there exists a separating axis.
        # For hexagons, we can check the 12 axes (6 from A, 6 from B).
        # If they overlap, the projection intervals on all axes intersect.
        # We can define a penalty as the sum of overlaps on all axes?
        # Overlap on axis n = max(min_A, min_B) - min(max_A, max_B)
        # This is non-differentiable. Smooth it with logsumexp?
        
        # Given the complexity, let's try a simpler strategy:
        # 1. Start with a valid lattice configuration (no overlaps).
        # 2. Use the smooth objective to minimize S.
        # 3. The optimizer might drift into invalid (overlapping) states.
        # 4. We can add a large penalty for overlaps using a non-smooth check, but L-BFGS-B needs gradients.
        # 5. Instead, let's use SLSQP or trust-constr with a constraint function?
        #    But constraint functions also need to be smooth.
        
        # Alternative: Keep the hexagons on a lattice? No, we want to move them.
        
        # Let's use a penalty method with a smooth approximation of the overlap.
        # For two convex polygons, the minimum distance between them can be computed.
        # If distance < 0 (overlap), penalty is distance^2?
        # Computing min distance between two hexagons is doable.
        
        penalty = 0.0
        for i in range(N):
            vi = get_hex_vertices(inner_params[i*3], inner_params[i*3+1], inner_params[i*3+2], 1.0)
            for j in range(i+1, N):
                vj = get_hex_vertices(inner_params[j*3], inner_params[j*3+1], inner_params[j*3+2], 1.0)
                
                # Check if they overlap using separating axes
                # Axes: normals of edges of vi and vj
                axes = []
                for k in range(6):
                    p1 = vi[k]
                    p2 = vi[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-9)
                    axes.append(normal)
                    axes.append(-normal) # Redundant but safe
                    # Actually just unique normals.
                
                # Simplified: Just check if any vertex is inside the other.
                # This is not smooth.
                
                # Let's use a different penalty: Sum of squared distances between closest points?
                # Too expensive.
                
                # Let's rely on the fact that if we start close to a valid solution and use small steps,
                # we might stay valid. But SLSQP/LBFGS can take large steps.
                
                # Let's implement a smooth penalty for "vertex inside polygon".
                # For vertex v of A, distance to polygon B.
                # If inside, distance is 0 (or negative signed distance).
                # Signed distance to convex polygon can be computed as min over edges of dot(v - p1, n_edge).
                # This is a min of linear functions, which is concave, so -distance is convex.
                # We want to penalize negative signed distance.
                # Penalty = max(0, -signed_dist)^2.
                
                # Let's compute signed distance of vertices of A to B.
                # And vertices of B to A.
                
                def signed_dist_to_polygon(p, poly):
                    # poly: (6,2)
                    # Returns min distance to edges, positive if outside, negative if inside.
                    min_dist = float('inf')
                    for k in range(6):
                        p1 = poly[k]
                        p2 = poly[(k+1)%6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        normal = normal / (np.linalg.norm(normal) + 1e-9)
                        # Ensure normal points outward
                        # Centroid of poly
                        c = np.mean(poly, axis=0)
                        mid = (p1 + p2) / 2
                        if np.dot(normal, mid - c) < 0:
                            normal = -normal
                        dist = np.dot(p - p1, normal)
                        if dist < min_dist:
                            min_dist = dist
                    return min_dist

                # This is not differentiable at the min.
                # We can use logsumexp for the min.
                # min(dist_i) approx -logsumexp(-dist_i * beta) / beta.
                
                # Let's compute distances for all vertices of A to B.
                dists = []
                for v in vi:
                    # Compute distances to all 6 edges of B
                    edge_dists = []
                    for k in range(6):
                        p1 = vj[k]
                        p2 = vj[(k+1)%6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        normal = normal / (np.linalg.norm(normal) + 1e-9)
                        c = np.mean(vj, axis=0)
                        mid = (p1 + p2) / 2
                        if np.dot(normal, mid - c) < 0:
                            normal = -normal
                        dist = np.dot(v - p1, normal)
                        edge_dists.append(dist)
                    edge_dists = np.array(edge_dists)
                    # Smooth min
                    from scipy.special import logsumexp
                    smooth_min = -logsumexp(-edge_dists * beta, axis=0) / beta
                    dists.append(smooth_min)
                
                dists = np.array(dists)
                # If any dist < 0, they overlap.
                # Penalty = sum(max(0, -dists)^2)
                # max(0, x) approx logsumexp(x*beta, 0)/beta? No.
                # Let's just use a simple penalty: sum of negative parts squared.
                neg_dists = np.maximum(0, -dists)
                penalty += np.sum(neg_dists**2)
                
        return penalty

    def objective(params, beta):
        S = compute_outer_side_smooth(params, beta)
        pen = compute_overlap_penalty(params, beta)
        return S + 10.0 * pen

    # Initial guess: Lattice
    sp = math.sqrt(3)
    init_params = []
    # Center hex
    init_params.extend([0.0, 0.0, 0.0])
    # Ring 1
    for i in range(6):
        theta = math.radians(60 * i)
        init_params.extend([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    # Ring 2 (partial)
    # We need 11 total. 1 + 6 = 7. Need 4 more.
    # Place them in the gaps of ring 2?
    # Or just extend the lattice.
    # Let's use the lattice points closest to origin.
    # Triangular lattice points:
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * sp + j * sp / 2
            y = j * sp * math.sqrt(3) / 2
            pts.append((x, y))
    pts.sort(key=lambda p: p[0]**2 + p[1]**2)
    pts = pts[:N]
    
    init_params = []
    for x, y in pts:
        init_params.extend([x, y, 0.0])
    
    # Add outer angle
    init_params.append(0.0)
    
    init_params = np.array(init_params)
    
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False
    
    def check_validity(inner, S, angle):
        # Exact check
        outer_verts = get_hex_vertices(0.0, 0.0, S, angle)
        inner_verts = []
        for i in range(N):
            x = inner[i*3]
            y = inner[i*3+1]
            a = inner[i*3+2]
            inner_verts.append(get_hex_vertices(x, y, 1.0, a))
        
        # Check inside
        def point_in_polygon(p, polygon):
            x, y = p
            n = len(polygon)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside

        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        
        # Check disjoint
        def segments_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(N):
            for j in range(i+1, N):
                # Vertex in polygon check
                for v in inner_verts[i]:
                    if point_in_polygon(v, inner_verts[j]):
                        return False
                for v in inner_verts[j]:
                    if point_in_polygon(v, inner_verts[i]):
                        return False
                for k in range(6):
                    a1, a2 = inner_verts[i][k], inner_verts[i][(k+1)%6]
                    b1, b2 = inner_verts[j][k], inner_verts[j][(k+1)%6]
                    if segments_intersect(a1, a2, b1, b2):
                        return False
        return True

    def update_best(params):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        inner = []
        for i in range(N):
            inner.append([params[i*3], params[i*3+1], params[i*3+2]])
        angle = params[-1]
        S = compute_outer_side_smooth(params, 100.0) # High beta for accuracy
        if check_validity(inner, S, angle):
            if S < best_S:
                best_S = S
                best_inner = inner
                best_center = (0.0, 0.0)
                best_angle = angle
                best_valid = True
            # Also try to compute exact S
            # The smooth S is an upper bound? Or approximation?
            # logsumexp is an upper bound on max. So S_smooth >= S_exact.
            # If valid, we can compute S_exact.
            S_exact = 0.0
            apothem_factor = math.sqrt(3) / 2.0
            inner_verts = []
            for i in range(N):
                x = inner[i*3]
                y = inner[i*3+1]
                a = inner[i*3+2]
                inner_verts.append(get_hex_vertices(x, y, 1.0, a))
            
            phi = angle
            angles = phi + math.radians(30 + 60 * np.arange(6))
            normals = np.array([np.cos(angles), np.sin(angles)]).T
            
            all_verts = np.vstack(inner_verts)
            projs = all_verts @ normals.T
            max_projs = np.max(projs, axis=0)
            S_exact = np.max(max_projs) / apothem_factor
            
            if S_exact < best_S:
                best_S = S_exact
                best_inner = inner
                best_center = (0.0, 0.0)
                best_angle = angle
                best_valid = True

    # Run optimization
    # Use multiple beta values (annealing)
    betas = [10.0, 20.0, 50.0, 100.0]
    cur_params = init_params.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        res = minimize(objective, cur_params, args=(beta,), method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4})
        cur_params = res.x
        update_best(cur_params)
        
        # Try a few random restarts if time permits?
        # No, keep it simple.
        
    # Final polish with Nelder-Mead on the exact objective?
    # Exact objective is non-differentiable.
    # Let's just do a few local searches with the current best if valid.
    
    if best_valid:
        # Refine with local search similar to seed, but starting from best_inner
        # We can reuse the seed's local search logic if we want, but let's stick to the new family.
        # Maybe run L-BFGS with higher beta?
        res = minimize(objective, cur_params, args=(200.0,), method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
        update_best(res.x)

    if not best_valid:
        # Fallback
        sp = math.sqrt(3)
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_side_smooth(np.array([p for row in fallback_inner for p in row] + [0.0]), 100.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle

