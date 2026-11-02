import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # Vertices of a unit hexagon with one vertex at (1, 0)
    # Angles: 0, 60, 120, 180, 240, 300
    hex_angles = np.arange(6) * np.pi / 3
    hex_vertices = np.column_stack((np.cos(hex_angles), np.sin(hex_angles)))
    
    # Normals for the outer hexagon (fixed rotation 0)
    # The current program uses normals at 30, 90, ...
    # This corresponds to a hexagon with flat top/bottom?
    # Let's verify: Normal at 90 deg is (0,1). Side is horizontal.
    # Normal at 30 deg is (cos30, sin30).
    # This is a standard hexagon orientation.
    outer_normal_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    outer_normals = np.column_stack((np.cos(outer_normal_angles), np.sin(outer_normal_angles)))
    
    # Define the radial arrangement
    # 1 Center, 5 Inner, 6 Outer
    # Center: index 0
    # Inner: indices 1-5
    # Outer: indices 6-11
    
    # Angular positions (relative to global theta)
    # Center: 0
    # Inner: 0, 72, 144, 216, 288
    # Outer: 30, 90, 150, 210, 270, 330
    # Note: Using 30 offset for outer helps interlocking with 5-fold inner?
    # 5-fold: 0, 72, 144...
    # 6-fold: 30, 90...
    
    angles_radial = [0.0]
    inner_angles = [np.deg2rad(a) for a in [0, 72, 144, 216, 288]]
    outer_angles = [np.deg2rad(a) for a in [30, 90, 150, 210, 270, 330]]
    
    # Total 12 angles
    all_angles = angles_radial + inner_angles + outer_angles
    
    # Pre-calculate minimum distances D_ij for non-overlap
    # D_ij depends on relative rotation delta = beta_i - beta_j
    # beta_i = angle_i + theta
    # delta = angle_i - angle_j
    # Since theta cancels out, D_ij is constant.
    
    def min_dist_between_hexagons(delta):
        # Rotate hex_vertices by delta
        c, s = np.cos(delta), np.sin(delta)
        rot_mat = np.array([[c, -s], [s, c]])
        v2 = hex_vertices @ rot_mat.T
        
        # Minkowski difference H1 - H2
        # Points are v1 - v2
        diff_points = []
        for v1 in hex_vertices:
            for v2 in v2:
                diff_points.append(v1 - v2)
        diff_points = np.array(diff_points)
        
        # The set H1 - H2 is the convex hull of these points
        # We need the minimum distance from origin to the boundary of this convex hull
        # Since it's a convex polygon containing the origin?
        # If the hexagons can overlap, the origin is inside.
        # The minimum distance to be disjoint is the minimum distance from origin to the boundary.
        # This is equivalent to the minimum support function value in any direction?
        # No, min_{u} h_{H1-H2}(u) is not it.
        # The distance from origin to a convex polygon P is min_{p in boundary(P)} ||p||.
        
        # We can find the convex hull and check distances to edges.
        # Or simpler: The boundary consists of segments.
        # Let's just find the point in the convex hull closest to origin.
        # We can use scipy.spatial.ConvexHull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(diff_points)
            # Check each edge
            min_d = float('inf')
            for simplex in hull.simplices:
                # In 2D, simplex has 2 points
                p1 = diff_points[simplex[0]]
                p2 = diff_points[simplex[1]]
                # Distance from origin to segment p1-p2
                # Vector p2 - p1
                v = p2 - p1
                w = -p1
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1)
                elif c1 >= c2:
                    d = np.linalg.norm(p2)
                else:
                    t = c1 / c2
                    proj = p1 + t * v
                    d = np.linalg.norm(proj)
                if d < min_d:
                    min_d = d
            return min_d
        except Exception:
            # Fallback: brute force min norm of diff points (approximation)
            return np.min(np.linalg.norm(diff_points, axis=1))

    # Calculate D_ij for all pairs
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            delta = all_angles[i] - all_angles[j]
            # Symmetry: delta and -delta give same distance
            D[i, j] = min_dist_between_hexagons(delta)
            D[j, i] = D[i, j]
            
    def objective(params):
        R1, R2, theta = params
        
        # Calculate centers
        centers = np.zeros((N, 2))
        # 0: Center
        centers[0] = [0.0, 0.0]
        
        # 1-5: Inner
        for k, ang in enumerate(inner_angles):
            a = ang + theta
            centers[1+k] = [R1 * np.cos(a), R1 * np.sin(a)]
            
        # 6-11: Outer
        for k, ang in enumerate(outer_angles):
            a = ang + theta
            centers[6+k] = [R2 * np.cos(a), R2 * np.sin(a)]
            
        # Calculate L_required
        # L = max over all inner hexagons and all outer normals
        # Support of inner hex i in direction n_k:
        # C_i . n_k + h_hex(n_k - beta_i)
        # h_hex(u) for unit hex rotated by beta:
        # max_m (v_m . u_rotated) where v_m are vertices of hex at 0 rot?
        # If hex is rotated by beta, its vertices are R_beta v_m.
        # Support = max_m ( (R_beta v_m) . n_k ) = max_m ( v_m . (R_beta^T n_k) )
        
        L_req = 0.0
        for i in range(N):
            beta = all_angles[i] + theta
            # Rotate normal by -beta to get normal in hex's local frame
            c, s = np.cos(-beta), np.sin(-beta)
            rot_mat = np.array([[c, -s], [s, c]])
            local_normals = outer_normals @ rot_mat.T
            
            # Support function of unit hex (vertices at 0, 60..)
            # h(u) = max_k (v_k . u)
            dots = local_normals @ hex_vertices.T
            max_supports = np.max(dots, axis=1)
            
            # Projection of center
            proj = centers[i] @ outer_normals.T
            
            # Total support
            supports = proj + max_supports
            L_req = max(L_req, np.max(supports))
            
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < D[i, j]:
                    viol = D[i, j] - dist
                    penalty += viol * viol
                    
        return L_req + 100.0 * penalty

    # Initial guess
    # R1 should be at least sqrt(3) ~ 1.73 to clear center
    # R2 should be larger
    x0 = np.array([2.0, 3.5, 0.0])
    
    best_L = float('inf')
    best_params = None
    
    # Try multiple starts
    for seed in range(5):
        np.random.seed(seed)
        x0_trial = x0 + np.random.uniform(-0.5, 0.5, 3)
        
        try:
            result = minimize(objective, x0_trial, method='L-BFGS-B', bounds=[(0.5, 10), (0.5, 10), (-np.pi, np.pi)], options={'maxiter': 2000})
            
            # Verify
            R1, R2, theta = result.x
            # Re-calculate to ensure validity
            # We need to check if penalty is 0
            # We can just check the objective value, but better to re-evaluate constraints
            
            # Reconstruct centers
            centers = np.zeros((N, 2))
            centers[0] = [0.0, 0.0]
            for k, ang in enumerate(inner_angles):
                a = ang + theta
                centers[1+k] = [R1 * np.cos(a), R1 * np.sin(a)]
            for k, ang in enumerate(outer_angles):
                a = ang + theta
                centers[6+k] = [R2 * np.cos(a), R2 * np.sin(a)]
                
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < D[i, j] - 1e-5:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                # Calculate final L
                L_final = 0.0
                for i in range(N):
                    beta = all_angles[i] + theta
                    c, s = np.cos(-beta), np.sin(-beta)
                    rot_mat = np.array([[c, -s], [s, c]])
                    local_normals = outer_normals @ rot_mat.T
                    dots = local_normals @ hex_vertices.T
                    max_supports = np.max(dots, axis=1)
                    proj = centers[i] @ outer_normals.T
                    supports = proj + max_supports
                    L_final = max(L_final, np.max(supports))
                    
                if L_final < best_L:
                    best_L = L_final
                    best_params = result.x
                    
        except Exception:
            continue
            
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    R1, R2, theta = best_params
    theta_deg = np.rad2deg(theta)
    
    inner = []
    # 0: Center
    inner.append([0.0, 0.0, theta_deg])
    # 1-5: Inner
    for ang in inner_angles:
        a = ang + theta
        x = R1 * np.cos(a)
        y = R1 * np.sin(a)
        inner.append([x, y, theta_deg + np.rad2deg(ang)])
    # 6-11: Outer
    for ang in outer_angles:
        a = ang + theta
        x = R2 * np.cos(a)
        y = R2 * np.sin(a)
        inner.append([x, y, theta_deg + np.rad2deg(ang)])
        
    return inner, (0.0, 0.0), best_L, 0.0
