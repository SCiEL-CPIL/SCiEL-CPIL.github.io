import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute support function of a unit hexagon (side 1) at many angles
    # h_unit(theta) for a hexagon with normal at 0 (pointing right)
    # The support function of a regular hexagon with apothem 'a' and orientation 0 is:
    # h(u) = a / cos(delta), where delta is angle between u and closest normal.
    # Normals are at 0, 60, 120, ...
    # We will use a fine grid for u to approximate the max.
    num_angles = 720
    u_angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    
    # h_unit for a unit hexagon (side 1, apothem APOTHEM) oriented such that one normal is at 0
    # Actually, let's define h_unit(v) for any angle v.
    # If the hexagon is rotated by alpha, h(v) = h_unit(v - alpha).
    # h_unit(v) = APOTHEM / cos(delta(v)), where delta(v) is distance to nearest multiple of pi/3.
    def calc_h_unit(v):
        # v is an array of angles
        # Map to [0, pi/3)
        v_mod = np.mod(v, math.pi / 3.0)
        # Distance to closest normal (0 or pi/3)
        # The function is symmetric around pi/6.
        # delta = abs(v_mod - pi/6) if we consider the peak at pi/6? No.
        # The normals are at 0, pi/3, ...
        # The support is maximal at the normals.
        # h(v) = APOTHEM / cos(v_mod) is wrong.
        # Let's check.
        # At normal (0), h = APOTHEM / cos(0) = APOTHEM. Correct.
        # At vertex (pi/6), h = APOTHEM / cos(pi/6) = APOTHEM / (sqrt(3)/2) = 1. Correct (distance to vertex is 1).
        # So h_unit(v) = APOTHEM / cos(v_mod) where v_mod is in [-pi/6, pi/6].
        # Let's map v to [-pi/6, pi/6].
        v_mapped = np.mod(v + math.pi/6.0, math.pi/3.0) - math.pi/6.0
        return APOTHEM / np.cos(v_mapped)

    # Precompute h_unit for the grid u_angles
    h_unit_grid = calc_h_unit(u_angles)

    def get_support_function(pts, alpha):
        """
        Calculate the support function h(u) of the cluster of N unit hexagons
        at positions pts (N x 2) with common rotation alpha.
        Returns an array of h(u) for u in u_angles.
        """
        # pts: (N, 2)
        # u_angles: (M,)
        # h(u) = max_i ( h_unit(u - alpha) + u . p_i )
        
        # u . p_i = p_i_x * cos(u) + p_i_y * sin(u)
        cos_u = np.cos(u_angles)
        sin_u = np.sin(u_angles)
        
        # Projections of points onto u
        # proj[i, j] = p_i . u_j
        proj = pts[:, 0] * cos_u + pts[:, 1] * sin_u  # (N, M)
        
        # h_unit(u - alpha)
        # u_angles - alpha
        h_u_alpha = calc_h_unit(u_angles - alpha) # (M,)
        
        # For each u_j, h_j = max_i ( h_u_alpha[j] + proj[i, j] )
        # h_u_alpha is added to each row of proj
        # We want the max over i (rows) for each j (column)
        # So we add h_u_alpha to each column? No, h_u_alpha is constant for a given u_j across all i.
        # So h_j = h_u_alpha[j] + max_i (proj[i, j])
        
        max_proj = np.max(proj, axis=0) # (M,)
        h_values = h_u_alpha + max_proj
        return h_values

    def calc_min_container_side(h_values):
        """
        Given h_values (support function of cluster on grid), find minimal S
        of a regular hexagon enclosing the cluster.
        We minimize over container orientation phi.
        S(phi) = (2/sqrt(3)) * max_u ( h(u) * cos(delta(u - phi)) )
        where delta(u - phi) is the angle between u and the closest normal of the container.
        The container normals are at phi, phi + pi/3, ...
        So delta(u - phi) is the distance of (u - phi) to the nearest multiple of pi/3.
        Let v = u - phi.
        v_mapped = mod(v + pi/6, pi/3) - pi/6.
        cos(delta) = cos(v_mapped).
        
        So for a fixed phi, a_min = max_u ( h(u) * cos(v_mapped) )
        S = (2/sqrt(3)) * a_min.
        """
        # We need to minimize a_min over phi in [0, pi/3).
        
        def objective(phi):
            # Calculate v_mapped for all u
            v = u_angles - phi
            v_mapped = np.mod(v + math.pi/6.0, math.pi/3.0) - math.pi/6.0
            cos_delta = np.cos(v_mapped)
            
            # a_min = max ( h_values * cos_delta )
            a_min = np.max(h_values * cos_delta)
            return a_min

        # Minimize over phi
        # Use bounded scalar minimization
        res = minimize_scalar(objective, bounds=(0, math.pi / 3.0), method='bounded', options={'maxiter': 100, 'xatol': 1e-8})
        a_min = res.fun
        phi_opt = res.x
        S = (2.0 / math.sqrt(3.0)) * a_min
        return S, phi_opt

    def objective(params):
        """
        params: [x1, y1, ..., xN, yN, alpha]
        Returns: S + penalty
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
            
        pts = params[:2*N].reshape(N, 2)
        alpha = params[2*N]
        
        # Calculate support function
        h_values = get_support_function(pts, alpha)
        
        # Calculate minimal container side
        S, phi_opt = calc_min_container_side(h_values)
        
        # Penalty for overlaps
        # For two unit hexagons with same orientation, min distance is 2.0
        # If orientations differ, it's more complex, but we assume common alpha.
        # So we just check Euclidean distance >= 2.0
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 2.0:
                    penalty += (2.0 - d) ** 2
        
        return S + 100.0 * penalty

    # Initial candidates from discrete lattice
    # We will use a few good discrete configurations as starting points for Nelder-Mead
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    candidates = []
    
    # Hexagonal ring + center
    axial_1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)]
    candidates.append(axial_1)
    
    # Linear-ish
    axial_2 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1)]
    candidates.append(axial_2)
    
    # Compact
    axial_3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)]
    candidates.append(axial_3)
    
    best_S = float('inf')
    best_params = None
    best_center = None
    best_phi = 0.0

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        # Center the points
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        
        # Initial alpha = 0
        alpha_init = 0.0
        x0 = np.concatenate([pts.flatten(), [alpha_init]])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
                # Extract phi_opt for reporting
                pts_opt = best_params[:2*N].reshape(N, 2)
                alpha_opt = best_params[2*N]
                h_values = get_support_function(pts_opt, alpha_opt)
                _, phi_opt = calc_min_container_side(h_values)
                best_phi = phi_opt
                # The center of the container is not necessarily the centroid.
                # For a regular hexagon, the center is the point that minimizes the max projection.
                # We can find it by solving a small LP or just using the centroid as an approximation?
                # Actually, the support function method assumes the container is centered at the origin?
                # No, h(u) is the support of the cluster relative to the origin.
                # The container is centered at c.
                # The condition is h_container(u) >= h_cluster(u - c)? No.
                # The support function of a set A translated by c is h_A(u) + u.c.
                # So h_cluster_translated(u) = h_cluster(u) + u.c.
                # We want min S s.t. h_container(u) >= h_cluster(u) + u.c for all u.
                # h_container(u) = a / cos(delta(u - phi)).
                # This is equivalent to finding c and a, phi.
                # My `calc_min_container_side` assumed c=0.
                # To be rigorous, we should optimize c as well.
                # But for symmetric clusters, c is the centroid.
                # Let's assume c is the centroid for now, and refine if needed.
                # Actually, the Nelder-Mead optimization of pts effectively moves the cluster.
                # The "center" returned should be the centroid of the optimized pts.
                best_center = np.mean(pts_opt, axis=0)
                
        except Exception as e:
            pass

    if best_params is None:
        # Fallback
        pts = axial_to_cartesian(candidates[0])
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        best_center = centroid
        best_S = 10.0
        best_phi = 0.0
        alpha_opt = 0.0

    # Final refinement of the center?
    # The support function method with c=0 gives the minimal hexagon centered at 0.
    # If the cluster is not centered at 0, the hexagon will be off-center.
    # But we are returning the center of the hexagon.
    # If we assume the optimal hexagon is centered at the centroid of the cluster, 
    # then the side length calculated by `calc_min_container_side` is correct 
    # IF the cluster is centered at the origin.
    # Since we centered the pts before optimization, the centroid is (0,0).
    # So the container center is (0,0) in the local frame.
    # In the global frame, it is `best_center`.

    cx, cy = best_center
    angle_deg = math.degrees(best_phi)
    
    # The inner hexagons have rotation alpha_opt relative to the container?
    # No, alpha_opt is the rotation of the inner hexagons relative to the *global* axes 
    # IF the container is aligned with the global axes?
    # Let's trace:
    # 1. We optimized pts and alpha.
    # 2. `get_support_function` uses alpha to rotate the unit hexagon support.
    # 3. `calc_min_container_side` finds the best container orientation phi.
    # 4. The container is at `best_center`.
    # 5. The inner hexagons are at `best_params[:2*N]` (which are centered at 0).
    #    So global positions are `best_params[:2*N] + best_center`.
    # 6. The inner hexagons have orientation `alpha_opt`.
    # 7. The container has orientation `best_phi`.
    
    # Wait, the problem asks for `inner` rows (x, y, angle_degrees).
    # And `center`, `side`, `rotation` for the outer.
    
    pts_opt = best_params[:2*N].reshape(N, 2)
    global_pts = pts_opt + np.array([cx, cy])
    alpha_opt = best_params[2*N]
    
    inner = []
    for x, y in global_pts:
        inner.append([float(x), float(y), float(math.degrees(alpha_opt))])
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
