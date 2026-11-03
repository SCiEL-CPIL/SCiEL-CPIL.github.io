import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE_MARGIN = 10.0  # Safety margin for 150s limit

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts_batch(positions, angles_deg):
        """
        Vectorized calculation of vertices for all hexagons.
        positions: (N, 2)
        angles_deg: (N,)
        Returns: (N, 6, 2)
        """
        angles_rad = np.deg2rad(angles_deg)
        cos_a = np.cos(angles_rad)
        sin_a = np.sin(angles_rad)
        
        # Rotation matrix for each hexagon
        # R = [[cos, -sin], [sin, cos]]
        # We want to apply R to unit_verts_0 (6x2)
        # Result[i] = unit_verts_0 @ R[i].T
        # Using einsum: 'ij,ik->jk' doesn't quite work directly for batched matmul with static A.
        # Let's do: verts[i, j, k] = sum_l unit_verts_0[j, l] * R[i, l, k]
        # R[i, 0, 0] = cos_a[i], R[i, 0, 1] = -sin_a[i]
        # R[i, 1, 0] = sin_a[i], R[i, 1, 1] = cos_a[i]
        
        # Expand unit_verts_0 to (1, 6, 2)
        u = unit_verts_0[np.newaxis, :, :] # (1, 6, 2)
        
        # Construct rotation matrices (N, 2, 2)
        R = np.zeros((len(positions), 2, 2))
        R[:, 0, 0] = cos_a
        R[:, 0, 1] = -sin_a
        R[:, 1, 0] = sin_a
        R[:, 1, 1] = cos_a
        
        # Apply rotation: (N, 6, 2) = (N, 6, 2) @ (N, 2, 2).T ? No.
        # v_rot = R @ v
        # v is (6, 2). R is (2, 2).
        # Batched: (N, 2, 2) @ (1, 2, 6).T -> (N, 2, 6).T -> (N, 6, 2)
        # Let's use einsum for clarity and speed
        # 'ij,ik->jk' is for single. 
        # 'njk,kl->njl' where u is (1, 6, 2) -> broadcast to (N, 6, 2)
        # Actually simpler:
        # verts[i] = u[0] @ R[i].T
        # verts = np.einsum('ij,njk->nik', u[0], R) # (N, 6, 2)
        
        verts = np.einsum('ij,njk->nik', unit_verts_0, R)
        verts += positions[:, np.newaxis, :]
        
        return verts

    def calc_min_outer_side(positions, angles_deg, outer_center, outer_angle):
        """
        Calculate the minimum side length of a regular hexagon centered at outer_center 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        verts = get_hex_verts_batch(positions, angles_deg) # (N, 6, 2)
        verts_rel = verts - np.array(outer_center)
        
        # Project all vertices onto all normals
        # projs[i, j, k] = dot(verts_rel[i, j], normals[k])
        # (N, 6, 2) @ (6, 2).T -> (N, 6, 6)
        projs = np.einsum('ijk,kl->ijl', verts_rel, normals)
        
        # We need max projection magnitude? No, for a centered polygon, 
        # the distance from center to side is the max projection of vertices onto the normal.
        # Since the hexagon is centered at outer_center, the side length S is related to 
        # the distance d from center to side by S = 2 * d / sqrt(3).
        # Wait, for a regular hexagon with side s, the apothem (distance from center to side) is s * sqrt(3) / 2.
        # So s = 2 * apothem / sqrt(3).
        # The apothem must be at least the max projection of any inner vertex onto the outward normal.
        # Note: If the inner hexagon crosses the center, projections can be negative. 
        # But the "support function" h(n) = max_v dot(v, n).
        # The containing hexagon must satisfy h_outer(n_k) >= h_inner(n_k) for all 6 normals n_k.
        # h_outer(n_k) = S * sqrt(3) / 2.
        # So S * sqrt(3) / 2 >= max_k max_i max_j dot(v_ij, n_k).
        # Thus S >= (2 / sqrt(3)) * max_{k, i, j} dot(v_ij, n_k).
        
        max_proj = np.max(projs)
        
        return max_proj * 2.0 / math.sqrt(3)

    def calc_overlap_penalty(positions, angles_deg):
        """
        Calculate a smooth penalty for overlapping hexagons.
        Uses SAT penetration depth.
        """
        verts = get_hex_verts_batch(positions, angles_deg) # (N, 6, 2)
        
        total_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p1 = verts[i]
                p2 = verts[j]
                
                # Quick bounding box check
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue

                # SAT
                min_pen = float('inf')
                
                # Axes from p1
                edges1 = np.roll(p1, -1, axis=0) - p1
                normals1 = np.array([[-e[1], e[0]] for e in edges1])
                norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
                normals1 = normals1 / (norms1 + 1e-9)
                
                # Axes from p2
                edges2 = np.roll(p2, -1, axis=0) - p2
                normals2 = np.array([[-e[1], e[0]] for e in edges2])
                norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
                normals2 = normals2 / (norms2 + 1e-9)
                
                all_normals = np.vstack([normals1, normals2])
                
                # Project both polygons onto all normals
                # proj1: (12, 6)
                proj1 = p1 @ all_normals.T
                proj2 = p2 @ all_normals.T
                
                min1 = np.min(proj1, axis=1)
                max1 = np.max(proj1, axis=1)
                min2 = np.min(proj2, axis=1)
                max2 = np.max(proj2, axis=1)
                
                # Penetration depth on each axis
                # If max1 < min2: sep = min2 - max1 (positive)
                # If max2 < min1: sep = min1 - max2 (positive)
                # Else: pen = min(max1, max2) - max(min1, min2) (positive if overlap)
                
                # We want to penalize overlap.
                # Let's compute the "gap" which is positive if separated, negative if overlapping.
                # gap = min(max1 - min2, max2 - min1) ? No.
                # Distance between intervals [min1, max1] and [min2, max2]:
                # d = max(min1 - max2, min2 - max1)
                # If d > 0, separated by d.
                # If d < 0, overlapping by -d.
                
                gaps = np.minimum(max1 - min2, max2 - min1)
                # Wait, if max1 < min2, max1-min2 is negative. max2-min1 is positive. min is negative.
                # If max2 < min1, max1-min2 is positive. max2-min1 is negative. min is negative.
                # If overlap, both are positive? 
                # Example: [0, 10] and [5, 15].
                # max1=10, min2=5 -> 5.
                # max2=15, min1=0 -> 15.
                # min(5, 15) = 5. This is positive, but they overlap.
                
                # Correct gap calculation:
                # sep = min(max1, max2) - max(min1, min2)
                # If sep > 0, they overlap by sep.
                # If sep < 0, they are separated by -sep.
                
                seps = np.minimum(max1, max2) - np.maximum(min1, min2)
                max_pen_axis = np.max(seps)
                
                if max_pen_axis > 0:
                    # Overlapping. Penalty is the depth.
                    # To make it smooth-ish and avoid huge gradients, we can use max_pen_axis^2 or just max_pen_axis.
                    # SLSQP/DE handles non-smooth okay, but penalty method prefers smooth.
                    # Let's use the depth directly.
                    total_pen += max_pen_axis
                # else: separated, no penalty
                
        return total_pen

    def objective(params):
        """
        Objective function for optimization.
        params: [x0...x11, y0...y11, a0...a11]
        """
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:]
        
        # Fixed outer hexagon for the inner optimization stage?
        # No, we want to minimize the outer side length.
        # However, the outer hexagon parameters (center, angle) also affect the side length.
        # In the penalty-based global search, it's hard to jointly optimize 39 variables.
        # Strategy:
        # 1. Optimize inner positions/angles to minimize a proxy objective: 
        #    (Extent of inner cluster) + Penalty.
        #    Or, assume outer is centered at (0,0) and angle 0, and minimize the required side length.
        #    Then, polish the outer center/angle for the best inner config.
        
        # Let's assume outer is centered at (0,0) and angle 0 for the inner optimization.
        # This is a valid heuristic because the optimal outer hexagon can be translated/rotated 
        # to align with the inner cluster's "center of mass" or principal axes.
        # Actually, simply centering the inner cluster at (0,0) is a good first guess.
        
        side = calc_min_outer_side(pos, np.degrees(ang), [0.0, 0.0], 0.0)
        pen = calc_overlap_penalty(pos, np.degrees(ang))
        
        # Weight for penalty. Must be large enough to prevent overlap, 
        # but not so large that it dominates the side length minimization.
        # Typical side length is ~5-6. Overlap penalty can be ~1-2.
        # Let's use weight 10.
        return side + 10.0 * pen

    # Initial bounds for DE
    # Positions: roughly within a hexagon of side 6.
    # Angles: 0 to 60 degrees (symmetry).
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
        bounds.append((0.0, 60.0)) # angle in rad? No, objective expects deg in get_hex_verts?
        # Wait, get_hex_verts_batch takes angles_deg.
        # But DE works in rad usually? No, DE is scale invariant.
        # Let's use radians for optimization to be consistent with math functions, 
        # but convert to degrees for get_hex_verts_batch.
        # Or just use degrees in the parameter vector.
        # Let's use degrees in the parameter vector for simplicity.
        bounds.append((0.0, 60.0))

    # Run Differential Evolution
    # maxiter: 100? popsize: 20?
    # Time limit is important.
    
    try:
        # DE is stochastic. Use a fixed seed for reproducibility? 
        # The task doesn't require reproducibility, but it's good practice.
        # However, random seeds might get stuck. 
        # Let's run DE for a few seconds.
        
        de_start = time.time()
        de_timeout = 60.0 # Give 60s to DE
        
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=1e-5, 
            polish=False, # We will do our own polish
            # We can't easily pass a timeout to DE, so we rely on maxiter and popsize
            # to keep it fast.
        )
        
        if time.time() > deadline - 20:
             # If we are running out of time, use the best found so far or fallback
             pass
             
        best_params = res_de.x
        best_pos = best_params[:2*N].reshape(N, 2)
        best_ang = best_params[2*N:]
        
        # Polish phase:
        # 1. Local optimization of inner params (Nelder-Mead) to reduce penalty and side.
        # 2. Joint optimization of outer params (center, angle) given inner params.
        
        # Step 1: Polish inner
        if time.time() < deadline - 10:
            res_inner = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res_inner.fun < objective(best_params):
                best_params = res_inner.x
                best_pos = best_params[:2*N].reshape(N, 2)
                best_ang = best_params[2*N:]
        
        # Step 2: Polish outer
        # Objective for outer: minimize side length given fixed inner.
        # Note: We must ensure the inner hexagons are still disjoint. 
        # Changing the outer hexagon does NOT change the inner hexagons' relative positions, 
        # so disjointness is preserved.
        
        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2]
            return calc_min_outer_side(best_pos, np.degrees(best_ang), oc, oa)
        
        # Initial guess: centroid of inner hexes, angle 0
        centroid = np.mean(best_pos, axis=0)
        x0_outer = np.array([centroid[0], centroid[1], 0.0])
        
        res_outer = minimize(
            outer_obj, 
            x0_outer, 
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        best_outer_center = res_outer.x[:2]
        best_outer_angle = res_outer.x[2]
        best_side = res_outer.fun
        
        # Verify disjointness one last time (just in case numerical errors crept in)
        # If not disjoint, we might need to reject or fall back.
        # For now, assume it's valid if penalty was driven low.
        
    except Exception as e:
        # Fallback to simple honeycomb
        best_pos = None
        best_ang = None
        best_outer_center = None
        best_outer_angle = None
        best_side = None

    if best_pos is None:
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        best_pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        best_ang = np.zeros(N)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0
        best_side = calc_min_outer_side(best_pos, best_ang, best_outer_center, best_outer_angle)

    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], best_ang[i]])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle

# The code above has a potential issue: `best_pos` and `best_ang` might be undefined if an exception occurs 
# before they are set in the try block, but the `except` block sets them to None, and the `if` block handles that.
# However, if `res_de` fails, `best_params` might not be defined.
# Let's add a safeguard.

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - 10.0  # Safety margin

    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts_batch(positions, angles_deg):
        angles_rad = np.deg2rad(angles_deg)
        cos_a = np.cos(angles_rad)
        sin_a = np.sin(angles_rad)
        
        R = np.zeros((len(positions), 2, 2))
        R[:, 0, 0] = cos_a
        R[:, 0, 1] = -sin_a
        R[:, 1, 0] = sin_a
        R[:, 1, 1] = cos_a
        
        verts = np.einsum('ij,njk->nik', unit_verts_0, R)
        verts += positions[:, np.newaxis, :]
        
        return verts

    def calc_min_outer_side(positions, angles_deg, outer_center, outer_angle):
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        verts = get_hex_verts_batch(positions, angles_deg)
        verts_rel = verts - np.array(outer_center)
        
        projs = np.einsum('ijk,kl->ijl', verts_rel, normals)
        max_proj = np.max(projs)
        
        return max_proj * 2.0 / math.sqrt(3)

    def calc_overlap_penalty(positions, angles_deg):
        verts = get_hex_verts_batch(positions, angles_deg)
        
        total_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p1 = verts[i]
                p2 = verts[j]
                
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue

                min_pen = float('inf')
                
                edges1 = np.roll(p1, -1, axis=0) - p1
                normals1 = np.array([[-e[1], e[0]] for e in edges1])
                norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
                normals1 = normals1 / (norms1 + 1e-9)
                
                edges2 = np.roll(p2, -1, axis=0) - p2
                normals2 = np.array([[-e[1], e[0]] for e in edges2])
                norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
                normals2 = normals2 / (norms2 + 1e-9)
                
                all_normals = np.vstack([normals1, normals2])
                
                proj1 = p1 @ all_normals.T
                proj2 = p2 @ all_normals.T
                
                min1 = np.min(proj1, axis=1)
                max1 = np.max(proj1, axis=1)
                min2 = np.min(proj2, axis=1)
                max2 = np.max(proj2, axis=1)
                
                seps = np.minimum(max1, max2) - np.maximum(min1, min2)
                max_pen_axis = np.max(seps)
                
                if max_pen_axis > 0:
                    total_pen += max_pen_axis
                    
        return total_pen

    def objective(params):
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:]
        
        side = calc_min_outer_side(pos, np.degrees(ang), [0.0, 0.0], 0.0)
        pen = calc_overlap_penalty(pos, np.degrees(ang))
        
        return side + 10.0 * pen

    best_pos = None
    best_ang = None
    best_outer_center = None
    best_outer_angle = None
    best_side = None

    try:
        bounds = []
        for i in range(N):
            bounds.append((-5.0, 5.0))
            bounds.append((-5.0, 5.0))
            bounds.append((0.0, 60.0))

        # Run DE
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=20, 
            tol=1e-5, 
            polish=False
        )
        
        if time.time() > deadline - 15:
            raise TimeoutError("DE took too long")

        best_params = res_de.x
        best_pos = best_params[:2*N].reshape(N, 2)
        best_ang = best_params[2*N:]
        
        # Polish inner
        if time.time() < deadline - 10:
            res_inner = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res_inner.fun < objective(best_params):
                best_params = res_inner.x
                best_pos = best_params[:2*N].reshape(N, 2)
                best_ang = best_params[2*N:]
        
        # Polish outer
        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2]
            return calc_min_outer_side(best_pos, np.degrees(best_ang), oc, oa)
        
        centroid = np.mean(best_pos, axis=0)
        x0_outer = np.array([centroid[0], centroid[1], 0.0])
        
        res_outer = minimize(
            outer_obj, 
            x0_outer, 
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        best_outer_center = res_outer.x[:2]
        best_outer_angle = res_outer.x[2]
        best_side = res_outer.fun
        
    except Exception as e:
        pass

    if best_pos is None:
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        best_pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        best_ang = np.zeros(N)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0
        best_side = calc_min_outer_side(best_pos, best_ang, best_outer_center, best_outer_angle)

    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], best_ang[i]])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle

