import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute unit hexagon vertices for angle 0
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, angle_deg):
        """Get vertices of a unit hexagon at center (cx, cy) with rotation angle_deg."""
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        rot = np.array([[ca, -sa], [sa, ca]])
        return unit_verts_0 @ rot.T + np.array([cx, cy])

    def minkowski_diff_dist(p1, p2):
        """Compute minimum distance between two convex hexagons using Minkowski difference.
        Returns negative if overlapping, positive if separated."""
        # Minkowski difference: check if origin is inside P1 - P2
        # Equivalent to checking separating axes
        min_dist = float('inf')
        for poly, other in [(p1, p2), (p2, p1)]:
            for i in range(6):
                v1 = poly[i]
                v2 = poly[(i+1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                proj1 = p1 @ normal
                proj2 = other @ normal
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                gap = max(min1 - max2, min2 - max1)
                if gap < 0:
                    min_dist = min(min_dist, gap)
                else:
                    return gap
        return min_dist

    def calc_outer_side(centers, angles, outer_angle=0.0):
        """Calculate minimum side length of regular hexagon containing all inner hexagons."""
        theta = math.radians(outer_angle)
        max_proj = 0.0
        for k in range(6):
            na = theta + math.pi/6 + k * math.pi/3
            nx, ny = math.cos(na), math.sin(na)
            for i in range(N):
                cx, cy = centers[i]
                a = math.radians(angles[i])
                best = 0.0
                for k2 in range(6):
                    va = a + k2 * math.pi/3
                    proj = math.cos(va)*nx + math.sin(va)*ny
                    if proj > best:
                        best = proj
                proj_val = cx*nx + cy*ny + best
                if proj_val > max_proj:
                    max_proj = proj_val
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid(centers, angles, tol=1e-6):
        """Check if all hexagons are pairwise disjoint."""
        polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if minkowski_diff_dist(polys[i], polys[j]) < -tol:
                    return False
        return True

    def min_separation(centers, angles, tol=1e-6):
        """Find the minimum separation among all pairs. Negative means overlap."""
        polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        min_sep = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = minkowski_diff_dist(polys[i], polys[j])
                if d < min_sep:
                    min_sep = d
        return min_sep

    # Phase 1: Generate initial configuration from honeycomb lattice
    sqrt3 = math.sqrt(3)
    a1 = np.array([1.5, sqrt3/2])
    a2 = np.array([0.0, sqrt3])

    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * a1 + j * a2
            lattice_points.append(p)
    
    lattice_points.sort(key=lambda p: np.dot(p, p))
    init_points = np.array(lattice_points[:12])
    centroid = np.mean(init_points, axis=0)
    init_points -= centroid

    # Try different global rotations
    best_config = None
    best_score = float('inf')
    for rot_deg in [0, 30, 60, 90, 120, 150]:
        rot_rad = math.radians(rot_deg)
        rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                           [math.sin(rot_rad), math.cos(rot_rad)]])
        rotated = init_points @ rot_mat.T
        angles = np.full(N, rot_deg)
        if is_valid(rotated, angles, tol=1e-5):
            score = calc_outer_side(rotated, angles, 0.0)
            if score < best_score:
                best_score = score
                best_config = (rotated.copy(), angles.copy())

    if best_config is None:
        for scale in [1.01, 1.02, 1.05, 1.1]:
            scaled = init_points * scale
            for rot_deg in [0, 30]:
                rot_rad = math.radians(rot_deg)
                rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                                   [math.sin(rot_rad), math.cos(rot_rad)]])
                rotated = scaled @ rot_mat.T
                angles = np.full(N, rot_deg)
                if is_valid(rotated, angles, tol=1e-5):
                    score = calc_outer_side(rotated, angles, 0.0)
                    if score < best_score:
                        best_score = score
                        best_config = (rotated.copy(), angles.copy())
                    break
            if best_config is not None:
                break

    if best_config is None:
        init_points = np.array([[i*4, j*4] for i in range(-2,3) for j in range(-2,3)][:12])
        init_points -= np.mean(init_points, axis=0)
        best_config = (init_points, np.zeros(N))
        best_score = calc_outer_side(init_points, np.zeros(N), 0.0)

    init_centers, init_angles = best_config
    best_val = best_score
    best_params = np.concatenate([init_centers.flatten(), init_angles, [0.0]])

    # Phase 2: Homotopy continuation - track from large container to minimal
    # We'll use a penalty-based approach with gradually increasing penalty weight
    # and shrinking the target container size
    
    def homotopy_objective(params, penalty_weight, target_side):
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        outer_angle = params[3*N]
        
        # Compute actual outer side
        actual_side = calc_outer_side(centers, angles, outer_angle)
        
        # Penalty for exceeding target side
        if actual_side > target_side:
            side_penalty = penalty_weight * (actual_side - target_side)**2
        else:
            side_penalty = 0.0
        
        # Penalty for overlaps
        polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = minkowski_diff_dist(polys[i], polys[j])
                if d < 0:
                    overlap_penalty += penalty_weight * d * d
        
        return actual_side + side_penalty + overlap_penalty

    def homotopy_constraints(params, min_sep=0.0):
        """Generate constraints ensuring minimum separation between hexagons."""
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                ci = centers[i]
                cj = centers[j]
                ai = angles[i]
                aj = angles[j]
                pi = get_hex_verts(ci[0], ci[1], ai)
                pj = get_hex_verts(cj[0], cj[1], aj)
                d = minkowski_diff_dist(pi, pj)
                if d < min_sep:
                    constraints.append(d - min_sep)
        return constraints

    # Start from a known valid configuration and gradually tighten
    # Use multiple homotopy steps with decreasing target side
    current_params = best_params.copy()
    current_side = best_val
    
    # Determine range of target sides to try
    # Start from current best and go down
    n_homotopy_steps = 8
    target_sides = np.linspace(current_side, current_side * 0.85, n_homotopy_steps)
    
    for step, target_side in enumerate(target_sides):
        if time.time() > deadline - 30:
            break
            
        penalty_weight = 10.0 + 50.0 * step  # Increase penalty weight
        
        def obj(params):
            return homotopy_objective(params, penalty_weight, target_side)
        
        # Try to optimize with SLSQP
        try:
            res = minimize(obj, current_params, method='SLSQP',
                          options={'maxiter': 100, 'ftol': 1e-10, 'disp': False})
            
            if res.fun < obj(current_params):
                new_params = res.x.copy()
                centers = new_params[:2*N].reshape(N, 2)
                angles = new_params[2*N:3*N]
                outer_angle = new_params[3*N]
                
                # Check validity
                if is_valid(centers, angles, tol=1e-5):
                    new_side = calc_outer_side(centers, angles, outer_angle)
                    if new_side < best_val:
                        best_val = new_side
                        best_params = new_params.copy()
                        current_params = new_params.copy()
                    else:
                        # Keep current_params but update best if valid
                        current_params = new_params.copy()
        except Exception:
            pass

    # Phase 3: Final polish with Nelder-Mead
    if time.time() < deadline - 10:
        def final_obj(params):
            centers = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            outer_angle = params[3*N]
            
            actual_side = calc_outer_side(centers, angles, outer_angle)
            
            # Strong penalty for overlaps
            polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = minkowski_diff_dist(polys[i], polys[j])
                    if d < 0:
                        overlap_penalty += 1000.0 * d * d
            
            return actual_side + overlap_penalty

        try:
            res2 = minimize(final_obj, best_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            centers = res2.x[:2*N].reshape(N, 2)
            angles = res2.x[2*N:3*N]
            outer_angle = res2.x[3*N]
            
            if is_valid(centers, angles, tol=1e-5):
                val = calc_outer_side(centers, angles, outer_angle)
                if val < best_val:
                    best_val = val
                    best_params = res2.x.copy()
        except Exception:
            pass

    # Final: extract result
    final_centers = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:3*N]
    final_outer_angle = best_params[3*N]
    final_side = calc_outer_side(final_centers, final_angles, final_outer_angle)

    inner = [[final_centers[i][0], final_centers[i][1], final_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
