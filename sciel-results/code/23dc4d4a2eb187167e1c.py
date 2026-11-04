import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute unit hexagon vertices for angle 0
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    def get_hex_verts(cx, cy, angle_deg):
        """Get vertices of a unit hexagon at center (cx, cy) with rotation angle_deg."""
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        rot = np.array([[ca, -sa], [sa, ca]])
        return unit_verts_0 @ rot.T + np.array([cx, cy])

    def sat_gap(poly1, poly2):
        """
        Calculate the minimum separation distance (gap) between two convex hexagons.
        Returns negative value if overlapping (penetration depth), positive if separated.
        Uses Separating Axis Theorem.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap # Found separating axis
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def calc_outer_side(params):
        """
        Calculate the side length of the minimal regular hexagon containing all inner hexagons.
        params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_angle]
        """
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        outer_angle_deg = params[3*N]
        
        # Normals for the outer hexagon sides
        theta_rad = math.radians(outer_angle_deg)
        max_proj = 0.0
        
        for k in range(6):
            na = theta_rad + math.pi/6 + k * math.pi/3
            nx, ny = math.cos(na), math.sin(na)
            
            for i in range(N):
                cx, cy = centers[i]
                ai = angles[i]
                # Get vertices of hex i
                verts = get_hex_verts(cx, cy, ai)
                # Project vertices onto normal
                projs = verts @ np.array([nx, ny])
                max_proj_i = np.max(projs)
                if max_proj_i > max_proj:
                    max_proj = max_proj_i
        
        # For a regular hexagon with side length s, the distance from center to side (apothem) is s * sqrt(3) / 2
        # max_proj is the maximum distance from center to any point, which must be <= apothem?
        # No, the vertices of the inner hexagons must be inside the outer hexagon.
        # The condition for a point p to be inside a regular hexagon centered at origin with apothem h is |p . n_k| <= h for all 6 normals n_k.
        # So max_proj must be <= h.
        # h = s * sqrt(3) / 2  => s = 2 * h / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def penalty(params, k=100.0):
        """
        Calculate the collision penalty.
        """
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        
        polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    # Quadratic penalty for overlap
                    pen += k * gap * gap
        return pen

    def objective(params):
        """
        Total objective: Outer Side Length + Collision Penalty
        """
        side = calc_outer_side(params)
        pen = penalty(params)
        return side + pen

    def is_valid(params, tol=1e-5):
        """
        Check if the configuration is valid (no overlaps).
        """
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # Phase 1: Generate initial configurations
    # Use a honeycomb lattice as a starting point
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
    
    best_params = None
    best_val = float('inf')
    
    # Try a few different initial rotations and global rotations
    for global_rot in [0, 30, 60]:
        rot_rad = math.radians(global_rot)
        rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                           [math.sin(rot_rad), math.cos(rot_rad)]])
        rotated_points = init_points @ rot_mat.T
        angles = np.full(N, global_rot)
        
        # Construct initial params
        init_params = np.concatenate([rotated_points.flatten(), angles, [0.0]])
        
        # Quick check validity
        if is_valid(init_params):
            val = objective(init_params)
            if val < best_val:
                best_val = val
                best_params = init_params.copy()
        
        # Run a short optimization from this start
        if time.time() < deadline - 10:
            try:
                res = minimize(objective, init_params, method='Powell',
                               options={'maxiter': 50, 'disp': False})
                if res.fun < best_val:
                    # Verify validity of the result
                    if is_valid(res.x):
                        best_val = res.fun
                        best_params = res.x.copy()
            except Exception:
                pass

    if best_params is None:
        # Fallback: very loose packing
        init_params = np.zeros(3*N + 1)
        for i in range(N):
            init_params[2*i] = (i % 4) * 5
            init_params[2*i+1] = (i // 4) * 5
        best_params = init_params
        best_val = objective(best_params)

    # Phase 2: Refine the best configuration
    # Use a more powerful optimizer with the penalty method
    # Powell is robust but slow. Let's try Nelder-Mead with adaptive step size or BFGS if gradients are approximated.
    # Since sat_gap is not differentiable, Powell or Nelder-Mead are preferred.
    
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                if is_valid(res.x):
                    best_val = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final polish with a stronger penalty to ensure zero overlap
    # Increase penalty coefficient and run a short optimization
    if time.time() < deadline - 2:
        # We need to modify the objective to use a higher penalty
        # Since we can't easily change the k in the nested function without redefining, 
        # we will just run one more step with the current objective.
        # If there are still overlaps, we might need to force them apart.
        
        # Check for overlaps
        centers = best_params[:2*N].reshape(N, 2)
        angles = best_params[2*N:3*N]
        polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        
        overlap = False
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -1e-5:
                    overlap = True
                    break
            if overlap: break
        
        if overlap:
            # Simple relaxation: push overlapping hexagons apart
            for iter in range(10):
                if time.time() > deadline - 1: break
                moved = False
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap(polys[i], polys[j])
                        if gap < 0:
                            # Move centers apart
                            diff = centers[j] - centers[i]
                            dist = np.linalg.norm(diff)
                            if dist < 1e-6:
                                diff = np.array([1.0, 0.0])
                                dist = 1.0
                            else:
                                diff /= dist
                            
                            # Move each by half the penetration depth
                            shift = -0.5 * gap * diff
                            centers[i] -= shift
                            centers[j] += shift
                            moved = True
                            
                            # Update polys
                            polys[i] = get_hex_verts(centers[i][0], centers[i][1], angles[i])
                            polys[j] = get_hex_verts(centers[j][0], centers[j][1], angles[j])
                
                if not moved:
                    break
                
                # Update best_params
                best_params[:2*N] = centers.flatten()
                
                # Recalculate objective
                new_val = objective(best_params)
                if new_val < best_val:
                    best_val = new_val

    # Final Result
    final_centers = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:3*N]
    final_outer_angle = best_params[3*N]
    
    # Calculate final side length
    final_side = calc_outer_side(best_params)
    
    # Ensure validity one last time
    if not is_valid(best_params):
        # If invalid, return a valid but loose configuration
        # This should rarely happen if penalty is strong enough
        print("Warning: Final configuration is invalid")
        # Fallback to initial lattice
        init_params = np.concatenate([init_points.flatten(), np.full(N, 0.0), [0.0]])
        final_centers = init_points
        final_angles = np.full(N, 0.0)
        final_outer_angle = 0.0
        final_side = calc_outer_side(init_params)

    inner = [[final_centers[i][0], final_centers[i][1], final_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
