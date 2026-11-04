import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Precompute normals for the 6 sides of a unit hexagon
    # Edge 0: (1,0) to (0.5, sqrt(3)/2). Normal points outward: (sqrt(3)/2, 0.5) -> 30 deg
    unit_normals_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def find_best_outer_angle(positions, angles, coarse=False):
        """Find the optimal rotation of the outer hexagon that minimizes its side length."""
        best_side = float('inf')
        best_angle = 0.0
        
        if coarse:
            # Very fast check for use inside optimizers
            test_angles = np.linspace(0, 60, 13) # 5 degree steps
        else:
            # Accurate check for final result
            test_angles = np.linspace(0, 60, 61) # 1 degree steps

        for test_angle in test_angles:
            side = calc_min_outer_side(positions, angles, test_angle)
            if side < best_side:
                best_side = side
                best_angle = test_angle
        
        if not coarse:
            # Refine with finer search around the best
            fine_angles = np.linspace(max(0, best_angle - 1), min(60, best_angle + 1), 41)
            for test_angle in fine_angles:
                side = calc_min_outer_side(positions, angles, test_angle)
                if side < best_side:
                    best_side = side
                    best_angle = test_angle
        
        return best_side, best_angle

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
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
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
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

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- PHASE 1: DISCRETE GRID SEARCH FOR TOPOLOGY ---
    sqrt3 = math.sqrt(3)
    
    best_valid_config = None
    best_valid_score = float('inf')
    
    # Standard dense packing lattice vectors
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    # Try a few scales and rotations. Keep it small to avoid timeout.
    scales = [0.98, 1.0, 1.02]
    rotations = [0, 30]
    
    for sx in scales:
        for sy in scales:
            v1 = base_v1 * np.array([sx, 1.0])
            v2 = base_v2 * np.array([1.0, sy])
            
            points = []
            for i in range(-3, 4):
                for j in range(-3, 4):
                    p = i * v1 + j * v2
                    points.append(p)
            
            points.sort(key=lambda p: np.dot(p, p))
            selected = np.array(points[:12])
            centroid = np.mean(selected, axis=0)
            selected -= centroid
            
            for rot_deg in rotations:
                rot_rad = math.radians(rot_deg)
                rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                    [math.sin(rot_rad), math.cos(rot_rad)]])
                rotated_selected = selected @ rot_mat.T
                angles = np.full(N, rot_deg)
                
                if is_valid(rotated_selected, angles, tol=1e-6):
                    # Use coarse angle search for speed in grid phase
                    score, _ = find_best_outer_angle(rotated_selected, angles, coarse=True)
                    if score < best_valid_score:
                        best_valid_score = score
                        best_valid_config = (rotated_selected.copy(), angles.copy())
                
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break

    # Fallback if grid search failed or timed out
    if best_valid_config is None:
        # Simple fallback: honeycomb subset
        lattice_points = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        best_valid_config = (init_positions, init_angles)
        best_valid_score, _ = find_best_outer_angle(init_positions, init_angles, coarse=True)

    init_positions, init_angles = best_valid_config
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    # --- PHASE 2: SLSQP REFINEMENT ---
    # Use a FAST objective for SLSQP (coarse outer angle) to allow more iterations
    def objective_fast(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        side, _ = find_best_outer_angle(positions, angles, coarse=True)
        return side

    def make_constraint(i, j):
        def constraint(params):
            pos_i = params[2*i:2*i+2]
            pos_j = params[2*j:2*j+2]
            ang_i = params[2*N + i]
            ang_j = params[2*N + j]
            
            poly_i = get_hex_verts(pos_i[0], pos_i[1], 1.0, ang_i)
            poly_j = get_hex_verts(pos_j[0], pos_j[1], 1.0, ang_j)
            
            return sat_gap(poly_i, poly_j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    max_radius = 10.0
    bounds = []
    for i in range(N):
        bounds.append((-max_radius, max_radius))
        bounds.append((-max_radius, max_radius))
    for i in range(N):
        bounds.append((0, 360))

    best_score = best_valid_score
    best_result = x0.copy()
    best_outer_angle = 0.0

    if time.time() < deadline - 20:
        try:
            # Reduced maxiter to ensure it finishes
            res = minimize(
                objective_fast, 
                x0, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                pos = res.x[:2*N].reshape(N, 2)
                ang = res.x[2*N:]
                
                if is_valid(pos, ang, tol=1e-4):
                    score, outer_ang = find_best_outer_angle(pos, ang, coarse=False)
                    if score < best_score:
                        best_score = score
                        best_result = res.x.copy()
                        best_outer_angle = outer_ang
        except Exception as e:
            print(f"SLSQP error: {e}")

    # --- PHASE 3: LOCAL POLISHING (Nelder-Mead) ---
    if time.time() < deadline - 10:
        def obj_penalty(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            side_len, _ = find_best_outer_angle(positions, angles, coarse=True)
            penalty = 0.0
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += 1000.0 * gap**2
            return side_len + penalty

        try:
            res2 = minimize(
                obj_penalty, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'disp': False}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:]
            
            if is_valid(pos, ang, tol=1e-5):
                score2, outer_ang2 = find_best_outer_angle(pos, ang, coarse=False)
                if score2 < best_score:
                    best_score = score2
                    best_result = res2.x.copy()
                    best_outer_angle = outer_ang2
        except Exception:
            pass

    # Finalize
    final_positions = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:]
    
    # One last accurate check for outer angle
    final_side, final_outer_angle = find_best_outer_angle(final_positions, final_angles, coarse=False)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
