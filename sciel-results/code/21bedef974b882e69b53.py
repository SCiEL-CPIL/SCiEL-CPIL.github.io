import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Normals for the 6 sides of a hexagon
    unit_normals_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def get_hex_normals(angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        normals = unit_normals_0 @ rot.T
        return normals

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
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
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
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

    def check_validity(positions, angles, tol=1e-4):
        """Check if all hexagons are disjoint."""
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def repair_overlaps(positions, angles, iterations=20):
        """
        Push overlapping hexagons apart.
        Returns updated positions and angles.
        """
        for _ in range(iterations):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    poly_i = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
                    poly_j = get_hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
                    
                    # Check for overlap using SAT
                    overlap = False
                    min_gap = float('inf')
                    push_vec = np.array([0.0, 0.0])
                    
                    # Determine the axis of minimum penetration
                    axes = []
                    for k in range(6):
                        edge_i = poly_i[(k+1)%6] - poly_i[k]
                        n_i = np.array([-edge_i[1], edge_i[0]])
                        n_i = n_i / np.linalg.norm(n_i)
                        axes.append(n_i)
                        edge_j = poly_j[(k+1)%6] - poly_j[k]
                        n_j = np.array([-edge_j[1], edge_j[0]])
                        n_j = n_j / np.linalg.norm(n_j)
                        axes.append(n_j)
                        
                    for axis in axes:
                        proj_i = poly_i @ axis
                        proj_j = poly_j @ axis
                        min_i, max_i = np.min(proj_i), np.max(proj_i)
                        min_j, max_j = np.min(proj_j), np.max(proj_j)
                        
                        gap = max(min_i - max_j, min_j - max_i)
                        if gap < 0:
                            overlap = True
                            if abs(gap) < abs(min_gap):
                                min_gap = gap
                                # Direction to push: from i to j if min_i > min_j? 
                                # Actually, we need to push them apart along the axis.
                                # If max_i > min_j and max_j > min_i, they overlap.
                                # We push i in -axis direction and j in +axis direction?
                                # Or just push the pair apart.
                                # Let's determine the direction from center_i to center_j projected on axis.
                                center_diff = positions[j] - positions[i]
                                proj_diff = np.dot(center_diff, axis)
                                if proj_diff > 0:
                                    # i is "behind" j on this axis
                                    push_vec = axis * (-gap) # Push j forward? No, push both apart.
                                    # We will push i by -push_vec/2 and j by +push_vec/2
                                    # But we need the vector that separates them.
                                    # The gap is negative. The separation needed is -gap.
                                    # Direction: if center_i < center_j on axis, push i left, j right.
                                    if proj_diff >= 0:
                                         push_vec = axis * (-gap)
                                    else:
                                         push_vec = -axis * (-gap)
                                else:
                                    if proj_diff >= 0:
                                         push_vec = axis * (-gap)
                                    else:
                                         push_vec = -axis * (-gap)
                    
                    if overlap:
                        # Apply push
                        # We push i by -push_vec/2 and j by +push_vec/2
                        # But we must be careful with the direction logic above.
                        # Simpler: just move them apart by the penetration depth along the MTV axis.
                        # Let's re-calculate MTV more carefully.
                        min_pen = float('inf')
                        mtv_axis = None
                        for axis in axes:
                            proj_i = poly_i @ axis
                            proj_j = poly_j @ axis
                            min_i, max_i = np.min(proj_i), np.max(proj_i)
                            min_j, max_j = np.min(proj_j), np.max(proj_j)
                            
                            gap = max(min_i - max_j, min_j - max_i)
                            if gap < 0:
                                if abs(gap) < min_pen:
                                    min_pen = abs(gap)
                                    mtv_axis = axis
                                    
                        if mtv_axis is not None:
                            # Determine direction
                            center_diff = positions[j] - positions[i]
                            proj_diff = np.dot(center_diff, mtv_axis)
                            if proj_diff >= 0:
                                mtv_vec = mtv_axis * min_pen
                            else:
                                mtv_vec = -mtv_axis * min_pen
                                
                            # Push i by -mtv_vec/2, j by +mtv_vec/2
                            positions[i] -= mtv_vec / 2.0
                            positions[j] += mtv_vec / 2.0
                            moved = True
            if not moved:
                break
        return positions, angles

    def penalized_objective(params):
        if time.time() > deadline - 5:
            return 1000.0
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Base cost: outer side length
        side = calc_min_outer_side(positions, angles, 0.0)
        
        # Penalty for overlaps
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    # Quadratic penalty to encourage separation
                    penalty += 100.0 * (-gap) ** 2
        
        return side + penalty

    def generate_honeycomb_init():
        """Generate honeycomb lattice initialization."""
        lattice_points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * 1.5
                y = i * SQRT3 / 2.0 + j * SQRT3
                lattice_points.append((x, y))
        
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:N])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        return np.concatenate([init_positions.flatten(), init_angles])

    def generate_ring_init():
        """Generate ring + center initialization."""
        positions = np.zeros((N, 2))
        # Center
        positions[0] = [0, 0]
        # Inner ring of 6
        for i in range(1, 7):
            ang = (i-1) * 60 * math.pi / 180
            positions[i] = [SQRT3 * math.cos(ang), SQRT3 * math.sin(ang)]
        # Outer ring of 5 (in gaps)
        for i in range(7, 12):
            ang = (i-7) * 72 * math.pi / 180 + 30 * math.pi / 180
            positions[i] = [2.5 * math.cos(ang), 2.5 * math.sin(ang)]
        positions -= np.mean(positions, axis=0)
        angles = np.zeros(N)
        return np.concatenate([positions.flatten(), angles])

    def generate_linear_init():
        """Generate linear chain initialization."""
        positions = np.zeros((N, 2))
        for i in range(N):
            positions[i] = [(i - N/2 + 0.5) * SQRT3, 0]
        positions -= np.mean(positions, axis=0)
        angles = np.zeros(N)
        return np.concatenate([positions.flatten(), angles])

    def generate_grid_init():
        """Generate 3x4 grid initialization."""
        positions = np.zeros((N, 2))
        for i in range(3):
            for j in range(4):
                idx = i * 4 + j
                positions[idx] = [(j - 1.5) * 1.8, (i - 1) * 1.6]
        positions -= np.mean(positions, axis=0)
        angles = np.zeros(N)
        return np.concatenate([positions.flatten(), angles])

    def generate_random_init(seed=42):
        """Generate random initialization."""
        rng = np.random.RandomState(seed)
        positions = rng.uniform(-3, 3, size=(N, 2))
        angles = rng.uniform(0, 360, size=N)
        return np.concatenate([positions.flatten(), angles])

    initial_configs = [
        generate_honeycomb_init(),
        generate_ring_init(),
        generate_linear_init(),
        generate_grid_init(),
        generate_random_init(42),
        generate_random_init(123),
    ]

    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 360))

    best_score = float('inf')
    best_result = None

    # Main optimization loop
    for x0 in initial_configs:
        if time.time() > deadline - 20:
            break
        
        current_params = x0.copy()
        
        # Run several rounds of Optimize -> Repair -> Optimize
        for round_idx in range(3):
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    penalized_objective, 
                    current_params, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
                )
                current_params = res.x
                
                # Repair overlaps
                pos = current_params[:2*N].reshape(N, 2)
                ang = current_params[2*N:]
                pos, ang = repair_overlaps(pos, ang, iterations=10)
                current_params = np.concatenate([pos.flatten(), ang])
                
                # Second optimization pass after repair
                res2 = minimize(
                    penalized_objective, 
                    current_params, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
                )
                current_params = res2.x
                
                # Final check
                pos = current_params[:2*N].reshape(N, 2)
                ang = current_params[2*N:]
                
                # If still overlapping, try to repair one more time and accept if valid
                if not check_validity(pos, ang):
                     pos, ang = repair_overlaps(pos, ang, iterations=20)
                     if not check_validity(pos, ang):
                         continue # Skip this result if invalid
                     
                score = calc_min_outer_side(pos, ang, 0.0)
                if score < best_score:
                    best_score = score
                    best_result = np.concatenate([pos.flatten(), ang])
            except Exception:
                pass

    # Final polish with SLSQP if we have a good result
    if best_result is not None and time.time() < deadline - 10:
        try:
            res3 = minimize(
                penalized_objective, 
                best_result, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
            )
            
            pos = res3.x[:2*N].reshape(N, 2)
            ang = res3.x[2*N:]
            
            if check_validity(pos, ang):
                score3 = calc_min_outer_side(pos, ang, 0.0)
                if score3 < best_score:
                    best_score = score3
                    best_result = res3.x
        except Exception:
            pass

    # Finalize
    if best_result is not None:
        final_positions = best_result[:2*N].reshape(N, 2)
        final_angles = best_result[2*N:]
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        # Fallback
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
