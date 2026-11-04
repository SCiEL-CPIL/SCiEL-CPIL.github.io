import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        return max_proj * 2.0 / math.sqrt(3)

    def find_best_outer_angle(positions, angles, coarse=False):
        best_side = float('inf')
        best_angle = 0.0
        n_steps = 61 if coarse else 361
        test_angles = np.linspace(0, 60, n_steps)
        for test_angle in test_angles:
            side = calc_min_outer_side(positions, angles, test_angle)
            if side < best_side:
                best_side = side
                best_angle = test_angle
        if not coarse:
            fine_angles = np.linspace(max(0, best_angle - 0.5), min(60, best_angle + 0.5), 101)
            for test_angle in fine_angles:
                side = calc_min_outer_side(positions, angles, test_angle)
                if side < best_side:
                    best_side = side
                    best_angle = test_angle
        return best_side, best_angle

    def sat_gap(poly1, poly2):
        min_gap = float('inf')
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            axis = normal / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            axis = normal / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
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

    # 3x4 lattice topology: 3 columns, 4 rows
    # Standard hex lattice: v1 = (1.5, sqrt(3)/2), v2 = (0, sqrt(3))
    # For 3x4: positions = i*v1 + j*v2 for i in 0..2, j in 0..3
    # Center the configuration

    sqrt3 = math.sqrt(3)

    def lattice_config(dx, dy, shear, rot_deg, angle_offset):
        """
        Generate 12 positions on a sheared 3x4 hex lattice.
        dx: x-spacing scale, dy: y-spacing scale, shear: shear angle (degrees),
        rot_deg: global rotation, angle_offset: individual hex rotation
        """
        # Base lattice vectors for touching hexagons
        v1 = np.array([1.5 * dx, sqrt3/2 * dx])
        v2 = np.array([0, sqrt3 * dy])
        
        # Apply shear
        shear_rad = math.radians(shear)
        shear_mat = np.array([[1.0, math.tan(shear_rad)], [0, 1.0]])
        v1 = shear_mat @ v1
        v2 = shear_mat @ v2
        
        # Generate 3x4 grid
        points = []
        for i in range(3):
            for j in range(4):
                p = i * v1 + j * v2
                points.append(p)
        points = np.array(points)
        
        # Center
        centroid = np.mean(points, axis=0)
        points -= centroid
        
        # Global rotation
        rot_rad = math.radians(rot_deg)
        rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                            [math.sin(rot_rad), math.cos(rot_rad)]])
        points = points @ rot_mat.T
        
        angles = np.full(N, angle_offset + rot_deg)
        return points, angles

    # Phase 1: Coarse grid search over lattice parameters
    best_config = None
    best_score = float('inf')

    dx_range = [0.95, 0.98, 1.0, 1.02, 1.05]
    dy_range = [0.95, 0.98, 1.0, 1.02, 1.05]
    shear_range = [-10, -5, 0, 5, 10]
    rot_range = [0, 10, 20, 30, 40, 50]
    ang_range = [0, 30, 60, 90]

    for dx in dx_range:
        for dy in dy_range:
            for sh in shear_range:
                for rot in rot_range:
                    for ang in ang_range:
                        if time.time() > deadline - 40:
                            break
                        positions, angles = lattice_config(dx, dy, sh, rot, ang)
                        if not is_valid(positions, angles, tol=1e-4):
                            continue
                        score, _ = find_best_outer_angle(positions, angles, coarse=True)
                        if score < best_score:
                            best_score = score
                            best_config = (positions.copy(), angles.copy())

    # Fallback
    if best_config is None:
        positions, angles = lattice_config(1.0, 1.0, 0, 0, 0)
        best_config = (positions, angles)
        best_score, _ = find_best_outer_angle(positions, angles, coarse=True)

    init_positions, init_angles = best_config
    x0 = np.concatenate([init_positions.flatten(), init_angles])

    # Phase 2: SLSQP refinement
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

    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 360))

    best_result = x0.copy()
    best_outer_angle = 0.0

    if time.time() < deadline - 25:
        try:
            res = minimize(
                objective_fast,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-10, 'disp': False}
            )
            if res.fun < best_score:
                pos = res.x[:2*N].reshape(N, 2)
                ang = res.x[2*N:]
                if is_valid(pos, ang, tol=1e-4):
                    score, outer_ang = find_best_outer_angle(pos, ang, coarse=False)
                    if score < best_score:
                        best_score = score
                        best_result = res.x.copy()
                        best_outer_angle = outer_ang
        except Exception:
            pass

    # Phase 3: Nelder-Mead polishing
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
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'disp': False}
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

    final_positions = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:]
    final_side, final_outer_angle = find_best_outer_angle(final_positions, final_angles, coarse=False)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])

    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
