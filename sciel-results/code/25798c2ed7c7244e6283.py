import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Unit hexagon vertices (pointy-top, angle 0)
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in [30, 90, 150, 210, 270, 330]])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Compute the minimum side length of a regular hexagon (at given outer_angle)
        that contains all inner hexagons.
        
        The outer hexagon has 6 sides. For each side, the support function gives
        the maximum projection. The side length s relates to the inradius r by s = 2*r/sqrt(3).
        The inradius is the minimum over all 6 side normals of (max projection - center projection).
        We center the outer hexagon at the centroid of the configuration for symmetry.
        """
        # Center of outer hexagon - use centroid of inner hexagon centers
        outer_cx = np.mean(positions[:, 0])
        outer_cy = np.mean(positions[:, 1])
        
        theta_rad = math.radians(outer_angle)
        # For a regular hexagon with side s, the inradius is s*sqrt(3)/2
        # The 6 side normals are at angles: outer_angle + 0, 60, 120, 180, 240, 300 degrees
        # But actually the side normals point outward from center
        # For a hexagon with vertices at angles 0,60,...,300 (pointy-top), 
        # the side normals (perpendicular to edges, pointing outward) are at 30, 90, 150, 210, 270, 330
        # Let me reconsider: for a regular hexagon centered at origin with one vertex at angle 0,
        # vertices are at 0, 60, 120, 180, 240, 300
        # Edge between vertex 0 and vertex 60: midpoint direction is 30 deg, normal is 30 deg
        # So side normals are at 30, 90, 150, 210, 270, 330 degrees
        
        max_proj = 0.0
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            nx = math.cos(ang)
            ny = math.sin(ang)
            
            # Project all vertices of all inner hexagons onto this normal
            for i in range(N):
                verts = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
                projs = verts[:, 0] * nx + verts[:, 1] * ny
                max_p = np.max(projs)
                # Distance from outer center to this projection
                dist = max_p - (outer_cx * nx + outer_cy * ny)
                if dist > max_proj:
                    max_proj = dist
        
        # max_proj is the inradius needed
        side = 2.0 * max_proj / SQRT3
        return side

    def sat_gap(poly1, poly2):
        """Return minimum gap between two convex polygons. Negative means overlap."""
        min_gap = float('inf')
        
        all_axes = []
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                all_axes.append(normal / norm)
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                all_axes.append(normal / norm)
        
        for axis in all_axes:
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

    def is_valid(positions, angles, tol=1e-4):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def compute_penalty(params, penalty_weight=50.0):
        """
        params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, outer_angle]
        Total: 3*N + 1 = 37 parameters
        """
        positions = params[:3*N].reshape(N, 3)[:, :2]
        angles = params[:3*N].reshape(N, 3)[:, 2]
        outer_angle = params[3*N]
        
        side_len = calc_min_outer_side(positions, angles, outer_angle)
        
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap**2
        
        return side_len + penalty

    # Generate multiple initial configurations
    def gen_lattice_init(offset_seed=0):
        random.seed(offset_seed)
        np.random.seed(offset_seed)
        points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5 + random.uniform(-0.1, 0.1)
                y = i * SQRT3 / 2.0 + j * SQRT3 + random.uniform(-0.1, 0.1)
                points.append((x, y))
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_pos = np.array(points[:12])
        init_pos -= np.mean(init_pos, axis=0)
        init_ang = np.zeros(N)
        return init_pos, init_ang

    def gen_ring_init(offset_seed=0):
        random.seed(offset_seed)
        np.random.seed(offset_seed)
        # Place hexagons in a ring pattern
        init_pos = np.zeros((N, 2))
        init_ang = np.zeros(N)
        # 1 in center, 5 in inner ring, 6 in outer ring
        init_pos[0] = [0, 0]
        r1 = 2.0
        for k in range(1, 6):
            ang = 2 * math.pi * (k-1) / 5 + random.uniform(-0.1, 0.1)
            init_pos[k] = [r1 * math.cos(ang), r1 * math.sin(ang)]
            init_ang[k] = random.uniform(0, 360)
        r2 = 3.8
        for k in range(6, 12):
            ang = 2 * math.pi * (k-6) / 6 + math.pi/6 + random.uniform(-0.1, 0.1)
            init_pos[k] = [r2 * math.cos(ang), r2 * math.sin(ang)]
            init_ang[k] = random.uniform(0, 360)
        return init_pos, init_ang

    def gen_random_init(offset_seed=0):
        random.seed(offset_seed)
        np.random.seed(offset_seed)
        init_pos = np.random.randn(N, 2) * 2.0
        init_pos -= np.mean(init_pos, axis=0)
        init_ang = np.random.uniform(0, 360, N)
        return init_pos, init_ang

    # Run simulated annealing from multiple starts
    def simulated_annealing(init_pos, init_ang, max_iter=30000, T_start=0.8, cooling=0.9997, step_start=0.4):
        x0 = np.zeros(3*N + 1)
        for i in range(N):
            x0[3*i] = init_pos[i][0]
            x0[3*i+1] = init_pos[i][1]
            x0[3*i+2] = init_ang[i]
        x0[3*N] = 0.0  # outer angle
        
        best_params = x0.copy()
        best_score = compute_penalty(x0)
        current_params = x0.copy()
        current_score = best_score
        
        T = T_start
        step_size = step_start
        np.random.seed(int(time.time() * 1000) % 100000 + hash(tuple(init_pos[0])) % 1000)
        random.seed(int(time.time() * 2000) % 100000 + hash(tuple(init_ang[0])) % 1000)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            new_params = current_params.copy()
            n_perturb = random.randint(1, 3)
            for _ in range(n_perturb):
                idx = random.randint(0, N-1)
                new_params[3*idx] += np.random.randn() * step_size
                new_params[3*idx+1] += np.random.randn() * step_size
                new_params[3*idx+2] += np.random.randn() * step_size * 40
            # Occasionally perturb outer angle
            if random.random() < 0.1:
                new_params[3*N] += np.random.randn() * step_size * 20
            
            new_score = compute_penalty(new_params)
            
            if new_score < current_score:
                current_params = new_params
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_params = new_params.copy()
            else:
                delta = new_score - current_score
                if T > 1e-6:
                    acceptance_prob = math.exp(-delta / T)
                    if random.random() < acceptance_prob:
                        current_params = new_params
                        current_score = new_score
            
            T *= cooling
            if iteration % 400 == 0:
                step_size *= 0.96
        
        return best_params, best_score

    def slsqp_refine(params, max_iter=50):
        """Use SLSQP to refine a good solution."""
        def objective(x):
            if time.time() > deadline - 10:
                return 1e6
            return compute_penalty(x, penalty_weight=200.0)
        
        def constraint_nonoverlap(x):
            positions = x[:3*N].reshape(N, 3)[:, :2]
            angles = x[:3*N].reshape(N, 3)[:, 2]
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            gaps = []
            for i in range(N):
                for j in range(i+1, N):
                    gaps.append(sat_gap(polys[i], polys[j]))
            return gaps
        
        # Create nonlinear constraints for non-overlap
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                def make_constraint(i, j):
                    def con(x):
                        positions = x[:3*N].reshape(N, 3)[:, :2]
                        angles = x[:3*N].reshape(N, 3)[:, 2]
                        p1 = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
                        p2 = get_hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
                        return sat_gap(p1, p2)
                    return con
                constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
        
        try:
            result = minimize(objective, params, method='SLSQP', constraints=constraints,
                            options={'maxiter': max_iter, 'ftol': 1e-8})
            if result.success or result.fun < compute_penalty(params):
                return result.x
        except Exception:
            pass
        return params

    # Main search: try multiple initializations
    best_overall_params = None
    best_overall_score = float('inf')
    
    initializations = [
        ("lattice0", gen_lattice_init(0)),
        ("lattice1", gen_lattice_init(100)),
        ("lattice2", gen_lattice_init(200)),
        ("ring0", gen_ring_init(0)),
        ("ring1", gen_ring_init(50)),
        ("random0", gen_random_init(0)),
        ("random1", gen_random_init(42)),
    ]
    
    for name, (init_pos, init_ang) in initializations:
        if time.time() > deadline - 30:
            break
        
        # Run SA
        params, score = simulated_annealing(init_pos, init_ang, max_iter=25000)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_params = params.copy()
        
        # If score looks good, try SLSQP refinement
        if score < best_overall_score + 0.5 and time.time() < deadline - 25:
            refined = slsqp_refine(params, max_iter=40)
            refined_score = compute_penalty(refined, penalty_weight=200.0)
            if refined_score < best_overall_score:
                best_overall_score = refined_score
                best_overall_params = refined.copy()
    
    # Final extraction
    final_params = best_overall_params
    final_positions = final_params[:3*N].reshape(N, 3)[:, :2]
    final_angles = final_params[:3*N].reshape(N, 3)[:, 2]
    final_outer_angle = final_params[3*N]
    
    # Validate
    if not is_valid(final_positions, final_angles, tol=1e-4):
        # Fallback: use the best SA result without SLSQP
        # Try lattice init directly
        init_pos, init_ang = gen_lattice_init(0)
        for i in range(N):
            final_positions[i] = init_pos[i]
            final_angles[i] = init_ang[i]
        final_outer_angle = 0.0
    
    final_side = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([float(final_positions[i][0]), float(final_positions[i][1]), float(final_angles[i])])
    
    return inner, (0.0, 0.0), float(final_side), float(final_outer_angle)

if __name__ == "__main__":
    construct()
