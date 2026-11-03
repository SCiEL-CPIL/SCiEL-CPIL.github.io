import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

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

    def objective(params):
        if time.time() > deadline - 5:
            return 1000.0
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        return calc_min_outer_side(positions, angles, 0.0)

    def make_constraint(i, j):
        def constraint(params):
            if time.time() > deadline - 5:
                return 1.0
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            poly_i = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            poly_j = get_hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
            return sat_gap(poly_i, poly_j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    # Generate multiple initial configurations
    sqrt3 = math.sqrt(3)
    
    def generate_honeycomb_init():
        """Generate honeycomb lattice initialization."""
        lattice_points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
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
            positions[i] = [sqrt3 * math.cos(ang), sqrt3 * math.sin(ang)]
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
            positions[i] = [(i - N/2 + 0.5) * sqrt3, 0]
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

    # Try SLSQP from multiple initial configurations
    for x0 in initial_configs:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-8, 'disp': False}
            )
            
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:]
            
            if check_validity(pos, ang):
                score = calc_min_outer_side(pos, ang, 0.0)
                if score < best_score:
                    best_score = score
                    best_result = res.x
        except Exception:
            pass

    # Simulated Annealing with contact graph
    def simulated_annealing(start_params, max_iter=5000, t0=1.0, t_end=0.01):
        """SA that minimizes outer side while maintaining disjointness."""
        params = start_params.copy()
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        current_score = calc_min_outer_side(positions, angles, 0.0)
        if not check_validity(positions, angles):
            return None, float('inf')
        
        best_params = params.copy()
        best_score_sa = current_score
        
        rng = np.random.RandomState(42)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Generate perturbation
            new_params = params.copy()
            for i in range(N):
                if rng.random() < 0.7:  # Move position
                    new_params[2*i] += rng.normal(0, 0.3)
                    new_params[2*i+1] += rng.normal(0, 0.3)
                if rng.random() < 0.3:  # Rotate
                    new_params[2*N+i] = (new_params[2*N+i] + rng.normal(0, 15)) % 360
            
            new_positions = new_params[:2*N].reshape(N, 2)
            new_angles = new_params[2*N:]
            
            # Check validity
            if not check_validity(new_positions, new_angles):
                continue
            
            new_score = calc_min_outer_side(new_positions, new_angles, 0.0)
            
            # Acceptance probability
            t = t0 * (t_end / t0) ** (it / max_iter)
            delta = new_score - current_score
            if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
                params = new_params
                current_score = new_score
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_params = params.copy()
        
        return best_params, best_score_sa

    # Run SA from best SLSQP result and other good starts
    if best_result is not None:
        sa_params, sa_score = simulated_annealing(best_result, max_iter=8000)
        if sa_params is not None and sa_score < best_score:
            best_score = sa_score
            best_result = sa_params

    # Also try SA from other initial configs
    for x0 in initial_configs[:3]:
        if time.time() > deadline - 15:
            break
        sa_params, sa_score = simulated_annealing(x0, max_iter=3000)
        if sa_params is not None and sa_score < best_score:
            best_score = sa_score
            best_result = sa_params

    # Final polish with SLSQP
    if best_result is not None and time.time() < deadline - 10:
        try:
            res3 = minimize(
                objective, 
                best_result, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
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
