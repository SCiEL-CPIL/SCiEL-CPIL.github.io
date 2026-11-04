import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top vs flat-top? 
    # Standard math hexagon usually has vertices at 0, 60, ... degrees.
    # Let's use standard vertices: (cos(k*pi/3), sin(k*pi/3)) for k=0..5
    # This is a "pointy-top" if y is up.
    # Side length 1.
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
        """
        Calculate the minimum side length of a regular hexagon with center (0,0) 
        and rotation outer_angle that contains all inner hexagons.
        
        For a regular hexagon with side length S, the distance from center to side (inradius) is S*sqrt(3)/2.
        The condition for containment is that the projection of every point of every inner hexagon
        onto the 6 normals of the outer hexagon is <= inradius.
        
        Normals of outer hexagon (pointing outwards) are at angles:
        theta_outer + pi/6 + k*pi/3 for k=0..5
        (Assuming outer hexagon has vertices at theta_outer + k*pi/3, 
         the sides are perpendicular to these normal directions? 
         Let's check. 
         Vertices at 0, 60, ... 
         Side 0 connects 0 and 60. Midpoint angle 30. Normal is 30 degrees.
         So normals are at 30, 90, 150, 210, 270, 330.
         i.e., pi/6 + k*pi/3.
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
        
        # max_proj is the required inradius
        # inradius = side * sqrt(3) / 2
        # side = 2 * inradius / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Calculate minimum gap between two convex hexagons.
        Returns negative value if overlapping (penetration depth).
        Returns positive value if separated.
        """
        min_gap = float('inf')
        
        # Check axes from poly1
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
            
            gap = max(min1 - max2, min2 - min1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        # Check axes from poly2
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
            
            gap = max(min1 - max2, min2 - min1)
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

    def compute_penalty(params, penalty_weight=100.0):
        """
        params layout: [x0, y0, a0, ..., x11, y11, a11, outer_angle]
        Total size: 3*N + 1 = 37
        """
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i][0] = params[3*i]
            positions[i][1] = params[3*i+1]
            angles[i] = params[3*i+2]
            
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

    # Generate initial configuration using hexagonal lattice
    sqrt3 = math.sqrt(3)
    # Triangular lattice vectors for unit hexagons (side 1)
    # Distance between centers for touching hexagons is 2 * inradius = sqrt(3)
    # Lattice vectors: a1 = (1.5, sqrt(3)/2), a2 = (0, sqrt(3))
    # |a1| = sqrt(2.25 + 0.75) = sqrt(3). Correct.
    a1 = np.array([1.5, sqrt3/2])
    a2 = np.array([0.0, sqrt3])
    
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * a1 + j * a2
            lattice_points.append(p)
    
    # Sort by distance from origin, take closest 12
    lattice_points.sort(key=lambda p: np.dot(p, p))
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    # Pack into params array: [x, y, angle] * N, outer_angle
    x0 = np.zeros(3*N + 1)
    for i in range(N):
        x0[3*i] = init_positions[i][0]
        x0[3*i+1] = init_positions[i][1]
        x0[3*i+2] = init_angles[i]
    x0[3*N] = 0.0 # Initial outer angle
    
    # Simulated Annealing
    best_params = x0.copy()
    best_score = compute_penalty(x0)
    current_params = x0.copy()
    current_score = best_score
    
    T = 0.5  # initial temperature
    cooling_rate = 0.9995
    step_size = 0.3
    max_iter = 50000
    
    np.random.seed(42)
    random.seed(42)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 30.0:
            break
        
        # Generate perturbation
        new_params = current_params.copy()
        
        # Randomly perturb 1-3 hexagons
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, N-1)
            # Perturb position
            new_params[3*idx] += np.random.randn() * step_size
            new_params[3*idx+1] += np.random.randn() * step_size
            # Perturb angle
            new_params[3*idx+2] += np.random.randn() * step_size * 30
            
        # Occasionally perturb outer angle
        if random.random() < 0.1:
            new_params[3*N] += np.random.randn() * 0.1 # Radians? No, degrees in our helper?
            # calc_min_outer_side takes degrees.
            # Let's keep angles in degrees for consistency with get_hex_verts.
            # Wait, get_hex_verts takes angle_deg.
            # So params are in degrees.
            # Perturbation 0.1 degrees is small. Maybe 5 degrees?
            new_params[3*N] += np.random.randn() * 5.0

        new_score = compute_penalty(new_params)
        
        # Acceptance criterion
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        else:
            delta = new_score - current_score
            if T > 0:
                acceptance_prob = math.exp(-delta / T)
                if random.random() < acceptance_prob:
                    current_params = new_params
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        # Reduce step size slowly
        if iteration % 500 == 0:
            step_size *= 0.95
    
    # Extract best configuration
    best_positions = np.zeros((N, 2))
    best_angles = np.zeros(N)
    best_outer_angle = best_params[3*N]
    for i in range(N):
        best_positions[i][0] = best_params[3*i]
        best_positions[i][1] = best_params[3*i+1]
        best_angles[i] = best_params[3*i+2]
    
    if not is_valid(best_positions, best_angles, tol=1e-5):
        # Try to repair by using a simpler initial configuration if SA failed
        # This is a fallback, usually SA with penalty should be valid
        fallback_positions = init_positions.copy()
        fallback_angles = init_angles.copy()
        best_positions = fallback_positions
        best_angles = fallback_angles
        best_outer_angle = 0.0

    # Now do a local polish with SLSQP to tighten the fit
    if time.time() < deadline - 10.0:
        def objective(params):
            positions = np.zeros((N, 2))
            angles = np.zeros(N)
            for i in range(N):
                positions[i][0] = params[3*i]
                positions[i][1] = params[3*i+1]
                angles[i] = params[3*i+2]
            
            outer_angle = params[3*N]
            side_len = calc_min_outer_side(positions, angles, outer_angle)
            
            # Penalty for overlaps
            penalty = 0.0
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += 50.0 * gap**2
            
            return side_len + penalty
        
        try:
            result = minimize(
                objective,
                best_params,
                method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.success or result.fun < best_score:
                # Check if the result is valid
                result_positions = np.zeros((N, 2))
                result_angles = np.zeros(N)
                for i in range(N):
                    result_positions[i][0] = result.x[3*i]
                    result_positions[i][1] = result.x[3*i+1]
                    result_angles[i] = result.x[3*i+2]
                
                if is_valid(result_positions, result_angles, tol=1e-5):
                    best_positions = result_positions
                    best_angles = result_angles
                    best_outer_angle = result.x[3*N]
                    best_params = result.x.copy()
                    best_score = result.fun
        except Exception:
            pass
    
    final_side = calc_min_outer_side(best_positions, best_angles, best_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), final_side, best_outer_angle
# EVOLVE-BLOCK-END
