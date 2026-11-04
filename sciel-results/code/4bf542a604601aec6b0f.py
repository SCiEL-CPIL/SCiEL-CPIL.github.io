import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precompute unit hexagon vertices (flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees)
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Precompute edge normals for unit hexagon (flat-top)
    # Edge 0: (1,0) to (0.5, sqrt(3)/2). Vector: (-0.5, sqrt(3)/2). Normal: (sqrt(3)/2, 0.5) -> 30 deg
    # The outward normals for a flat-top hexagon are at 30, 90, 150, 210, 270, 330 degrees
    unit_normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def get_hex_normals(angle_deg):
        """Get outward normals of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        normals = unit_normals @ rot.T
        return normals

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex hexagons using SAT.
        If they overlap, returns a negative value (penetration depth).
        If they are separated, returns the positive distance.
        """
        axes = []
        for poly in [poly1, poly2]:
            for i in range(6):
                edge = poly[(i+1)%6] - poly[i]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    axes.append(n / norm)
        
        min_gap = float('inf')
        for axis in axes:
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                if gap < min_gap:
                    min_gap = gap
            else:
                return gap
        
        return min_gap

    def calc_min_outer_side(positions, angles, outer_angle_deg=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at origin
        with rotation outer_angle_deg that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle_deg)
        # Normal directions for the container
        normal_angles = theta_rad + math.pi/6 + np.arange(6) * math.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj_all = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang_deg = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang_deg)
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs, axis=0)  # (6,)
            m = np.max(max_proj_i)
            if m > max_proj_all:
                max_proj_all = m
        
        return 2.0 * max_proj_all / math.sqrt(3)

    def is_valid(positions, angles, tol=1e-5):
        """Check if all hexagons are non-overlapping."""
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- Initialization ---
    # Hexagonal lattice basis
    sqrt3 = math.sqrt(3)
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    # Generate a grid of points
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * base_v1 + j * base_v2
            lattice_points.append(p)
    
    # Select 12 closest to origin
    lattice_points.sort(key=lambda p: np.dot(p, p))
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    # Try rotating the initial configuration to find a good start
    best_init = None
    best_init_score = float('inf')
    
    for rot_deg in [0, 15, 30, 45]:
        rot_rad = math.radians(rot_deg)
        rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                            [math.sin(rot_rad), math.cos(rot_rad)]])
        rotated = init_positions @ rot_mat.T
        angles = np.full(N, rot_deg)
        
        if is_valid(rotated, angles, tol=1e-6):
            score = calc_min_outer_side(rotated, angles, 0.0)
            if score < best_init_score:
                best_init_score = score
                best_init = (rotated.copy(), angles.copy())
    
    if best_init is None:
        best_init = (init_positions, init_angles)
        best_init_score = calc_min_outer_side(init_positions, init_angles, 0.0)
        
    init_positions, init_angles = best_init
    
    # State: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, outer_angle]
    def flatten_state(pos, ang, outer_ang):
        return np.concatenate([pos.flatten(), ang, [outer_ang]])
    
    def unpack_state(state):
        pos = state[:2*N].reshape(N, 2)
        ang = state[2*N:3*N]
        outer_ang = state[3*N]
        return pos, ang, outer_ang

    current_state = flatten_state(init_positions, init_angles, 0.0)
    current_score = best_init_score
    best_state = current_state.copy()
    best_score = current_score

    # --- Simulated Annealing ---
    # Objective for SA: Side length + Large Penalty for Overlap
    def sa_objective(state):
        pos, ang, outer_ang = unpack_state(state)
        side = calc_min_outer_side(pos, ang, outer_ang)
        
        # Calculate overlap penalty
        polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += -gap * 100.0  # Heavy penalty
        
        return side + penalty

    current_cost = sa_objective(current_state)
    
    T = 1.0
    T_min = 0.001
    alpha = 0.9995  # Cooling rate
    max_steps = 50000
    
    for step in range(max_steps):
        if time.time() > deadline - 20:
            break
            
        # Generate neighbor
        new_state = current_state.copy()
        
        # Randomly select which parameter to perturb
        r = random.random()
        if r < 0.6:
            # Move a hexagon
            idx = random.randint(0, N-1)
            new_state[2*idx] += random.gauss(0, 0.1)
            new_state[2*idx+1] += random.gauss(0, 0.1)
        elif r < 0.9:
            # Rotate a hexagon
            idx = random.randint(0, N-1)
            new_state[2*N+idx] += random.gauss(0, 5)
        else:
            # Rotate outer
            new_state[3*N] += random.gauss(0, 5)
            
        new_cost = sa_objective(new_state)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = new_state
            current_cost = new_cost
            
            # Update best if valid
            pos, ang, outer_ang = unpack_state(current_state)
            if is_valid(pos, ang, tol=1e-5):
                valid_score = calc_min_outer_side(pos, ang, outer_ang)
                if valid_score < best_score:
                    best_score = valid_score
                    best_state = current_state.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min

    # --- Phase 2: Local Refinement (Nelder-Mead) ---
    # Start from best valid state found by SA
    if time.time() < deadline - 10:
        def refine_objective(state):
            pos, ang, outer_ang = unpack_state(state)
            side = calc_min_outer_side(pos, ang, outer_ang)
            
            polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += 1000.0 * gap * gap
            return side + penalty

        try:
            res = minimize(
                refine_objective,
                best_state,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'disp': False}
            )
            
            pos, ang, outer_ang = unpack_state(res.x)
            if is_valid(pos, ang, tol=1e-5):
                score = calc_min_outer_side(pos, ang, outer_ang)
                if score < best_score:
                    best_score = score
                    best_state = res.x.copy()
        except Exception:
            pass

    # --- Final Polish: Fine-tune outer angle ---
    if time.time() < deadline - 2:
        final_positions = best_state[:2*N].reshape(N, 2)
        final_angles = best_state[2*N:3*N]
        
        best_outer_angle = best_state[3*N]
        best_outer_score = best_score
        
        # Fine grid search for outer angle
        for outer_angle in np.linspace(0, 60, 181):
            score = calc_min_outer_side(final_positions, final_angles, outer_angle)
            if score < best_outer_score:
                best_outer_score = score
                best_outer_angle = outer_angle
        
        if best_outer_score < best_score:
            best_score = best_outer_score
            best_state[3*N] = best_outer_angle

    # Finalize
    final_positions = best_state[:2*N].reshape(N, 2)
    final_angles = best_state[2*N:3*N]
    final_outer_angle = best_state[3*N]
    final_side = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
