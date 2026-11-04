import math
import time
import numpy as np
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit
    
    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
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
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normal_angles = theta_rad + math.pi/6 + np.arange(6) * math.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj_all = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang_deg = angles[i]
            ang_rad = math.radians(ang_deg)
            c, s = math.cos(ang_rad), math.sin(ang_rad)
            # Rotate unit hex vertices
            ux = unit_verts_0[:, 0]
            uy = unit_verts_0[:, 1]
            rx = c * ux - s * uy
            ry = s * ux + c * uy
            verts = np.column_stack([rx + cx, ry + cy])
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs, axis=0)  # (6,)
            m = np.max(max_proj_i)
            if m > max_proj_all:
                max_proj_all = m
        
        # The apothem of the outer hexagon is max_proj_all
        # Side length = 2 * apothem / sqrt(3)
        return max_proj_all * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex hexagons.
        Uses SAT. Returns negative value if intersecting (penetration depth).
        """
        axes1 = []
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes1.append(n / norm)
        
        axes2 = []
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes2.append(n / norm)
        
        min_gap = float('inf')
        for axis in axes1:
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        
        for axis in axes2:
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

    # --- PHASE 1: INITIAL TOPOLOGY ---
    # Start with a valid lattice configuration
    sqrt3 = math.sqrt(3)
    best_valid_config = None
    best_valid_score = float('inf')
    
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    # Generate a dense set of candidate initial positions from a scaled lattice
    # We want to explore different "tightness" levels
    for scale in [0.95, 1.0, 1.05, 1.1]:
        v1 = base_v1 * scale
        v2 = base_v2 * scale
        points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                p = i * v1 + j * v2
                points.append(p)
        
        points.sort(key=lambda p: np.dot(p, p))
        selected = np.array(points[:12])
        centroid = np.mean(selected, axis=0)
        selected -= centroid
        
        for rot_deg in [0, 30, 60, 15, 45, 75]:
            rot_rad = math.radians(rot_deg)
            rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                [math.sin(rot_rad), math.cos(rot_rad)]])
            rotated_selected = selected @ rot_mat.T
            angles = np.full(N, rot_deg)
            
            if is_valid(rotated_selected, angles, tol=1e-6):
                for oa in [0, 5, 10, 15, 20, 25, 30]:
                    score = calc_min_outer_side(rotated_selected, angles, oa)
                    if score < best_valid_score:
                        best_valid_score = score
                        best_valid_config = (rotated_selected.copy(), angles.copy(), oa)
            
            if time.time() > deadline - 60:
                break
        if time.time() > deadline - 60:
            break

    if best_valid_config is None:
        # Fallback: simple lattice
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        best_valid_config = (init_positions, init_angles, 0.0)
        best_valid_score = calc_min_outer_side(init_positions, init_angles, 0.0)

    init_positions, init_angles, init_outer_angle = best_valid_config
    best_score = best_valid_score
    best_positions = init_positions.copy()
    best_angles = init_angles.copy()
    best_outer_angle = init_outer_angle

    # --- PHASE 2: KINETIC ANNEALING ---
    # Strategy: 
    # 1. Randomly perturb positions and angles.
    # 2. Calculate new outer side length.
    # 3. Check validity.
    # 4. Accept if valid and better, or with probability exp(-delta/E) if worse but valid.
    # 5. If invalid, calculate penalty based on penetration depth and accept with lower prob.
    #    This allows escaping local minima by temporarily allowing overlaps.
    
    current_positions = init_positions.copy()
    current_angles = init_angles.copy()
    current_outer_angle = init_outer_angle
    current_side = best_score
    
    # Cooling schedule
    # Start with high temperature to allow large moves and overlaps
    T_start = 0.5
    T_end = 0.001
    # We don't know exact step count, so we use time-based cooling
    # T(t) = T_start * (T_end/T_start)^(t / T_total)
    T_total = deadline - time.time() - 5.0 # Leave 5s for final polish
    
    # Precompute pair list for collision checks
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Adaptive step size
    step_size_pos = 0.2
    step_size_ang = 15.0
    
    while time.time() < deadline - 5.0:
        # Update temperature
        elapsed = time.time() - start_time
        remaining_ratio = max(0.0, (T_total - (elapsed - (start_time + 0.1))) / T_total)
        # Ensure T doesn't drop too fast initially
        if elapsed < 1.0:
            T = T_start
        else:
            T = T_start * (T_end / T_start) ** (elapsed / T_total)
        
        # Generate move
        # Randomly select a subset of hexagons to move? Or all?
        # Moving all is simpler but less flexible. Let's move all with small noise.
        # Or move random single hexagons to allow local relaxation.
        # Hybrid: 50% move all, 50% move random single.
        
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        if random.random() < 0.5:
            # Move all
            for i in range(N):
                new_positions[i, 0] += random.gauss(0, step_size_pos)
                new_positions[i, 1] += random.gauss(0, step_size_pos)
                new_angles[i] += random.gauss(0, step_size_ang)
        else:
            # Move random single
            idx = random.randint(0, N-1)
            new_positions[idx, 0] += random.gauss(0, step_size_pos * 2)
            new_positions[idx, 1] += random.gauss(0, step_size_pos * 2)
            new_angles[idx] += random.gauss(0, step_size_ang * 2)
        
        # Normalize angles
        new_angles = np.mod(new_angles, 360.0)
        
        # Calculate new side length
        new_side = calc_min_outer_side(new_positions, new_angles, current_outer_angle)
        
        # Check validity and calculate penalty
        penalty = 0.0
        valid = True
        polys_new = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
        for i, j in pairs:
            gap = sat_gap(polys_new[i], polys_new[j])
            if gap < 0:
                valid = False
                # Penalty proportional to square of penetration depth
                penalty += gap * gap
        
        # Acceptance criteria
        if valid:
            if new_side < current_side:
                # Always accept improvement
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                
                if new_side < best_score:
                    best_score = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    # Occasionally update best outer angle? 
                    # For now, keep outer angle fixed during SA, polish later.
            else:
                # Worse side length but valid
                delta = new_side - current_side
                if delta < 0:
                    accept = True
                else:
                    accept_prob = math.exp(-delta / T)
                    accept = random.random() < accept_prob
                
                if accept:
                    current_positions = new_positions
                    current_angles = new_angles
                    current_side = new_side
        else:
            # Invalid configuration
            # Accept with probability based on penalty and temperature
            # Penalty is squared penetration. Scale it.
            penalty_scaled = penalty * 10.0 # Tune this scaling
            accept_prob = math.exp(-penalty_scaled / T)
            if random.random() < accept_prob:
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side # Note: side length is calculated even if invalid, but we track it for continuity
        
        # Adaptive step size adjustment
        # If we are stuck (no improvement in best_score for a while), increase T or step size?
        # Simple heuristic: if T is low, reduce step size to refine.
        if T < 0.1:
            step_size_pos *= 0.98
            step_size_ang *= 0.98
        else:
            step_size_pos = min(step_size_pos * 1.01, 0.5)
            step_size_ang = min(step_size_ang * 1.01, 30.0)

    # --- PHASE 3: FINAL LOCAL POLISH ---
    # Use Nelder-Mead or similar derivative-free optimization on the best found configuration
    # to squeeze out remaining slack.
    
    if time.time() < deadline - 2.0:
        # Define objective function for local optimizer
        def obj_fn(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            
            # Check validity
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            penalty = 0.0
            for i, j in pairs:
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += gap * gap * 100.0 # Strong penalty for overlap
            
            side = calc_min_outer_side(positions, angles, best_outer_angle)
            return side + penalty
        
        x0 = np.concatenate([best_positions.flatten(), best_angles])
        
        try:
            from scipy.optimize import minimize
            res = minimize(obj_fn, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_score + 1e-6: # Check if it's significantly better or just valid
                pos = res.x[:2*N].reshape(N, 2)
                ang = res.x[2*N:3*N]
                
                # Verify validity
                if is_valid(pos, ang, tol=1e-5):
                    final_side = calc_min_outer_side(pos, ang, best_outer_angle)
                    if final_side < best_score:
                        best_score = final_side
                        best_positions = pos
                        best_angles = ang
        except Exception:
            pass

    # --- PHASE 4: OUTER ANGLE OPTIMIZATION ---
    # The outer hexagon angle was fixed during SA. Now optimize it.
    # Since the objective is convex w.r.t outer angle for fixed inner positions,
    # a simple grid search or ternary search works.
    
    if time.time() < deadline - 0.5:
        best_oa_score = best_score
        best_oa_angle = best_outer_angle
        
        # Grid search around current best angle
        for oa in np.arange(best_outer_angle - 15, best_outer_angle + 15, 1.0):
            oa = oa % 60.0 # Hexagon symmetry
            s = calc_min_outer_side(best_positions, best_angles, oa)
            if s < best_oa_score:
                best_oa_score = s
                best_oa_angle = oa
        
        # Refine with finer grid
        for oa in np.arange(best_oa_angle - 1, best_oa_angle + 1, 0.1):
            oa = oa % 60.0
            s = calc_min_outer_side(best_positions, best_angles, oa)
            if s < best_oa_score:
                best_oa_score = s
                best_oa_angle = oa
        
        best_score = best_oa_score
        best_outer_angle = best_oa_angle

    # Finalize
    final_positions = best_positions
    final_angles = best_angles
    final_outer_angle = best_outer_angle
    final_side = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
