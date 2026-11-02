import numpy as np
import time
import random

N = 12
EPS = 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions, angles_deg):
        all_verts = []
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            all_verts.append(verts)
        
        verts_flat = np.vstack(all_verts)
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def check_validity(positions, angles_deg, tol=1e-5):
        vertices_list = []
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                disjoint = False
                for k in range(6):
                    edge = p1[(k + 1) % 6] - p1[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        disjoint = True
                        break
                
                if not disjoint:
                    for k in range(6):
                        edge = p2[(k + 1) % 6] - p2[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                            disjoint = True
                            break
                    
                    if not disjoint:
                        return False
        return True

    def overlap_penalty(positions, angles_deg):
        """
        Calculate a smooth penalty for overlaps.
        Uses the minimum separation distance along SAT axes.
        If separated, penalty is 0. If overlapping, penalty is proportional to the depth of overlap.
        """
        vertices_list = []
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            vertices_list.append(verts)
        
        total_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                min_sep = float('inf')
                
                # Check axes from p1
                for k in range(6):
                    edge = p1[(k + 1) % 6] - p1[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10: continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    # Separation is min(proj1) - max(proj2) OR min(proj2) - max(proj1)
                    sep = min(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
                    if sep < min_sep:
                        min_sep = sep
                
                # Check axes from p2
                for k in range(6):
                    edge = p2[(k + 1) % 6] - p2[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10: continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    sep = min(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
                    if sep < min_sep:
                        min_sep = sep
                
                # If min_sep is negative, they overlap. The depth is -min_sep.
                # We want a smooth penalty.
                if min_sep < 0:
                    total_penalty += (-min_sep) ** 2
        
        return total_penalty

    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles_deg = params[2*N:3*N]
        side = required_outer_side(positions, angles_deg)
        penalty = overlap_penalty(positions, angles_deg)
        # Heuristic weight for penalty
        return side + 100.0 * penalty

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1_pos = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1_pos.append([x, y])
    init1_pos = np.array(init1_pos)
    init1_angles = np.zeros(N)
    configs.append(np.concatenate([init1_pos.flatten(), init1_angles]))
    
    # 2. Circular arrangement
    init2_pos = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2_pos[i, 0] = r * np.cos(angle)
        init2_pos[i, 1] = r * np.sin(angle)
    init2_angles = np.zeros(N)
    configs.append(np.concatenate([init2_pos.flatten(), init2_angles]))
    
    # 3. Random
    np.random.seed(42)
    init3_pos = np.random.uniform(-5, 5, (N, 2))
    init3_angles = np.random.uniform(0, 60, N)
    configs.append(np.concatenate([init3_pos.flatten(), init3_angles]))
    
    best_params = None
    best_score = float('inf')
    best_valid_side = float('inf')
    
    # Simulated Annealing
    for init in configs:
        if time.time() > deadline:
            break
            
        current_params = init.copy()
        current_score = objective(current_params)
        
        # Ensure we start with a valid config if possible, or at least track the best valid one
        cur_pos = current_params[:2*N].reshape(N, 2)
        cur_angles = current_params[2*N:3*N]
        if check_validity(cur_pos, cur_angles):
            side = required_outer_side(cur_pos, cur_angles)
            if side < best_valid_side:
                best_valid_side = side
                best_params = current_params.copy()
        
        T = 1.0
        T_min = 1e-6
        alpha = 0.9995 # Cooling rate
        
        while T > T_min and time.time() < deadline:
            # Perturbation
            new_params = current_params.copy()
            
            # Move a random hexagon
            idx = random.randint(0, N-1)
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            da = random.gauss(0, 10)
            
            new_params[idx*2] += dx
            new_params[idx*2+1] += dy
            new_params[2*N+idx] += da
            
            # Occasionally move multiple
            if random.random() < 0.2:
                idx2 = random.randint(0, N-1)
                if idx2 != idx:
                    dx2 = random.gauss(0, 0.5)
                    dy2 = random.gauss(0, 0.5)
                    da2 = random.gauss(0, 10)
                    new_params[idx2*2] += dx2
                    new_params[idx2*2+1] += dy2
                    new_params[2*N+idx2] += da2
            
            new_score = objective(new_params)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_params = new_params
                current_score = new_score
                
                # Check validity and update best
                pos = current_params[:2*N].reshape(N, 2)
                angles = current_params[2*N:3*N]
                if check_validity(pos, angles):
                    side = required_outer_side(pos, angles)
                    if side < best_valid_side:
                        best_valid_side = side
                        best_params = current_params.copy()
            
            T *= alpha
    
    # Fallback
    if best_params is None:
        # Use the first config, scaled to be valid
        best_params = configs[0].copy()
        pos = best_params[:2*N].reshape(N, 2)
        angles = best_params[2*N:3*N]
        if not check_validity(pos, angles):
             for scale in [1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled_pos = pos * scale
                if check_validity(scaled_pos, angles):
                    best_params[:2*N] = scaled_pos.flatten()
                    break
        best_valid_side = required_outer_side(best_params[:2*N].reshape(N,2), best_params[2*N:3*N])
    
    final_pos = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:3*N]
    
    # Final validation
    if not check_validity(final_pos, final_angles):
        # Try to repair by scaling
        for scale in [1.1, 1.2, 1.5, 2.0]:
            scaled_pos = final_pos * scale
            if check_validity(scaled_pos, final_angles):
                final_pos = scaled_pos
                break
        else:
            # Last resort: linear arrangement
            final_pos = np.array([[i * 3.0, 0.0] for i in range(N)])
            final_angles = np.zeros(N)
            
    final_side = required_outer_side(final_pos, final_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(final_pos[i, 0]), float(final_pos[i, 1]), float(final_angles[i])])
        
    return inner, (0.0, 0.0), float(final_side), 0.0
