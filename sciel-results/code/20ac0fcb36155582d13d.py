import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
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

    def check_validity(positions):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                # SAT check
                disjoint = False
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                        disjoint = True
                        break
                
                if not disjoint:
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    # Simulated Annealing
    def simulated_annealing(init_positions, max_iters=50000, T_start=5.0, T_end=0.01):
        current = init_positions.copy()
        current_side = required_outer_side(current)
        if not check_validity(current):
            # Try to fix initial config by spreading out
            for i in range(N):
                current[i] *= 1.5
            current_side = required_outer_side(current)
            if not check_validity(current):
                return init_positions, init_side
        
        best = current.copy()
        best_side = current_side
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Temperature schedule
            T = T_start * (T_end / T_start) ** (it / max_iters)
            
            # Random perturbation: move one hexagon
            idx = np.random.randint(N)
            dx = np.random.normal(0, 0.5)
            dy = np.random.normal(0, 0.5)
            
            new_pos = current.copy()
            new_pos[idx] += np.array([dx, dy])
            
            # Check validity of new config
            if not check_validity(new_pos):
                continue
            
            new_side = required_outer_side(new_pos)
            delta = new_side - current_side
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-9)):
                current = new_pos
                current_side = new_side
                if current_side < best_side:
                    best = current.copy()
                    best_side = current_side
        
        return best, best_side

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    configs.append(np.array(init1))
    
    # 2. Circular arrangement
    init2 = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2[i, 0] = r * np.cos(angle)
        init2[i, 1] = r * np.sin(angle)
    configs.append(init2)
    
    # 3. Random
    np.random.seed(42)
    init3 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init3)
    
    best_positions = None
    best_side = float('inf')
    
    for init in configs:
        if time.time() > deadline:
            break
        
        # Ensure initial config is valid; if not, spread out
        if not check_validity(init):
            for scale in [1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = init * scale
                if check_validity(scaled):
                    init = scaled
                    break
            else:
                continue
        
        result_pos, result_side = simulated_annealing(init)
        
        if result_side < best_side:
            best_side = result_side
            best_positions = result_pos
    
    # Fallback
    if best_positions is None:
        best_positions = configs[0]
        best_side = required_outer_side(best_positions)
    
    # Final validation
    if not check_validity(best_positions):
        # Try to find a valid config
        for init in configs:
            pos = init.copy()
            for scale in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = pos * scale
                if check_validity(scaled):
                    best_positions = scaled
                    best_side = required_outer_side(scaled)
                    break
            if check_validity(best_positions):
                break
        else:
            # Last resort: spread out linearly
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_side = required_outer_side(best_positions)
    
    # Calculate final exact side
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
