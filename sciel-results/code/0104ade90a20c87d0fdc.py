import numpy as np
import time
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s kill

    # Unit hexagon vertices centered at origin, angle 0
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Normals of outer hexagon (centered at 0,0, rotation 0)
    # Faces are perpendicular to vectors at angles 0, 60, ...
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    def get_all_hex_verts(centers, angles):
        """Get vertices of all hexagons vectorized."""
        n = len(centers)
        all_verts = np.zeros((n, 6, 2))
        for k in range(n):
            angle_rad = np.deg2rad(angles[k])
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[c, -s], [s, c]])
            verts = unit_hex_verts @ R.T
            verts += np.array([centers[k, 0], centers[k, 1]])
            all_verts[k] = verts
        return all_verts

    def calc_side(centers, angles):
        """Calculate required outer hexagon side length (vectorized)."""
        all_verts = get_all_hex_verts(centers, angles)
        # all_verts shape: (N, 6, 2)
        # Project each vertex onto each normal
        # For each hexagon k, compute max projection over all 6 vertices and 6 normals
        max_apothem = 0.0
        for k in range(len(centers)):
            verts = all_verts[k]  # (6, 2)
            # Project onto all normals: (6, 6)
            projs = verts @ normals.T
            max_apothem = max(max_apothem, np.max(projs))
        # Side S = 2 * apothem / sqrt(3)
        return 2.0 * max_apothem / np.sqrt(3)

    def check_non_overlap(centers, angles, min_dist=np.sqrt(3) * 0.995):
        """Check if hexagons are non-overlapping using center distance."""
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    return False
        return True

    def objective(centers, angles):
        """Objective function: side length + penalty for overlaps."""
        side = calc_side(centers, angles)
        
        # Penalty for overlaps
        penalty = 0.0
        min_dist = np.sqrt(3) * 0.995
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 5000
        
        return side + penalty

    d = np.sqrt(3)  # Distance between centers of adjacent hexagons in lattice

    best_side = float('inf')
    best_centers = None
    best_angles = None

    # Simulated annealing
    def simulated_annealing(initial_centers, initial_angles, max_iters=20000, start_temp=1.0, end_temp=0.01):
        """Run simulated annealing to optimize the packing."""
        current_centers = initial_centers.copy()
        current_angles = initial_angles.copy()
        current_obj = objective(current_centers, current_angles)
        
        best_centers = current_centers.copy()
        best_angles = current_angles.copy()
        best_obj = current_obj
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
            
            # Generate perturbation
            idx = np.random.randint(N)
            dx = np.random.normal(0, 0.2)
            dy = np.random.normal(0, 0.2)
            dangle = np.random.normal(0, 3.0)
            
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle
            
            new_obj = objective(new_centers, new_angles)
            
            delta = new_obj - current_obj
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_centers = new_centers
                current_angles = new_angles
                current_obj = new_obj
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_centers = current_centers.copy()
                    best_angles = current_angles.copy()
        
        return best_centers, best_angles, best_obj

    # Generate multiple initial configurations
    np.random.seed(42)
    
    initial_configs = []
    
    # Config 1: Hexagonal cluster centered at origin
    centers1 = []
    centers1.append((0, 0))
    for k in range(6):
        angle = k * np.pi / 3
        centers1.append((d * np.cos(angle), d * np.sin(angle)))
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        if len(centers1) < N:
            centers1.append((x, y))
    
    while len(centers1) < N:
        angle = np.random.uniform(0, 2 * np.pi)
        r = 2.5 * d
        centers1.append((r * np.cos(angle), r * np.sin(angle)))
    
    centers1 = np.array(centers1[:N])
    angles1 = np.zeros(N)
    initial_configs.append((centers1, angles1))
    
    # Config 2: Linear arrangement
    centers2 = np.array([(i * d - (N - 1) * d / 2, 0) for i in range(N)])
    angles2 = np.zeros(N)
    initial_configs.append((centers2, angles2))
    
    # Config 3: Two rows
    centers3 = []
    for i in range(N):
        if i < 6:
            centers3.append((i * d - 2.5 * d, 0))
        else:
            centers3.append(((i - 6) * d - 1.5 * d + d / 2, 1.5))
    centers3 = np.array(centers3)
    angles3 = np.zeros(N)
    initial_configs.append((centers3, angles3))
    
    # Config 4: Random scatter
    centers4 = np.random.uniform(-3 * d, 3 * d, (N, 2))
    angles4 = np.random.uniform(-10, 10, N)
    initial_configs.append((centers4, angles4))
    
    # Config 5: Tighter hexagonal pattern
    centers5 = []
    centers5.append((-d / 2, 0))
    centers5.append((d / 2, 0))
    centers5.append((-d, 1.5))
    centers5.append((0, 1.5))
    centers5.append((d, 1.5))
    centers5.append((-d / 2, 3.0))
    centers5.append((d / 2, 3.0))
    centers5.append((-d, -1.5))
    centers5.append((0, -1.5))
    centers5.append((d, -1.5))
    centers5.append((0, -3.0))
    
    centers5 = np.array(centers5[:N])
    angles5 = np.zeros(N)
    initial_configs.append((centers5, angles5))
    
    # Config 6: Compact triangular arrangement
    centers6 = []
    # Row 0: 3 hexagons
    for i in range(3):
        centers6.append((i * d - d, 0))
    # Row 1: 4 hexagons
    for i in range(4):
        centers6.append((i * d - 1.5 * d + d/2, 1.5))
    # Row 2: 3 hexagons
    for i in range(3):
        centers6.append((i * d - d, 3.0))
    # Row 3: 1 hexagon
    centers6.append((0, 4.5))
    
    centers6 = np.array(centers6[:N])
    angles6 = np.zeros(N)
    initial_configs.append((centers6, angles6))
    
    # Config 7: Another compact arrangement
    centers7 = []
    # Center
    centers7.append((0, 0))
    # 6 surrounding
    for k in range(6):
        angle = k * np.pi / 3
        centers7.append((d * np.cos(angle), d * np.sin(angle)))
    # 4 more at distance ~1.8*d
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 4
        r = 1.8 * d
        centers7.append((r * np.cos(angle), r * np.sin(angle)))
    
    centers7 = np.array(centers7[:N])
    angles7 = np.zeros(N)
    initial_configs.append((centers7, angles7))
    
    # Run simulated annealing from each initial configuration
    for config_idx, (init_centers, init_angles) in enumerate(initial_configs):
        if time.time() > deadline - 8:
            break
        
        try:
            result_centers, result_angles, result_obj = simulated_annealing(
                init_centers, init_angles, max_iters=15000, start_temp=0.5, end_temp=0.001
            )
            
            # Check if the result is valid (no overlaps)
            if check_non_overlap(result_centers, result_angles):
                side = calc_side(result_centers, result_angles)
                if side < best_side:
                    best_side = side
                    best_centers = result_centers.copy()
                    best_angles = result_angles.copy()
        except Exception:
            pass
    
    # Final refinement: high-precision local search on the best solution
    if best_centers is not None and time.time() < deadline - 5:
        # Do a fine-tuning local search with decreasing step size
        current_centers = best_centers.copy()
        current_angles = best_angles.copy()
        current_obj = objective(current_centers, current_angles)
        
        step_size = 0.03
        for it in range(8000):
            if time.time() > deadline - 3:
                break
            
            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dangle = np.random.normal(0, step_size * 8)
            
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle
            
            new_obj = objective(new_centers, new_angles)
            
            if new_obj < current_obj:
                current_centers = new_centers
                current_angles = new_angles
                current_obj = new_obj
            
            # Decay step size
            step_size *= 0.9995
        
        if current_obj < objective(best_centers, best_angles):
            best_centers = current_centers
            best_angles = current_angles
            best_side = calc_side(best_centers, best_angles)
    
    # Construct output
    if best_centers is None:
        # Fallback: widely spaced configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for i in range(N):
        x, y = best_centers[i, 0], best_centers[i, 1]
        ang = best_angles[i]
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (0.0, 0.0), float(best_side), 0.0
