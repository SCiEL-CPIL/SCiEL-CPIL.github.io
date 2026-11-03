import numpy as np
import time
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

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

    def get_hex_verts(x, y, angle_deg):
        """Get vertices of hexagon at (x, y) with given rotation."""
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = unit_hex_verts @ R.T
        verts += np.array([x, y])
        return verts

    def calc_side(centers, angles):
        """Calculate required outer hexagon side length."""
        max_apothem = 0.0
        for k in range(N):
            verts = get_hex_verts(centers[k, 0], centers[k, 1], angles[k])
            # Project all vertices onto all normals
            for v in verts:
                for n in normals:
                    proj = np.dot(v, n)
                    if proj > max_apothem:
                        max_apothem = proj
        # Side S = 2 * apothem / sqrt(3)
        return 2.0 * max_apothem / np.sqrt(3)

    def check_non_overlap(centers, angles, min_dist_threshold=np.sqrt(3) * 0.95):
        """Check if hexagons are non-overlapping using distance heuristic."""
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist_threshold:
                    return False
        return True

    # Generate candidate configurations using hexagonal lattice
    d = np.sqrt(3)  # Distance between centers of adjacent hexagons in lattice

    # Strategy: Use hexagonal lattice coordinates
    # In a hexagonal lattice, centers are at positions:
    # (i * d + j * d/2, j * 1.5) for integers i, j
    # But we need to be careful about the exact geometry

    # Let's use axial coordinates for hexagonal grid
    # Position in Cartesian: x = d * (i + j/2), y = 1.5 * j
    # Actually, for unit hexagons (side=1), the distance between centers
    # of adjacent hexagons sharing an edge is sqrt(3)

    # Let me use a simpler approach: generate candidate positions on a grid
    # and try different subsets

    # Hexagonal lattice positions within a reasonable range
    # Centers at (x, y) where x = a * sqrt(3) + b * sqrt(3)/2, y = b * 1.5
    # for integers a, b

    candidate_positions = []
    for b in range(-3, 4):  # y coordinate index
        for a in range(-3, 4):  # x coordinate index
            x = a * d + b * d / 2
            y = b * 1.5
            candidate_positions.append((x, y))

    # Convert to numpy array
    candidate_positions = np.array(candidate_positions)
    n_candidates = len(candidate_positions)

    # Precompute pairwise distances
    # dist_matrix[i, j] = distance between candidate i and candidate j
    dist_matrix = np.zeros((n_candidates, n_candidates))
    for i in range(n_candidates):
        for j in range(i + 1, n_candidates):
            dx = candidate_positions[i, 0] - candidate_positions[j, 0]
            dy = candidate_positions[i, 1] - candidate_positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Find all valid subsets of N positions that are non-overlapping
    # Use backtracking with pruning

    best_side = float('inf')
    best_centers = None
    best_angles = None

    # For efficiency, use a greedy approach with local search
    # First, try to find a good configuration using simulated annealing

    def objective(centers, angles):
        """Objective function: side length + penalty for overlaps."""
        side = calc_side(centers, angles)
        
        # Penalty for overlaps
        penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 1000
        
        return side + penalty

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
            if time.time() > deadline - 2:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
            
            # Generate perturbation
            idx = np.random.randint(N)
            dx = np.random.normal(0, 0.3)
            dy = np.random.normal(0, 0.3)
            dangle = np.random.normal(0, 5.0)
            
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
    # 7 hexagons in a flower pattern + 4 more
    centers1 = []
    # Center
    centers1.append((0, 0))
    # 6 surrounding
    for k in range(6):
        angle = k * np.pi / 3
        centers1.append((d * np.cos(angle), d * np.sin(angle)))
    # 4 more: extend in some directions
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        # Place at distance 2*d in these directions
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        if len(centers1) < N:
            centers1.append((x, y))
    
    while len(centers1) < N:
        # Add more in a ring
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
    
    # Config 5: Tighter hexagonal with some rotations
    centers5 = []
    # Try a 3-4-3 or 2-3-2-3-1 pattern
    # Row 0: 2 hexagons
    centers5.append((-d / 2, 0))
    centers5.append((d / 2, 0))
    # Row 1: 3 hexagons
    centers5.append((-d, 1.5))
    centers5.append((0, 1.5))
    centers5.append((d, 1.5))
    # Row 2: 2 hexagons
    centers5.append((-d / 2, 3.0))
    centers5.append((d / 2, 3.0))
    # Row 3: 3 hexagons
    centers5.append((-d, -1.5))
    centers5.append((0, -1.5))
    centers5.append((d, -1.5))
    # Row 4: 1 hexagon
    centers5.append((0, -3.0))
    
    centers5 = np.array(centers5[:N])
    angles5 = np.zeros(N)
    initial_configs.append((centers5, angles5))
    
    # Run simulated annealing from each initial configuration
    for config_idx, (init_centers, init_angles) in enumerate(initial_configs):
        if time.time() > deadline - 5:
            break
        
        try:
            result_centers, result_angles, result_obj = simulated_annealing(
                init_centers, init_angles, max_iters=15000, start_temp=0.5, end_temp=0.001
            )
            
            # Check if the result is valid (no overlaps)
            if check_non_overlap(result_centers, result_angles, min_dist_threshold=np.sqrt(3) * 0.98):
                side = calc_side(result_centers, result_angles)
                if side < best_side:
                    best_side = side
                    best_centers = result_centers.copy()
                    best_angles = result_angles.copy()
        except Exception:
            pass
    
    # If we haven't found a good solution, try more aggressive search
    if best_side > 4.0 or best_centers is None:
        # Try a different approach: start from a known good packing and perturb
        # Known: 7 hexagons in a flower fit in a hexagon of side 3
        # Let's try to extend this
        
        # Use a grid of candidate positions and do a combinatorial search
        # for the best subset of N positions
        
        # Sort candidate positions by distance from origin
        dists = np.linalg.norm(candidate_positions, axis=1)
        sorted_indices = np.argsort(dists)
        
        # Try subsets of the closest positions
        for start_idx in range(0, min(50, n_candidates - N + 1), 2):
            if time.time() > deadline - 3:
                break
            
            # Take N positions from a window of candidates
            window_start = sorted_indices[start_idx]
            window_end = min(window_start + N + 10, n_candidates)
            candidate_subset = sorted_indices[window_start:window_end]
            
            if len(candidate_subset) < N:
                continue
            
            # Try all combinations of N from this subset (if small enough)
            if len(candidate_subset) <= 15:
                for combo in itertools.combinations(range(len(candidate_subset)), N):
                    indices = candidate_subset[list(combo)]
                    centers = candidate_positions[indices]
                    angles = np.zeros(N)
                    
                    if not check_non_overlap(centers, angles):
                        continue
                    
                    side = calc_side(centers, angles)
                    if side < best_side:
                        best_side = side
                        best_centers = centers.copy()
                        best_angles = angles.copy()
            
            else:
                # Use greedy: pick N positions that minimize the bounding hexagon
                # Start with the closest N
                selected = list(candidate_subset[:N])
                best_subset_obj = objective(candidate_positions[selected], np.zeros(N))
                
                # Try swapping
                for _ in range(100):
                    if time.time() > deadline - 3:
                        break
                    
                    # Random swap
                    i = np.random.randint(N)
                    j = np.random.randint(len(candidate_subset))
                    if candidate_subset[j] in selected:
                        continue
                    
                    new_selected = selected.copy()
                    new_selected[i] = candidate_subset[j]
                    new_obj = objective(candidate_positions[new_selected], np.zeros(N))
                    
                    if new_obj < best_subset_obj:
                        selected = new_selected
                        best_subset_obj = new_obj
                
                centers = candidate_positions[selected]
                angles = np.zeros(N)
                if check_non_overlap(centers, angles):
                    side = calc_side(centers, angles)
                    if side < best_side:
                        best_side = side
                        best_centers = centers.copy()
                        best_angles = angles.copy()
    
    # Final refinement: local search on the best solution
    if best_centers is not None and time.time() < deadline - 5:
        # Do a fine-tuning local search
        current_centers = best_centers.copy()
        current_angles = best_angles.copy()
        current_obj = objective(current_centers, current_angles)
        
        step_size = 0.05
        for it in range(5000):
            if time.time() > deadline - 2:
                break
            
            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dangle = np.random.normal(0, step_size * 10)
            
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
            step_size *= 0.999
        
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
