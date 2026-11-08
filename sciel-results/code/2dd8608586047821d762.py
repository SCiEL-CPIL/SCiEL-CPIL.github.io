import numpy as np
import time
import itertools

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2  # height of unit base equilateral triangle

def vectorized_min_triangle_area(points):
    """
    Compute the minimum area of triangles formed by all triplets of points.
    points: shape (N, 2)
    Returns: min area (float)
    """
    # points[:, np.newaxis, :] has shape (N, 1, 2)
    # points[np.newaxis, :, :] has shape (1, N, 2)
    # diff = p_i - p_j, shape (N, N, 2)
    p1 = points[:, np.newaxis, :]
    p2 = points[np.newaxis, :, :]
    diff = p1 - p2
    
    # Cross product in 2D: (x1*y2 - x2*y1)
    # We want area of triangle (i, j, k).
    # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # Let's compute all pairwise cross products first? No, that's not quite right for triplets.
    
    # Efficient way:
    # For each pair (i, j), the vector is v_ij = p_j - p_i.
    # The area of triangle (i, j, k) is 0.5 * | v_ij x v_ik |.
    # v_ij x v_ik = (p_j - p_i) x (p_k - p_i)
    
    # Let's use the determinant formula directly for all triplets.
    # This requires iterating or clever broadcasting.
    # Broadcasting (N, N, N) is 11^3 = 1331 elements, which is very small.
    
    # p_i, p_j, p_k
    # Shape (N, 1, 1, 2), (1, N, 1, 2), (1, 1, N, 2)
    pi = points[:, np.newaxis, np.newaxis, :]
    pj = points[np.newaxis, :, np.newaxis, :]
    pk = points[np.newaxis, np.newaxis, :, :]
    
    # Vectors from i to j and i to k
    v_ij = pj - pi  # (N, N, 1, 2)
    v_ik = pk - pi  # (N, 1, N, 2)
    
    # Cross product (z-component)
    # (v_ij_x * v_ik_y - v_ij_y * v_ik_x)
    # v_ij is (N, N, 1, 2) -> we need to broadcast v_ik to (N, N, N, 2)
    # Actually, v_ij has shape (N, N, 1, 2). v_ik has shape (N, 1, N, 2).
    # Broadcasting them together results in shape (N, N, N, 2).
    
    cross_z = v_ij[..., 0] * v_ik[..., 1] - v_ij[..., 1] * v_ik[..., 0]
    
    # Areas
    areas = 0.5 * np.abs(cross_z)
    
    # We only want unique triplets i < j < k to avoid counting duplicates and self-triangles (area 0)
    # But taking min of the whole array including zeros will give 0.
    # We must exclude cases where indices are equal.
    
    # Create a mask for i < j < k
    idx = np.arange(N)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
        
    return np.min(valid_areas)

def is_inside_triangle_vectorized(points):
    """
    Check if all points are inside the triangle.
    Returns True if all are inside, False otherwise.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Conditions:
    # y >= 0
    # y <= H
    # x >= y / SQRT3
    # x <= 1 - y / SQRT3
    
    cond1 = y >= 0
    cond2 = y <= H
    cond3 = x >= y / SQRT3
    cond4 = x <= 1 - y / SQRT3
    
    return np.all(cond1 & cond2 & cond3 & cond4)

def generate_initial_guess(seed=0):
    """
    Generate a structured initial guess based on triangular lattice rows.
    """
    rng = np.random.RandomState(seed)
    
    # Rows of points: 4, 3, 2, 1, 1 (Total 11)
    # Or 3, 3, 3, 1, 1?
    # Let's try to pack them somewhat evenly.
    # A common heuristic is to start with a grid.
    
    points = []
    
    # Row 0 (bottom): 4 points
    y0 = 0.1
    x_min0 = y0 / SQRT3
    x_max0 = 1 - y0 / SQRT3
    for i in range(4):
        t = (i + 0.5) / 4
        x = x_min0 + t * (x_max0 - x_min0)
        points.append([x, y0])
        
    # Row 1: 3 points
    y1 = 0.35
    x_min1 = y1 / SQRT3
    x_max1 = 1 - y1 / SQRT3
    for i in range(3):
        t = (i + 0.5) / 3
        x = x_min1 + t * (x_max1 - x_min1)
        points.append([x, y1])
        
    # Row 2: 2 points
    y2 = 0.6
    x_min2 = y2 / SQRT3
    x_max2 = 1 - y2 / SQRT3
    for i in range(2):
        t = (i + 0.5) / 2
        x = x_min2 + t * (x_max2 - x_min2)
        points.append([x, y2])
        
    # Row 3: 1 point
    y3 = 0.8
    x3 = 0.5
    points.append([x3, y3])
    
    # Row 4: 1 point
    y4 = 0.9
    x4 = 0.5
    points.append([x4, y4])
    
    pts = np.array(points)
    
    # Add small random perturbation to break symmetry
    pts += rng.normal(0, 0.01, pts.shape)
    
    return pts

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    best_points = None
    best_value = -1.0
    
    # Simulated Annealing parameters
    T_start = 0.05
    T_end = 0.001
    cooling_rate = 0.995
    T = T_start
    
    # Initial guess
    current_points = generate_initial_guess(0)
    current_value = vectorized_min_triangle_area(current_points)
    if is_inside_triangle_vectorized(current_points):
        best_points = current_points.copy()
        best_value = current_value
    else:
        # If initial guess is invalid, project back or use random
        # For simplicity, let's just try random until we get a valid one or use a very conservative grid
        current_points = generate_initial_guess(1)
        current_value = vectorized_min_triangle_area(current_points)
        if is_inside_triangle_vectorized(current_points):
            best_points = current_points.copy()
            best_value = current_value
        else:
             # Fallback to a very safe grid
             pts = []
             for i in range(4):
                 pts.append([0.1 + i*0.2, 0.1])
             for i in range(3):
                 pts.append([0.2 + i*0.2, 0.3])
             for i in range(2):
                 pts.append([0.3 + i*0.2, 0.5])
             pts.append([0.5, 0.7])
             pts.append([0.5, 0.8])
             current_points = np.array(pts)
             current_value = vectorized_min_triangle_area(current_points)
             best_points = current_points.copy()
             best_value = current_value

    rng = np.random.RandomState(42)
    
    steps = 0
    while time.time() - start_time < deadline:
        # Generate a neighbor
        # Move a random point by a small amount
        idx = rng.randint(0, N)
        step_size = T * 0.1  # Scale step with temperature
        delta = rng.normal(0, step_size, 2)
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Check validity
        if not is_inside_triangle_vectorized(new_points):
            # If invalid, reject (or penalize, but rejection is simpler for hard constraints)
            T *= cooling_rate
            continue
            
        new_value = vectorized_min_triangle_area(new_points)
        
        # Acceptance criterion
        if new_value > current_value:
            current_points = new_points
            current_value = new_value
            if current_value > best_value:
                best_value = current_value
                best_points = new_points.copy()
        else:
            # Accept with probability exp((new - old) / T)
            # Since we are maximizing, delta is negative.
            delta_val = new_value - current_value
            if delta_val < 0:
                prob = np.exp(delta_val / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_value = new_value
            # else reject
            
        T *= cooling_rate
        if T < T_end:
            # Restart with a slightly perturbed best solution or a new random one
            if rng.random() < 0.5:
                # Perturb best
                current_points = best_points + rng.normal(0, 0.05, best_points.shape)
                if not is_inside_triangle_vectorized(current_points):
                    # Project back to inside? Simplest is to resample
                    current_points = generate_initial_guess(rng.randint(0, 100))
                    if not is_inside_triangle_vectorized(current_points):
                         # Force valid
                         current_points = np.clip(current_points, 0, 1) # Rough clip, might need better projection
                         # For now, just accept the risk or use the safe grid
                         current_points = best_points
                current_value = vectorized_min_triangle_area(current_points)
            else:
                # New random start
                current_points = generate_initial_guess(rng.randint(0, 100))
                if not is_inside_triangle_vectorized(current_points):
                    current_points = best_points
                    current_value = best_value
                else:
                    current_value = vectorized_min_triangle_area(current_points)
                    if current_value > best_value:
                        best_value = current_value
                        best_points = current_points.copy()
            
            T = T_start
            
        steps += 1
        # Throttle if needed, but vectorized op is fast. 
        # 1000 steps/sec is easy. We can do millions.
        
    return best_points

