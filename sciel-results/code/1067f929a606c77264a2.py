import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 such triples
_triples = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_triples = np.array(_triples)
_N_TRIPLES = len(_triples)

def _min_triangle_area_bary(bary_pts):
    """
    Calculate the minimum area of triangles formed by points given in barycentric coordinates.
    bary_pts: array of shape (N, 3) where each row sums to 1.
    Returns the minimum area.
    """
    # Convert barycentric to Cartesian for area calculation
    # x = v + 0.5*w
    # y = H * w
    x = bary_pts[:, 1] + 0.5 * bary_pts[:, 2]
    y = H * bary_pts[:, 2]
    
    pi = x[_triples[:, 0]]
    pj = x[_triples[:, 1]]
    pk = x[_triples[:, 2]]
    
    qi = y[_triples[:, 0]]
    qj = y[_triples[:, 1]]
    qk = y[_triples[:, 2]]
    
    # Area = 0.5 * | (xi - xk)(yj - yk) - (yi - yk)(xj - xk) |
    dx1 = pi - pk
    dy1 = qi - qk
    dx2 = pj - pk
    dy2 = qj - qk
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _bary_to_cart(bary_pts):
    """Convert barycentric coordinates to Cartesian coordinates."""
    x = bary_pts[:, 1] + 0.5 * bary_pts[:, 2]
    y = H * bary_pts[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary_pts):
    """
    Project barycentric points back to valid barycentric coordinates.
    Ensures u, v, w >= 0 and u+v+w=1.
    """
    # Clip negative values
    bary_pts = np.clip(bary_pts, 0.0, None)
    # Normalize to sum to 1
    sums = np.sum(bary_pts, axis=1, keepdims=True)
    # Avoid division by zero (should not happen if we start valid and perturb small)
    sums = np.where(sums == 0, 1.0, sums)
    bary_pts = bary_pts / sums
    return bary_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def _is_time_up():
        return time.time() > deadline

    # Initialize with a reasonable configuration
    # Start with a regular grid or random points
    # Let's start with points distributed somewhat evenly
    # Using a Dirichlet distribution for random valid barycentric points
    rng = np.random.default_rng(42)
    
    # Initial guess: 11 points
    # We can try to initialize with a pattern that is known to be decent
    # e.g., vertices, midpoints, centroid, etc., plus some randoms
    # Or just random Dirichlet
    init_bary = rng.dirichlet([1, 1, 1], size=N)
    
    current_bary = init_bary.copy()
    current_area = _min_triangle_area_bary(current_bary)
    
    best_bary = current_bary.copy()
    best_area = current_area

    # Simulated Annealing Parameters
    # We want to maximize area, so minimize -area
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    
    # Perturbation scale
    # Barycentric coordinates are in [0,1]. A step of 0.05 is reasonable.
    step_scale = 0.05

    # Main Loop
    while not _is_time_up():
        # Generate a new candidate by perturbing the current solution
        # Add Gaussian noise
        noise = rng.normal(0, step_scale, size=(N, 3))
        new_bary = current_bary + noise
        
        # Project back to valid barycentric space
        new_bary = _project_bary(new_bary)
        
        # Calculate new area
        new_area = _min_triangle_area_bary(new_bary)
        
        # Simulated Annealing acceptance criterion
        # We want to maximize area
        delta_area = new_area - current_area
        
        if delta_area > 0:
            # Better solution, always accept
            current_bary = new_bary
            current_area = new_area
            
            if current_area > best_area:
                best_area = current_area
                best_bary = current_bary.copy()
        else:
            # Worse solution, accept with probability exp(delta/T)
            # delta_area is negative
            prob = np.exp(delta_area / T)
            if rng.random() < prob:
                current_bary = new_bary
                current_area = new_area
        
        # Cool down
        T *= cooling_rate
        
        # If T is too low, restart with a higher T to escape local optima (Simulated Annealing with restarts)
        if T < T_min:
            T = 0.01
            # Optionally, we could also randomize the current state slightly or keep it
            # Keeping it allows us to refine, but restarting T allows escaping.
            # Let's also occasionally jump to a new random region if stuck
            if rng.random() < 0.1:
                current_bary = rng.dirichlet([1, 1, 1], size=N)
                current_area = _min_triangle_area_bary(current_bary)
                # Update best if this random jump was good (unlikely but possible)
                if current_area > best_area:
                    best_area = current_area
                    best_bary = current_bary.copy()

    # Convert best barycentric coordinates to Cartesian
    best_pts = _bary_to_cart(best_bary)
    
    # Final check to ensure validity (should be guaranteed by projection, but safe to check)
    # The problem asks for points on or inside the unit-area triangle.
    # Our barycentric projection ensures this.
    
    return best_pts

