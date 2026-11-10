import numpy as np

def compute_ratio_fast(points):
    """
    Fast computation of A_min / A_hull assuming points are in convex position 
    and ordered by angle.
    """
    n = points.shape[0]
    
    # Check for invalid points
    if not np.all(np.isfinite(points)):
        return -np.inf
    
    # Calculate Convex Hull Area using Shoelace formula
    # Points are assumed to be ordered by angle (which they are, from our parametrization)
    # Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    x = points[:, 0]
    y = points[:, 1]
    
    # Use np.roll for cyclic indices
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    
    cross_terms = x * y_next - x_next * y
    a_hull = 0.5 * np.sum(cross_terms)
    
    # If the area is negative, it means the winding is clockwise, take abs
    if a_hull < 0:
        a_hull = -a_hull
        
    if a_hull <= 1e-10:
        return -np.inf

    # Calculate all triangle areas
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    a_min = np.min(areas)
    
    if a_min <= 1e-10:
        return -np.inf
        
    return a_min / a_hull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    # Angles for regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Initial radii: 1.0 (Regular polygon)
    current_radii = np.ones(n)
    current_points = np.column_stack((current_radii * np.cos(angles), current_radii * np.sin(angles)))
    current_score = compute_ratio_fast(current_points)
    
    best_radii = current_radii.copy()
    best_score = current_score
    best_points = current_points.copy()
    
    # Simulated Annealing Parameters
    T = 0.01  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-6
    max_iter = 50000
    step_size = 0.05
    
    rng = np.random.default_rng(42)
    
    for _ in range(max_iter):
        if T < min_T:
            break
            
        # Generate candidate: perturb one radius
        idx = rng.integers(n)
        delta = rng.normal(0, step_size)
        new_radii = current_radii.copy()
        new_radii[idx] += delta
        
        # Keep radii positive and bounded
        if new_radii[idx] < 0.1 or new_radii[idx] > 2.0:
            continue
            
        new_points = np.column_stack((new_radii * np.cos(angles), new_radii * np.sin(angles)))
        
        # Check if points are still in convex position roughly
        # We can just compute the score. If it drops significantly, it will be rejected.
        new_score = compute_ratio_fast(new_points)
        
        # Acceptance criterion
        delta_score = new_score - current_score
        
        if delta_score > 0 or rng.random() < np.exp(delta_score / T):
            current_radii = new_radii
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_radii = current_radii.copy()
                best_points = current_points.copy()
                
        # Cool down
        T *= cooling_rate
        
    return best_points
