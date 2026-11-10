# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexexHull

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _polygon_area(radii: np.ndarray, angles: np.ndarray) -> float:
    """
    Compute the area of a polygon defined by radii at fixed angles.
    Using the shoelace formula on the vertices.
    """
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    # Shoelace formula
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    
    area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    return area

def _min_triangle_area(radii: np.ndarray, angles: np.ndarray) -> float:
    """
    Compute the minimum triangle area formed by any 3 points defined by radii and angles.
    """
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    n = len(radii)
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area.
    
    Strategy:
    1. Parametrize points as a convex polygon using 13 radial distances from the origin.
       This ensures convexity and reduces the search space dimension from 26 to 13.
    2. Use Simulated Annealing to optimize the objective function:
       Score = A_min / A_polygon
       Since the polygon is strictly convex (radii > 0), A_polygon is the convex hull area.
    3. The objective is scale-invariant, so we can fix the scale or just let the optimizer
       find the optimal shape. We normalize radii to have mean 1 to keep them in a reasonable range.
    """
    start_time = time.time()
    deadline = 600  # seconds, with a safety margin
    
    n = 13
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Initial radii: regular polygon
    best_radii = np.ones(n)
    best_score = 0.0
    best_points = None
    
    # Pre-compute triples for faster evaluation
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    def evaluate(radii):
        """
        Compute score for given radii.
        """
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        
        # Compute polygon area (Shoelace)
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        area_hull = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
        
        if area_hull < 1e-10:
            return 0.0, np.zeros((n, 2))
            
        points = np.column_stack((x, y))
        
        # Compute min triangle area
        p0 = points[triples_arr[:, 0]]
        p1 = points[triples_arr[:, 1]]
        p2 = points[triples_arr[:, 2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        if min_area < 1e-10:
            return 0.0, points
            
        score = min_area / area_hull
        return score, points

    # Initial evaluation
    best_score, best_points = evaluate(best_radii)
    
    # Simulated Annealing Parameters
    current_radii = best_radii.copy()
    current_score, current_points = evaluate(current_radii)
    
    T_start = 0.1
    T_end = 0.0001
    cooling_rate = 0.995
    steps = 50000  # Will be adjusted by time limit
    
    for i in range(steps):
        if time.time() - start_time > deadline - 5:
            break
            
        # Generate perturbation
        # Perturb one or two random radii
        num_perturbed = np.random.choice([1, 2], p=[0.7, 0.3])
        indices = np.random.choice(n, num_perturbed, replace=False)
        
        new_radii = current_radii.copy()
        for idx in indices:
            # Gaussian perturbation
            perturbation = np.random.normal(0, 0.1)
            new_radii[idx] += perturbation
            # Ensure radius is positive
            if new_radii[idx] < 0.1:
                new_radii[idx] = 0.1 + np.random.rand() * 0.1
                
        # Normalize to prevent drift (optional, but helps stability)
        # Since score is scale invariant, we can normalize mean radius to 1
        new_radii = new_radii / np.mean(new_radii)
        
        new_score, new_points = evaluate(new_radii)
        
        # Acceptance probability
        if new_score > current_score:
            current_score = new_score
            current_radii = new_radii
            current_points = new_points
            
            if current_score > best_score:
                best_score = current_score
                best_radii = current_radii.copy()
                best_points = current_points.copy()
        else:
            # Calculate delta
            delta = new_score - current_score
            # If delta is 0, accept to allow exploration
            if delta == 0:
                current_radii = new_radii
                current_points = new_points
            else:
                # Boltzmann factor
                # T is current temperature
                T = T_start * (T_end / T_start) ** (i / steps)
                # Avoid division by zero or overflow
                exp_arg = delta / T
                if exp_arg > -700: # np.exp underflow limit
                    accept_prob = np.exp(exp_arg)
                else:
                    accept_prob = 0.0
                    
                if np.random.rand() < accept_prob:
                    current_radii = new_radii
                    current_points = new_points
                    
    # Final fallback if best_points is invalid
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_radii = np.ones(n)
        best_score, best_points = evaluate(best_radii)
        
    return best_points

# EVOLVE-BLOCK-END
