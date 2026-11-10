# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _compute_scores(points):
    """
    Calculate the minimum triangle area and convex hull area.
    Returns (A_min, A_hull).
    """
    n = len(points)
    # Generate all unique triplets
    indices = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)])
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas
    areas = 0.5 * np.abs(p1[:, 0]*(p2[:, 1] - p3[:, 1]) + 
                         p2[:, 0]*(p3[:, 1] - p1[:, 1]) + 
                         p3[:, 0]*(p1[:, 1] - p2[:, 1]))
    
    a_min = np.min(areas)
    
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        a_hull = 0.0
        
    return a_min, a_hull

def _objective_ratio(points):
    """
    Calculate the ratio A_min / A_hull.
    """
    a_min, a_hull = _compute_scores(points)
    if a_hull < 1e-10 or a_min < 1e-10:
        return -1e10
    return a_min / a_hull

def _local_optimize(points):
    """
    Refine points using Nelder-Mead.
    """
    def _nm_obj(params):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        rest = params.reshape(-1, 2)
        pts = np.vstack((p0, p1, rest))
        # Normalize to keep scale consistent? No, let NM handle it, but we fixed p0, p1.
        # Actually, for NM, it's better to optimize all 13 points but constrain p0, p1?
        # The current seed fixes p0, p1. I will stick to that convention to be safe, 
        # but allow the rest to move freely.
        ratio = _objective_ratio(pts)
        return -ratio

    # Fix p0=(0,0) and p1=(1,0) to remove translational/rotational/scale invariance
    # This reduces dimensionality from 26 to 22.
    initial_params = points[2:].flatten()
    
    result = minimize(_nm_obj, initial_params, method='Nelder-Mead', 
                      options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 500})
    
    if result.fun < -1e-9: # Valid improvement
        optimal_rest = result.x.reshape(-1, 2)
        pts = np.vstack((np.array([0.0, 0.0]), np.array([1.0, 0.0]), optimal_rest))
        return pts
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the minimum
    triangle area to the convex hull area using Simulated Annealing.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Run multiple restarts to ensure we find a good global optimum
    # Each restart uses a different random seed and initialization
    num_restarts = 5
    
    for restart in range(num_restarts):
        # Initialize points
        if restart == 0:
            # Start with regular polygon for the first restart
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            x = np.cos(angles)
            y = np.sin(angles)
            points = np.column_stack((x, y))
        else:
            # Random initialization within a unit square/box
            points = np.random.rand(n, 2) * 2 - 1
        
        # Normalize initial points to have p0=(0,0) and p1=(1,0)
        # This ensures the configuration is well-scaled for optimization
        p0_idx = 0
        p1_idx = 1
        v0 = points[p0_idx]
        v1 = points[p1_idx]
        
        # Translate
        points = points - v0
        # Scale and Rotate
        d = np.linalg.norm(v1 - v0)
        if d < 1e-10:
            # Degenerate, re-randomize
            points = np.random.rand(n, 2)
            continue
            
        points = points / d
        
        # Rotate so v1 is along x-axis
        angle = np.arctan2(points[p1_idx, 1], points[p1_idx, 0])
        rot_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])
        points = points @ rot_matrix.T
        
        # Ensure p0 is (0,0) and p1 is (1,0) explicitly to avoid numerical drift
        points[p0_idx] = [0.0, 0.0]
        points[p1_idx] = [1.0, 0.0]
        
        current_score = _objective_ratio(points)
        
        # Simulated Annealing
        T = 0.05 # Initial temperature
        cooling = 0.995 # Cooling rate
        steps = 2000 # Number of SA steps
        current_points = points.copy()
        
        for step in range(steps):
            # Pick a random point (excluding 0 and 1) to move
            move_idx = np.random.randint(2, n)
            
            # Gaussian perturbation
            sigma = T * 0.1
            perturbation = np.random.normal(0, sigma, 2)
            
            new_points = current_points.copy()
            new_points[move_idx] += perturbation
            
            new_score = _objective_ratio(new_points)
            
            # Acceptance criterion
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
                current_points = new_points
                current_score = new_score
            
            T *= cooling
            
        # Local refinement
        refined_points = _local_optimize(current_points)
        final_score = _objective_ratio(refined_points)
        
        if final_score > best_score:
            best_score = final_score
            best_points = refined_points

    # Final safety check
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    # One final polish
    best_points = _local_optimize(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
