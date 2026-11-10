import numpy as np
from scipy.spatial import ConvexHull
import itertools

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
        except Exception:
            return 0.0
        if a_hull < 1e-10:
            return 0.0
        
        # Vectorized triangle area calculation
        # We need areas of all C(n,3) triangles.
        # Using the formula: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # This can be computed as 0.5 * |det([p1-p3, p2-p3])| for each triplet.
        # Or simpler: use the cross product of differences.
        
        # Let's generate all indices
        indices = np.array(list(itertools.combinations(range(n), 3)))
        p1 = pts[indices[:, 0]]
        p2 = pts[indices[:, 1]]
        p3 = pts[indices[:, 2]]
        
        # Vectorized area
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        # Using p1, p2, p3 arrays of shape (M, 2)
        v1 = p2 - p1
        v2 = p3 - p1
        # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        if min_area < 1e-10:
            return 0.0
        return min_area / a_hull

    best_score = 0.0
    best_points = None
    
    # Simulated Annealing Parameters
    rng = np.random.default_rng(seed=42)
    
    # Number of restarts to escape local minima
    n_restarts = 10
    
    for restart in range(n_restarts):
        # Initialization: Regular polygon is a good start for convexity
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Add some randomness to break symmetry and allow exploration
        if restart == 0:
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            # Jittered regular polygon
            angles_jittered = angles + rng.uniform(-0.2, 0.2, n)
            radii = 1.0 + rng.uniform(-0.2, 0.2, n)
            points = np.column_stack([radii * np.cos(angles_jittered), radii * np.sin(angles_jittered)])
        
        current_score = get_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Simulated Annealing Loop
        T = 0.1
        T_min = 1e-5
        alpha = 0.995 # Cooling rate
        max_iter = 5000
        
        for it in range(max_iter):
            if T < T_min:
                break
            
            # Perturb a random point
            idx = rng.integers(n)
            orig = points[idx].copy()
            # Step size scaled by temperature? Or fixed?
            # Fixed step is safer, temperature controls acceptance.
            step_size = 0.1
            points[idx] += rng.normal(0, step_size, 2)
            
            new_score = get_score(points)
            
            # Acceptance criterion
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # Metropolis acceptance
                # If new_score is 0 (invalid), we should reject strongly unless T is huge
                # But if current_score is also 0, we might want to move.
                if new_score == 0 and current_score == 0:
                    # Both invalid, accept to explore
                    current_score = new_score
                else:
                    # Calculate delta. If new is worse, delta is negative.
                    # We maximize, so delta = new - current
                    delta = new_score - current_score
                    # P = exp(delta / T)
                    # If delta is negative, P < 1
                    # If new_score is 0, delta is -current_score.
                    p_accept = np.exp(delta / T)
                    if rng.random() < p_accept:
                        current_score = new_score
                    else:
                        points[idx] = orig
            
            T *= alpha
            
    return best_points
