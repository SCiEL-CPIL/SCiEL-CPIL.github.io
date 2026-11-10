import numpy as np
import time

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Minimizes the number of convex 7-gons among 33 points in the unit square.
    Uses Simulated Annealing to minimize the number of convex 4-gons as a proxy,
    starting from a 'cup-cap' initialization.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 550.0  # Leave 50s margin
    
    def count_convex_quads(points):
        """
        Counts the number of convex 4-gons in the set of points.
        A 4-point set is convex if no point lies inside the triangle formed by the other 3.
        Equivalently, if sorted by angle around the centroid, the turns are all in the same direction.
        Faster check: For 4 points, sort by x, then check if the diagonal intersection or similar.
        Simplest robust check: 
        1. Compute convex hull of the 4 points. If hull size is 4, it's convex.
        """
        count = 0
        n_pts = len(points)
        # Pre-compute? No, just loop. 40920 iterations is fine.
        for i in range(n_pts - 3):
            for j in range(i + 1, n_pts - 2):
                for k in range(j + 1, n_pts - 1):
                    for l in range(k + 1, n_pts):
                        p = points[[i, j, k, l]]
                        # Check if convex
                        # Calculate cross products of consecutive edges in sorted order?
                        # Easier: Check if any point is inside the triangle of the other 3.
                        # If none is inside, it's convex.
                        # Triangle 012, check 3.
                        # Triangle 013, check 2.
                        # Triangle 023, check 1.
                        # Triangle 123, check 0.
                        # If any point is strictly inside, not convex.
                        # If on boundary, not in general position (handled by init).
                        
                        # Helper: area sign
                        def cross(a, b, c):
                            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                        
                        c012 = cross(p[0], p[1], p[2])
                        c013 = cross(p[0], p[1], p[3])
                        c023 = cross(p[0], p[2], p[3])
                        c123 = cross(p[1], p[2], p[3])
                        
                        # If all cross products have the same sign (and non-zero), it's convex.
                        # If signs differ, one point is inside.
                        # Since general position, no zeros.
                        signs = [1 if c > 0 else -1 for c in [c012, c013, c023, c123]]
                        if len(set(signs)) == 1:
                            count += 1
        return count

    # Initialization: Cup-Cap
    # 16 points on lower arc, 17 on upper arc
    # Use sine/cosine for smooth curves
    n_lower = 16
    n_upper = 17
    
    # Lower arc: y = 0.1 + 0.1 * sin(pi * x)
    x_lower = np.linspace(0.1, 0.9, n_lower)
    y_lower = 0.1 + 0.1 * np.sin(np.pi * x_lower)
    
    # Upper arc: y = 0.9 - 0.1 * sin(pi * x)
    x_upper = np.linspace(0.1, 0.9, n_upper)
    y_upper = 0.9 - 0.1 * np.sin(np.pi * x_upper)
    
    points = np.vstack([
        np.column_stack([x_lower, y_lower]),
        np.column_stack([x_upper, y_upper])
    ])
    
    # Add small noise to ensure general position and avoid symmetry issues
    np.random.seed(42)
    points += np.random.normal(0, 0.001, points.shape)
    
    # Clip to [0, 1]
    points = np.clip(points, 0.0, 1.0)
    
    # Ensure distinctness and non-collinearity
    # The noise should handle this.
    
    best_points = points.copy()
    best_score = count_convex_quads(best_points)
    
    # Simulated Annealing
    T = 0.05
    T_min = 0.001
    alpha = 0.995
    step_size = 0.05
    
    iteration = 0
    while time.time() < deadline:
        # Perturb a random point
        idx = np.random.randint(n)
        new_points = best_points.copy()
        new_points[idx] += np.random.normal(0, step_size, 2)
        
        # Clip to [0,1]
        new_points[idx] = np.clip(new_points[idx], 0.0, 1.0)
        
        # Check validity (distinctness)
        # If points are too close, reject
        min_dist_sq = np.min(np.sum((new_points - new_points[idx])**2, axis=1))
        # Exclude self
        min_dist_sq = np.min(np.delete(np.sum((new_points - new_points[idx])**2, axis=1), idx))
        if min_dist_sq < 1e-6:
            T *= alpha
            continue
            
        new_score = count_convex_quads(new_points)
        
        if new_score < best_score:
            best_points = new_points
            best_score = new_score
        else:
            delta = new_score - best_score
            if np.random.rand() < np.exp(-delta / T):
                best_points = new_points
                best_score = new_score
                
        T *= alpha
        iteration += 1
        
        # Adaptive step size: if stuck, increase step size? 
        # Or just let it cool.
        
    return best_points
