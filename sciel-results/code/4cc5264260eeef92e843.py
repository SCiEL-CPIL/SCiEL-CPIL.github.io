import numpy as np
import time
from scipy.spatial import ConvexHull

def get_score(points):
    """
    Calculate the score: minimum triangle area divided by convex hull area.
    """
    N = len(points)
    if N < 3:
        return 0.0

    # Check for degeneracy
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area calculation
    # For N=13, M=286 triangles.
    idx = np.indices((N, N, N))
    i, j, k = idx
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    dx_ji = p_j[:, 0] - p_i[:, 0]
    dy_ji = p_j[:, 1] - p_i[:, 1]
    dx_ki = p_k[:, 0] - p_i[:, 0]
    dy_ki = p_k[:, 1] - p_i[:, 1]
    
    cross = dx_ji * dy_ki - dy_ji * dx_ki
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin from 150s limit
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # 1. Baseline: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = get_score(points_reg)
    best_points = points_reg.copy()
    
    # 2. Simulated Annealing Global Search
    # Strategy:
    # - Start from a slightly perturbed regular polygon.
    # - Objective: Maximize min_triangle_area / hull_area.
    # - Penalty: If the points are not in convex position (i.e., ConvexHull has < N vertices),
    #   we apply a penalty. However, in SA, we can allow temporary violations to escape local minima.
    #   Actually, for this specific problem, the optimal configuration is known to be strictly convex.
    #   So we can enforce convexity via a barrier or simply reject moves that break it if we are close to the end.
    #   A better approach: Define the score as min_area / hull_area. If the hull shrinks (points go inside), 
    #   the denominator shrinks, but the min_area might shrink faster. 
    #   Let's use a robust score function that handles non-convex sets gracefully (it will just compute the hull of the subset).
    #   But we want to encourage convexity.
    #   
    #   Let's define the objective for SA:
    #   f(points) = min_triangle_area / hull_area
    #   We want to maximize f.
    #   
    #   Initial T: 1.0
    #   Cooling: T = T * 0.999
    #   Steps: ~50,000 steps (fast enough in Python with NumPy vectorization)
    
    current_points = points_reg + np.random.normal(0, 0.05, (N, 2))
    current_score = get_score(current_points)
    
    if current_score > best_score:
        best_score = current_score
        best_points = current_points.copy()
    
    T = 0.1  # Initial temperature
    cooling = 0.9995
    step_size = 0.1
    
    # Precompute indices for speed? No, get_score is already vectorized.
    # To speed up SA, we can approximate the score or use a cheaper metric?
    # get_score is ~0.1ms per call? 50k calls * 0.1ms = 5s. This is feasible.
    
    steps = 0
    while time.time() < deadline - 10:
        # Generate a random move
        # Pick a random point and a random direction
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        new_score = get_score(new_points)
        
        # Acceptance criterion
        delta_e = new_score - current_score
        if delta_e > 0 or np.random.rand() < np.exp(delta_e / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        # Cool down
        T *= cooling
        steps += 1
        
        # Adaptive step size: if we accept too many, increase step size? 
        # Or just keep it constant. Constant is simpler and robust.
        
    # 3. Local Refinement of the Best Point
    # Use Nelder-Mead for final polish
    if best_points is not None:
        def neg_score_2d(x):
            pts = x.reshape(N, 2)
            return -get_score(pts)
        
        x0 = best_points.flatten()
        
        # Run a few quick Nelder-Mead restarts from this improved base
        for i in range(5):
            if time.time() > deadline - 5:
                break
            try:
                noise = np.random.normal(0, 0.01, N * 2)
                x_start = x0 + noise
                from scipy.optimize import minimize
                res = minimize(neg_score_2d, x_start, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(N, 2).copy()
                    x0 = best_points.flatten()
            except Exception:
                pass

    return best_points
