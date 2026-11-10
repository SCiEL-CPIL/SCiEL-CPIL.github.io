# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def heilbronn_convex13():
    """
    Maximize the ratio of the smallest triangle area to the convex hull area
    for 13 points using Differential Evolution.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    # Precompute triangle indices
    tri_indices = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tri = len(tri_indices)

    def calc_areas(points):
        """
        Calculate areas of all triangles defined by tri_indices.
        points: (n, 2)
        returns: (n_tri,)
        """
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return np.abs(cross) * 0.5

    def calc_hull_area(points):
        """
        Calculate the area of the convex hull of the points.
        """
        try:
            # Check for collinearity or degeneracy
            if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
                return 0.0
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def objective_flat(flat_points):
        """
        Objective function for minimization (negative of the score).
        Input: flat array of shape (2*n,)
        """
        points = flat_points.reshape(n, 2)
        
        # Check for non-finite values
        if not np.all(np.isfinite(points)):
            return 1e10

        areas = calc_areas(points)
        min_area = np.min(areas)
        
        if min_area < 1e-9:
            return 1e10

        hull_area = calc_hull_area(points)
        if hull_area < 1e-9:
            return 1e10

        score = min_area / hull_area
        return -score

    # Differential Evolution Parameters
    pop_size = 60
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover rate
    max_gens = 1000
    
    # Bounds for each coordinate
    bounds = [(0.0, 1.0) for _ in range(2 * n)]
    bounds_np = np.array(bounds)
    low = bounds_np[:, 0]
    high = bounds_np[:, 1]

    # Initialize population
    # Start with a mix of random and structured initializations
    pop = np.random.uniform(low, high, (pop_size, 2 * n))
    
    # Add some structured seeds to the population
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_poly = np.column_stack([0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)])
    pop[0] = reg_poly.flatten()
    
    # 2. Hexagonal grid (subset)
    s = 0.25
    hex_pts = []
    for row in range(-2, 3):
        for col in range(-2, 3):
            x = 0.5 + col * s + (row % 2) * s / 2
            y = 0.5 + row * s * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    mask = (hex_pts[:, 0] - 0.5)**2 + (hex_pts[:, 1] - 0.5)**2 <= 0.45**2
    hex_pts = hex_pts[mask]
    if len(hex_pts) >= n:
        # Select n points that are most spread out or just first n
        # For simplicity, take a random subset if too many, or just first n
        idx = np.random.choice(len(hex_pts), n, replace=False)
        pop[1] = hex_pts[idx].flatten()

    # Evaluate initial population
    scores = np.array([objective_flat(p) for p in pop])
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_point = pop[best_idx].copy()

    # Main DE Loop
    gen = 0
    while gen < max_gens:
        if time.time() > deadline - 10:
            break
            
        for i in range(pop_size):
            if time.time() > deadline - 10:
                break
            
            # Select 3 distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            while r1 == r2 or r1 == r3 or r2 == r3:
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = np.empty_like(mutant)
            j_rand = np.random.randint(0, 2 * n)
            for j in range(2 * n):
                if np.random.rand() <= CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]
            
            # Clip to bounds
            trial = np.clip(trial, low, high)
            
            # Evaluate
            trial_score = objective_flat(trial)
            
            # Selection
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_point = trial.copy()
        
        gen += 1
    
    # Convert best_point back to (n, 2)
    best_points = best_point.reshape(n, 2)
    
    # Local Refinement using Nelder-Mead
    # We use a simple coordinate descent or Nelder-Mead for final polish
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            
            # Use Nelder-Mead on the best point found
            # We need to handle the bounds carefully. 
            # Since DE already kept it in bounds, small steps should be safe.
            result = minimize(
                objective_flat, 
                best_point, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if result.fun < best_score:
                best_point = result.x
                best_points = best_point.reshape(n, 2)
        except Exception:
            pass
            
    return best_points

# EVOLVE-BLOCK-END
