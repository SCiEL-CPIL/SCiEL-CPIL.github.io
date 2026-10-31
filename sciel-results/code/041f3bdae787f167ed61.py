import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11

tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

def _calculate_min_area(points):
    """Compute minimum triangle area for 11 points."""
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5
    return np.min(areas)

def _is_inside_triangle(x, y):
    """Check if point (x,y) is inside the triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    if y < -1e-9:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True

def _project_to_triangle(points):
    """Project points into the triangle."""
    pts = points.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        # Simple projection: if outside, move towards centroid (0.5, sqrt(3)/6)
        # Or clip to boundaries. Clipping is safer for area calc.
        if y < 0:
            pts[i, 1] = 0
        if y > np.sqrt(3) * x:
            # Project onto y = sqrt(3)*x
            # Line: sqrt(3)x - y = 0
            # Point: (x,y)
            # dist = (sqrt(3)x - y) / 2
            # proj = (x,y) - (sqrt(3)x - y)/2 * (sqrt(3), -1)
            d = (np.sqrt(3) * x - y) / 2.0
            pts[i, 0] -= d * np.sqrt(3)
            pts[i, 1] += d
        if y > np.sqrt(3) * (1 - x):
            # Project onto y = sqrt(3)*(1-x) => sqrt(3)x + y - sqrt(3) = 0
            d = (np.sqrt(3) * x + y - np.sqrt(3)) / 2.0
            pts[i, 0] -= d * np.sqrt(3)
            pts[i, 1] -= d
    return pts

def _objective(points_flat):
    """Objective function for minimization: -min_area."""
    points = points_flat.reshape(N, 2)
    # Soft penalty for being outside
    # We rely on projection or constraints, but NM doesn't handle constraints well.
    # We will use a penalty method.
    penalty = 0.0
    for i in range(N):
        x, y = points[i]
        if y < 0:
            penalty += (y ** 2) * 1000
        if y > np.sqrt(3) * x:
            penalty += ((np.sqrt(3) * x - y) ** 2) * 1000
        if y > np.sqrt(3) * (1 - x):
            penalty += ((np.sqrt(3) * (1 - x) - y) ** 2) * 1000
    
    if penalty > 1e-4:
        return penalty
        
    min_area = _calculate_min_area(points)
    return -min_area

def _local_refine(points):
    """Local refinement using Nelder-Mead."""
    result = minimize(
        _objective, 
        points.flatten(), 
        method='Nelder-Mead', 
        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    best_pts = result.x.reshape(N, 2)
    # Final projection to ensure validity
    best_pts = _project_to_triangle(best_pts)
    return best_pts

def _generate_symmetric_initial_guess():
    """
    Generate a symmetric initial guess for 11 points.
    Strategy:
    1. Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    2. Midpoints of edges: (0.5, 0), (0.25, sqrt(3)/4), (0.75, sqrt(3)/4)
    3. Centroid: (0.5, sqrt(3)/6)
    4. 4 points forming a smaller triangle/quad around centroid?
       Let's place them at 1/3 and 2/3 height.
       At height h = sqrt(3)/6 * 2 = sqrt(3)/3 (1/3 up):
       Width is 2/3. Center x=0.5.
       Points at x = 0.5 +/- 0.25 => 0.25, 0.75.
       So (0.25, sqrt(3)/3) and (0.75, sqrt(3)/3).
       At height h = sqrt(3)/6 * 4 = 2*sqrt(3)/3 (2/3 up):
       Width is 1/3. Center x=0.5.
       Points at x = 0.5 +/- 0.125 => 0.375, 0.625.
       So (0.375, 2*sqrt(3)/3) and (0.625, 2*sqrt(3)/3).
    
    Total points: 3 (vert) + 3 (mid) + 1 (cent) + 2 (h=1/3) + 2 (h=2/3) = 11.
    """
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, np.sqrt(3)/2.0])
    
    # Midpoints
    points.append([0.5, 0.0])
    points.append([0.25, np.sqrt(3)/4.0])
    points.append([0.75, np.sqrt(3)/4.0])
    
    # Centroid
    points.append([0.5, np.sqrt(3)/6.0])
    
    # Height 1/3 (y = sqrt(3)/3)
    points.append([0.25, np.sqrt(3)/3.0])
    points.append([0.75, np.sqrt(3)/3.0])
    
    # Height 2/3 (y = 2*sqrt(3)/3)
    points.append([0.375, 2.0*np.sqrt(3)/3.0])
    points.append([0.625, 2.0*np.sqrt(3)/3.0])
    
    return np.array(points)

def construct():
    """Main function: Symmetric Initialization + Multi-Start Local Search."""
    
    # 1. Generate symmetric guess
    initial_points = _generate_symmetric_initial_guess()
    
    # Ensure it's valid
    initial_points = _project_to_triangle(initial_points)
    
    best_points = initial_points.copy()
    best_score = _calculate_min_area(best_points)
    
    # 2. Local refinement from the symmetric start
    refined = _local_refine(initial_points)
    score = _calculate_min_area(refined)
    if score > best_score:
        best_score = score
        best_points = refined
        
    # 3. Multi-start with small perturbations to escape local minima
    rng = np.random.RandomState(42)
    for i in range(20):
        # Perturb the best known point
        # Use small noise to stay near the symmetric structure but explore neighbors
        noise = rng.normal(0, 0.05, size=(N, 2))
        perturbed = best_points + noise
        perturbed = _project_to_triangle(perturbed)
        
        # Refine
        refined = _local_refine(perturbed)
        score = _calculate_min_area(refined)
        
        if score > best_score:
            best_score = score
            best_points = refined
            # Restart perturbation from new best? Or continue from original?
            # Continuing from original symmetric guess is safer to avoid drifting too far
            # but restarting from new best can climb higher.
            # Let's restart from the new best for subsequent iterations to allow "hill climbing"
            # But we also want to explore the symmetric basin.
            # Let's just update best_points for the next perturbation source.
            
    return best_points
