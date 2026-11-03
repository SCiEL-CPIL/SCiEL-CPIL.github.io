import numpy as np
import time
from scipy.optimize import minimize

N = 11
# Unit area equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def _min_triangle_area(points):
    """
    Vectorized calculation of the minimum area of any triangle formed by the points.
    points: array of shape (N, 2)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute all pairwise differences
    # diff[i, j] = points[j] - points[i]
    # We need cross product of (p_j - p_i) and (p_k - p_i)
    
    # To be efficient and simple for N=11:
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _objective(params):
    """
    Objective function for minimization (we minimize negative area).
    Includes penalty for points outside the triangle.
    """
    points = params.reshape((N, 2))
    
    # Penalty for being outside
    penalty = 0.0
    for i in range(N):
        x, y = points[i]
        # Barycentric coordinates or simple inequalities
        # 1. y >= 0
        if y < -1e-9:
            penalty += (y) ** 2 * 100
        # 2. y <= sqrt(3)/2
        if y > np.sqrt(3)/2 + 1e-9:
            penalty += (y - np.sqrt(3)/2) ** 2 * 100
        # 3. x >= 0
        if x < -1e-9:
            penalty += (x) ** 2 * 100
        # 4. x <= 1
        if x > 1 + 1e-9:
            penalty += (x - 1) ** 2 * 100
        # 5. Line V0-V2: y <= sqrt(3) * x
        # y - sqrt(3)*x <= 0
        val5 = y - np.sqrt(3) * x
        if val5 > 1e-9:
            penalty += val5 ** 2 * 100
        # 6. Line V1-V2: y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)
        val6 = y - np.sqrt(3) * (1 - x)
        if val6 > 1e-9:
            penalty += val6 ** 2 * 100
            
    area = _min_triangle_area(points)
    return -area + penalty

def _project_to_triangle(points):
    """
    Project points back into the triangle if they are slightly outside.
    This helps stabilize the optimizer.
    """
    projected = points.copy()
    for i in range(N):
        x, y = projected[i]
        
        # Simple clamping for x, y bounds first
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, np.sqrt(3)/2)
        
        # Check line V0-V2: y <= sqrt(3) * x
        # If y > sqrt(3)*x, project onto line
        if y > np.sqrt(3) * x:
            # Line: y = sqrt(3) * x
            # Projection of (x,y) onto y=sqrt(3)x
            # Normal vector n = (-sqrt(3), 1)
            # Point on line closest to (x,y)
            # Let's just clamp x and y to satisfy y <= sqrt(3)*x
            # If we keep x, max y is sqrt(3)*x
            y = np.sqrt(3) * x
            
        # Check line V1-V2: y <= -sqrt(3) * (x - 1)
        if y > -np.sqrt(3) * (x - 1):
            # If we keep x, max y is -sqrt(3)*(x-1)
            y = -np.sqrt(3) * (x - 1)
            
        # Re-check y >= 0
        y = max(0, y)
        # Re-check x bounds after y adjustment? 
        # If y was clamped to 0, x is fine.
        # If y was clamped to line, x is fine.
        
        projected[i] = [x, y]
    return projected

def _generate_symmetric_init():
    """
    Generate a high-quality initial configuration based on hexagonal lattice symmetry.
    We place points in a grid that respects the 3-fold symmetry of the triangle.
    
    Strategy:
    1. Start with a hexagonal grid.
    2. Select 11 points that are well-distributed.
    3. Ensure they are inside the triangle.
    
    A known good configuration for 11 points often involves:
    - 3 vertices? No, usually interior or near-boundary.
    - Let's try a 4-3-3-1 or similar layering.
    
    Alternative: Use a deterministic "relaxation" from a coarse grid.
    Let's create a grid of candidate points and pick 11.
    """
    # Create a fine grid of points inside the triangle
    # We will use a hexagonal lattice pattern
    # Side length of small triangles ~ 1/3 or 1/4
    
    # Let's try a specific symmetric pattern:
    # Layer 1 (bottom): 4 points
    # Layer 2: 3 points
    # Layer 3: 3 points
    # Layer 4 (top): 1 point
    # Total: 11
    
    # Coordinates in barycentric-like or Cartesian
    # Let's use a scaled hexagonal grid
    
    # Define a function to map (u, v) in a unit hex grid to Cartesian
    # For an equilateral triangle, we can use coordinates relative to vertices
    
    # Let's try a simpler approach: 
    # Place points at centers of a 3x3x3 subdivision?
    
    # Let's use a known good "seed" from literature or intuition.
    # For N=11, a common optimal structure is related to the triangular lattice.
    
    # Let's generate points on a triangular lattice with spacing 1/3
    # Rows:
    # Row 0 (y=0): x = 0, 1/3, 2/3, 1 (4 points) -> but 0 and 1 are vertices.
    # Maybe shift slightly inwards.
    
    # Let's use a "jittered" grid to break symmetry if needed, but start symmetric.
    
    # Candidate points:
    # We will define 11 specific points that form a nice pattern.
    
    # Pattern:
    # 1. Center
    # 2-7. Hexagon around center (radius r1)
    # 8-11. Four corners of a square? No, triangle.
    
    # Let's try:
    # 1 point at centroid (0.5, sqrt(3)/6)
    # 6 points in a ring around it
    # 4 points near the midpoints of edges?
    
    # Actually, a 4-3-3-1 layering is robust.
    # y levels:
    # y0 = 0.1
    # y1 = 0.2
    # y2 = 0.3
    # y3 = 0.4
    
    # Let's construct points manually for a good start.
    # The triangle is (0,0), (1,0), (0.5, 0.866)
    
    # Points:
    # Bottom row (y ~ 0.1): 4 points
    p1 = [0.1, 0.1]
    p2 = [0.35, 0.1]
    p3 = [0.65, 0.1]
    p4 = [0.9, 0.1]
    
    # Second row (y ~ 0.25): 3 points
    p5 = [0.2, 0.25]
    p6 = [0.5, 0.25]
    p7 = [0.8, 0.25]
    
    # Third row (y ~ 0.4): 3 points
    p8 = [0.3, 0.4]
    p9 = [0.5, 0.4]
    p10 = [0.7, 0.4]
    
    # Top (y ~ 0.55): 1 point
    p11 = [0.5, 0.55]
    
    points = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11])
    
    # Check if inside
    for i in range(N):
        if not _is_inside(points[i]):
             # If not, project
             points[i] = _project_single(points[i])
             
    return points

def _is_inside(pt):
    x, y = pt
    if y < 0 or y > np.sqrt(3)/2 or x < 0 or x > 1:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > -np.sqrt(3) * (x - 1) + 1e-9:
        return False
    return True

def _project_single(pt):
    x, y = pt
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, np.sqrt(3)/2)
    if y > np.sqrt(3) * x:
        y = np.sqrt(3) * x
    if y > -np.sqrt(3) * (x - 1):
        y = -np.sqrt(3) * (x - 1)
    return [x, y]

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # 1. Generate initial symmetric configuration
    init_points = _generate_symmetric_init()
    init_params = init_points.flatten()
    
    best_params = init_params
    best_score = -_objective(init_params)
    
    # 2. Run Nelder-Mead optimization
    # We will run multiple restarts with slightly perturbed initial points
    # to avoid local minima.
    
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on time
    # Each Nelder-Mead run takes ~1-2s for N=11 if converged.
    # We can afford ~50-100 restarts.
    
    max_restarts = 100
    
    for i in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Perturb initial points
        if i == 0:
            current_params = init_params
        else:
            # Add random noise
            noise = rng.normal(0, 0.05, size=(N, 2))
            current_points = init_points + noise
            current_params = current_points.flatten()
            
        # Optimize
        try:
            res = minimize(
                _objective, 
                current_params, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-6, 
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            # Project result to ensure validity
            res_points = res.x.reshape((N, 2))
            res_points = _project_to_triangle(res_points)
            
            # Evaluate
            score = -_objective(res_points.flatten())
            
            if score > best_score:
                best_score = score
                best_params = res_points.flatten()
                
        except Exception as e:
            pass
            
    # Final cleanup
    final_points = best_params.reshape((N, 2))
    final_points = _project_to_triangle(final_points)
    
    return final_points

# EVOLVE-BLOCK-END
