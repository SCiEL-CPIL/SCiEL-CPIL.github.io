"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    # Equilateral triangle vertices
    h = np.sqrt(3) / 2
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    
    # Function to check if a point is inside the triangle using barycentric coordinates
    def is_inside(p):
        # p is (x, y)
        # Barycentric coordinates relative to V
        # A = V[0], B = V[1], C = V[2]
        A, B, C = V
        # Calculate areas
        total_area = 0.5 * abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]))
        
        # Area of sub-triangles
        area1 = 0.5 * abs((B[0] - p[0]) * (C[1] - p[1]) - (C[0] - p[0]) * (B[1] - p[1]))
        area2 = 0.5 * abs((p[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (p[1] - A[1]))
        area3 = 0.5 * abs((B[0] - A[0]) * (p[1] - A[1]) - (p[0] - A[0]) * (B[1] - A[1]))
        
        return area1 + area2 + area3 <= total_area + 1e-9

    def min_triangle_area(points):
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * | (p2-p1) x (p3-p1) |
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                    if area < min_area:
                        min_area = area
        return min_area

    def objective(x):
        points = x.reshape(-1, 2)
        
        # Penalty for points outside the triangle
        penalty = 0.0
        for i in range(N):
            p = points[i]
            if not is_inside(p):
                # Calculate distance to triangle or just a fixed penalty
                # Simple penalty: 100.0 * N to ensure it's worse than any valid config
                penalty += 100.0
        
        if penalty > 0:
            return penalty # Maximize min area, so minimize this. Higher is worse.
            
        area = min_triangle_area(points)
        return -area # We want to maximize area, so minimize negative area

    # Bounds for differential evolution
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, h)] * N
    
    # Run Differential Evolution
    # maxiter and tol control the stopping criteria
    # popsize controls the population size
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=100, 
        tol=1e-8, 
        popsize=15, 
        seed=42,
        polish=True # Use L-BFGS-B to polish the result at the end
    )
    
    x_opt = result.x
    points = x_opt.reshape(-1, 2)
    
    # Final check: if any point is slightly outside due to numerical error, project it back?
    # DE with polish should handle this, but let's ensure validity.
    # If a point is outside, the penalty would have been high, so it likely won't be the global min
    # unless the penalty is not strong enough. 100.0 is very strong.
    
    return points
# EVOLVE-BLOCK-END
