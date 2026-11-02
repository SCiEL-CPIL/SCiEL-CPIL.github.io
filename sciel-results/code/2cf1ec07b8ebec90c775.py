"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    H = np.sqrt(3) / 2
    
    # Pre-compute indices for all triangles
    tri_indices = np.array(list(combinations(range(N), 3)))
    
    def project_to_triangle(points):
        """Project points onto the interior of the equilateral triangle."""
        # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
        # Inequalities:
        # 1. y >= 0
        # 2. y <= sqrt(3) * x
        # 3. y <= sqrt(3) * (1 - x)
        
        x = points[:, 0]
        y = points[:, 1]
        
        # Constraint 1: y >= 0
        mask1 = y < 0
        if np.any(mask1):
            y[mask1] = 0
            
        # Constraint 2: y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
        # If violated, project onto the line y = sqrt(3)*x
        # The line is sqrt(3)x - y = 0. Normal is (sqrt(3), -1).
        # Distance is (sqrt(3)x - y) / 2.
        # Point on line: P - dist * Normal_normalized?
        # Actually, simpler: for a point (x,y) with y > sqrt(3)x,
        # the projection onto y=sqrt(3)x is:
        # x_proj = (x + y/sqrt(3)) / (1 + 1/3) = (x + y/sqrt(3)) * 0.75
        # y_proj = sqrt(3) * x_proj
        mask2 = y > np.sqrt(3) * x
        if np.any(mask2):
            xm = x[mask2]
            ym = y[mask2]
            x_proj = (xm + ym/np.sqrt(3)) * 0.75
            y_proj = np.sqrt(3) * x_proj
            x[mask2] = x_proj
            y[mask2] = y_proj
            
        # Constraint 3: y <= sqrt(3) * (1 - x) => sqrt(3)*(1-x) - y >= 0
        # Line: y = -sqrt(3)x + sqrt(3) => sqrt(3)x + y - sqrt(3) = 0
        # Normal: (sqrt(3), 1).
        # If y > -sqrt(3)x + sqrt(3), project onto it.
        # x_proj = (x - y/sqrt(3) + 1) * 0.75 ? 
        # Let's derive: Line ax+by+c=0. a=sqrt(3), b=1, c=-sqrt(3).
        # d = (ax+by+c)/sqrt(a^2+b^2) = (sqrt(3)x + y - sqrt(3))/2.
        # P_proj = P - d * (a,b)/2.
        # x_proj = x - d*sqrt(3)/2 = x - (sqrt(3)x + y - sqrt(3))*sqrt(3)/4
        # y_proj = y - d/2 = y - (sqrt(3)x + y - sqrt(3))/4
        
        mask3 = y > np.sqrt(3) * (1 - x)
        if np.any(mask3):
            xm = x[mask3]
            ym = y[mask3]
            # Using the projection formula derived from geometry
            # Reflection across the line? No, projection.
            # Let's use the standard point-to-line projection.
            # Line: sqrt(3)x + y - sqrt(3) = 0
            # num = sqrt(3)*xm + ym - sqrt(3)
            # denom = sqrt(3)**2 + 1**2 = 4
            # x_proj = xm - sqrt(3)*num/denom
            # y_proj = ym - 1*num/denom
            num = np.sqrt(3)*xm + ym - np.sqrt(3)
            x[mask3] = xm - np.sqrt(3)*num/4.0
            y[mask3] = ym - num/4.0
            
        return np.column_stack((x, y))

    def calc_min_area(points):
        """Calculate the minimum area of any triangle formed by the points."""
        # points shape (N, 2)
        # tri_indices shape (M, 3)
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        
        # Cross product magnitude / 2
        # (p2-p1) x (p3-p1)
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        return np.min(areas)

    def objective(points):
        """Return negative min area. We want to maximize min area."""
        return -calc_min_area(points)

    def random_point_in_triangle(rng):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                return np.array([u + 0.5 * v, H * v])

    def run_sa(rng, n_iter=5000):
        # Initialize
        points = np.array([random_point_in_triangle(rng) for _ in range(N)])
        
        # Better initialization: Start with vertices and centroid, fill rest randomly?
        # Let's try a structured start for the first few restarts
        # Vertices: (0,0), (1,0), (0.5, H)
        # Centroid: (0.5, H/3)
        # This might help escape bad local minima.
        
        best_points = points.copy()
        best_score = objective(points)
        
        T = 1e-3 # Initial temperature (area scale)
        T_end = 1e-7
        cool_factor = (T_end / T) ** (1.0 / n_iter)
        
        for i in range(n_iter):
            # Perturb
            step_size = T * 10 # Scale step with temperature
            noise = rng.normal(0, step_size, size=(N, 2))
            new_points = points + noise
            new_points = project_to_triangle(new_points)
            
            new_score = objective(new_points)
            delta = new_score - best_score # Note: objective is negative, so "better" is more negative? 
            # Wait, objective returns -min_area. 
            # We want to maximize min_area, so minimize -min_area.
            # So if new_score < best_score (more negative), it's an improvement.
            # Standard SA: accept if new < old (minimization) or with prob exp((new-old)/T).
            
            if new_score < best_score or rng.random() < np.exp((best_score - new_score) / T):
                points = new_points
                best_score = new_score
                best_points = new_points.copy()
            
            T *= cool_factor
            
        return best_points, -best_score # Return positive min area

    # Run multiple restarts
    best_overall = None
    best_val = 0
    
    rng = np.random.RandomState(42)
    
    # A few restarts with different seeds/initializations
    for r in range(10):
        pts, val = run_sa(rng, n_iter=3000)
        if val > best_val:
            best_val = val
            best_overall = pts
            
    if best_overall is None:
        # Fallback
        return np.array([random_point_in_triangle(rng) for _ in range(N)])
        
    return best_overall
# EVOLVE-BLOCK-END
