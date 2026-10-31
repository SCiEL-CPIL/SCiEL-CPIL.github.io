"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    # Equilateral triangle vertices
    h = np.sqrt(3) / 2
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    
    # Pre-compute constant for area calculation if needed, but direct calc is fine.
    
    def points_from_bary(bary_coords):
        """Convert barycentric coordinates (a, b) for each point to Cartesian.
        bary_coords shape: (N, 2). c = 1 - a - b.
        Point = a*V[0] + b*V[1] + c*V[2]
        """
        points = np.zeros((N, 2))
        for i in range(N):
            a = bary_coords[i, 0]
            b = bary_coords[i, 1]
            c = 1.0 - a - b
            points[i] = a * V[0] + b * V[1] + c * V[2]
        return points

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
        """
        x: shape (2*N,) flattened barycentric coords (a1, b1, a2, b2, ...)
        Returns negative of min triangle area.
        """
        bary_coords = x.reshape(-1, 2)
        # Ensure barycentric coords are valid (a>=0, b>=0, a+b<=1)
        # Bounds and constraints handle this, but we can add a small penalty if violated
        # just in case, though SLSQP should respect them.
        if np.any(bary_coords < -1e-6) or np.any(bary_coords > 1 + 1e-6) or np.any(bary_coords[:, 0] + bary_coords[:, 1] > 1 + 1e-6):
            return 100.0 # Penalty
        
        points = points_from_bary(bary_coords)
        area = min_triangle_area(points)
        return -area

    # Bounds for a_i, b_i
    # 0 <= a <= 1, 0 <= b <= 1
    bounds = [(0, 1)] * (2 * N)
    
    # Constraints: a_i + b_i <= 1  => 1 - a_i - b_i >= 0
    cons = []
    for i in range(N):
        def make_constraint(idx):
            def constraint(x):
                a = x[2*idx]
                b = x[2*idx+1]
                return 1 - a - b
            return constraint
        cons.append({'type': 'ineq', 'fun': make_constraint(i)})

    # Multi-start optimization
    best_x = None
    best_score = -float('inf')
    
    # Use a fixed seed for reproducibility of the "random" starts
    rng = np.random.default_rng(42)
    
    num_starts = 30 # Number of random initializations
    
    for i in range(num_starts):
        # Generate random initial barycentric coordinates
        # Simple rejection sampling for a, b such that a+b <= 1
        a = rng.random(N)
        b = rng.random(N)
        # Scale to ensure a+b <= 1? No, just reject and resample or use transformation.
        # Transformation: u, v in [0,1]. a=u, b=v*(1-u). This maps square to triangle.
        u = rng.random(N)
        v = rng.random(N)
        a = u
        b = v * (1 - u)
        
        x0 = np.array([a[0], b[0], a[1], b[1], ..., a[N-1], b[N-1]]).flatten()
        # Actually, let's just build the array properly
        x0_list = []
        for j in range(N):
            x0_list.append(a[j])
            x0_list.append(b[j])
        x0 = np.array(x0_list)
        
        # Run SLSQP
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                              constraints=cons, options={'maxiter': 200, 'ftol': 1e-10})
            if result.success:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_x = result.x
        except Exception:
            pass
            
    if best_x is None:
        # Fallback: return vertices and center if optimization failed
        # Just return a grid
        points = []
        # Vertices
        points.append(V[0])
        points.append(V[1])
        points.append(V[2])
        # Midpoints of edges
        points.append((V[0] + V[1]) / 2)
        points.append((V[1] + V[2]) / 2)
        points.append((V[2] + V[0]) / 2)
        # Center
        points.append(np.mean(V, axis=0))
        # 4 more points... just random inside
        for _ in range(4):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u, v = 1-u, 1-v
            p = u * V[0] + v * V[1] + (1-u-v) * V[2]
            points.append(p)
        return np.array(points)

    # Convert best_x to points
    bary_best = best_x.reshape(-1, 2)
    final_points = points_from_bary(bary_best)
    
    return final_points
# EVOLVE-BLOCK-END
