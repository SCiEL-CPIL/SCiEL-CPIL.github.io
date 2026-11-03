import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Vertices of the equilateral triangle
# (0,0), (1,0), (0.5, sqrt(3)/2)
# Area of this triangle is sqrt(3)/4
TRI_AREA = np.sqrt(3) / 4.0

def _triangle_area(p1, p2, p3):
    """Calculate area of triangle given 3 points (each shape (2,))."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Calculate minimum area of any triangle formed by N points."""
    min_area = float('inf')
    # Pre-compute all pairs? No, just combinations.
    # For N=11, C(11,3) = 165. This is very fast.
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _objective(x):
    """Objective function to minimize: negative of min triangle area."""
    points = x.reshape(N, 2)
    min_area = _min_triangle_area(points)
    return -min_area

def _constraints(x):
    """
    Constraints to keep points inside the triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Inequalities:
    1. y >= 0
    2. y <= sqrt(3) * x  (Line from (0,0) to (0.5, sqrt(3)/2))
    3. y <= -sqrt(3) * x + sqrt(3) (Line from (1,0) to (0.5, sqrt(3)/2))
    """
    points = x.reshape(N, 2)
    xs = points[:, 0]
    ys = points[:, 1]
    sqrt3 = np.sqrt(3)
    
    cons = []
    # y >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, ys=ys: ys})
    # y - sqrt(3)x <= 0  =>  sqrt(3)x - y >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, xs=xs, ys=ys, s3=sqrt3: s3*xs - ys})
    # y + sqrt(3)x - sqrt(3) <= 0 => sqrt(3) - y - sqrt(3)x >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, xs=xs, ys=ys, s3=sqrt3: s3 - ys - s3*xs})
    
    return cons

def _random_points_in_triangle(rng):
    """Generate N random points uniformly distributed in the triangle."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        # Map (u,v) in unit triangle to our triangle
        # Unit triangle vertices: (0,0), (1,0), (0,1)
        # Our triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
        # Point = A + u*(B-A) + v*(C-A)
        # Point = (0,0) + u*(1,0) + v*(0.5, sqrt(3)/2)
        # Point = (u + 0.5v, sqrt(3)/2 * v)
        points[i, 0] = u + 0.5 * v
        points[i, 1] = np.sqrt(3) / 2.0 * v
    return points

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin for 150s limit
    
    best_score = -np.inf
    best_points = None
    
    rng = np.random.RandomState(42)
    
    # Number of restarts
    # We estimate each SLSQP run takes ~1-5 seconds depending on difficulty.
    # Let's try 20 restarts first, and break if time runs out.
    max_restarts = 30
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initial guess
        # Mix of random and grid-like structures to escape local minima
        if i < 5:
             # Grid-like initialization
             # Place points in a rough grid inside the triangle
             pts = []
             # Simple spiral or grid
             # For now, just random with some structure
             x_coords = np.linspace(0.1, 0.9, 4)
             y_coords = np.linspace(0.1, 0.7, 3)
             # This is getting complicated. Let's stick to random but with different seeds.
             pass
        
        x0 = _random_points_in_triangle(rng).flatten()
        
        try:
            res = minimize(_objective, x0, method='SLSQP', constraints=_constraints, options={'maxiter': 100, 'ftol': 1e-12})
            
            if res.success:
                min_area = -res.fun
                if min_area > best_score:
                    best_score = min_area
                    best_points = res.x.reshape(N, 2)
            else:
                # Even if not successful, check if the final point is valid and better
                # SLSQP often returns a valid point even if it doesn't converge to high precision
                x_final = res.x
                # Check constraints manually
                points = x_final.reshape(N, 2)
                xs = points[:, 0]
                ys = points[:, 1]
                sqrt3 = np.sqrt(3)
                valid = True
                if np.any(ys < -1e-9) or np.any(s3*xs - ys < -1e-9 if (s3:=sqrt3) else 0) or np.any(sqrt3 - ys - sqrt3*xs < -1e-9):
                    valid = False
                if valid:
                    min_area = _min_triangle_area(points)
                    if min_area > best_score:
                        best_score = min_area
                        best_points = points
                        
        except Exception:
            continue

    if best_points is None:
        # Fallback to seed-like random points if optimization failed completely
        return _random_points_in_triangle(rng)
        
    return best_points
