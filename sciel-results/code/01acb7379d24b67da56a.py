import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Triangle vertices
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3)/2])
    
    # Function to check if a point is inside the triangle
    # Using barycentric coordinates
    def barycentric(p, v1, v2, v3):
        v0 = v2 - v1
        v1_ = v3 - v1
        v2_ = p - v1
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1_)
        d02 = np.dot(v0, v2_)
        d11 = np.dot(v1_, v1_)
        d12 = np.dot(v1_, v2_)
        denom = d00 * d11 - d01 * d01
        if denom < 1e-10:
            return None
        v = (d11 * d02 - d01 * d12) / denom
        w = (d00 * d12 - d01 * d02) / denom
        u = 1.0 - v - w
        return u, v, w

    def is_inside(p):
        u, v, w = barycentric(p, v1, v2, v3)
        if u is None:
            return False
        # Add a small epsilon to allow points on the boundary
        eps = 1e-9
        return u >= -eps and v >= -eps and w >= -eps

    def min_area(points):
        min_a = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * |cross(p2-p1, p3-p1)|
                    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                    if area < min_a:
                        min_a = area
        return min_a

    def objective(flat_points):
        points = flat_points.reshape(-1, 2)
        # Check if all points are inside
        for p in points:
            if not is_inside(p):
                # Return a large value to penalize outside points
                # Calculate distance to boundary for smoother penalty?
                # For now, simple penalty
                return 1e6
        area = min_area(points)
        return -area # Minimize negative area

    # Initialization: Triangular Lattice
    # We want 11 points.
    # Let's try a 3-4-3 grid or similar.
    # Generate a fine grid and pick 11 points that are well spread.
    
    # A good initial guess for Heilbronn problems is often a distorted lattice.
    # Let's create a 4x3 grid of candidate positions and pick 11.
    # Or simply start with a 3-4-4 arrangement.
    
    # Row 0 (bottom): 3 points
    # Row 1 (middle): 4 points
    # Row 2 (top): 3 points
    # Total 10. Need 1 more.
    
    # Let's use a 4-4-3 arrangement.
    h = np.sqrt(3)/2
    points_init = []
    
    # Bottom row (y = h/4)
    y0 = h/4
    # Width of triangle at y0: w0 = 1 - y0/h = 1 - 0.25 = 0.75
    # Center x = 0.5
    # Place 4 points
    xs0 = np.linspace(0.5 - 0.75/2 * 0.8, 0.5 + 0.75/2 * 0.8, 4) # Inset by 10%
    for x in xs0:
        points_init.append([x, y0])
        
    # Middle row (y = h/2)
    y1 = h/2
    # Width at y1: w1 = 1 - 0.5 = 0.5
    # Place 4 points
    xs1 = np.linspace(0.5 - 0.5/2 * 0.8, 0.5 + 0.5/2 * 0.8, 4)
    for x in xs1:
        points_init.append([x, y1])
        
    # Top row (y = 3h/4)
    y2 = 3*h/4
    # Width at y2: w2 = 1 - 0.75 = 0.25
    # Place 3 points
    xs2 = np.linspace(0.5 - 0.25/2 * 0.8, 0.5 + 0.25/2 * 0.8, 3)
    for x in xs2:
        points_init.append([x, y2])
        
    # Total 11 points
    points_init = np.array(points_init)
    
    # Ensure they are strictly inside (clip if necessary, though linspace with inset should be safe)
    # But let's verify and adjust if any are outside
    for i in range(N):
        if not is_inside(points_init[i]):
             # Push towards center
             points_init[i] = 0.5 * points_init[i] + 0.5 * np.array([0.5, h/2])

    best_points = points_init
    best_score = min_area(best_points)
    
    # Optimization
    # Use Nelder-Mead
    # Flatten points
    x0 = points_init.flatten()
    
    # Run optimization
    # Set options to limit iterations and time
    # maxiter can be high, but we check time in the callback if we could, but scipy doesn't easily support time callbacks in Nelder-Mead.
    # We will run a few restarts with different initializations if time permits.
    
    def run_optimization(x_init):
        res = minimize(objective, x_init, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        return res.x, res.fun

    # Try the initial grid
    x_opt, fun_val = run_optimization(x0)
    if time.time() < deadline:
        points_opt = x_opt.reshape(-1, 2)
        # Check validity
        valid = all(is_inside(p) for p in points_opt)
        if valid:
            score = min_area(points_opt)
            if score > best_score:
                best_score = score
                best_points = points_opt

    # Try a second initialization: Perturbed grid
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        x0_2 = points_init.flatten() + rng.normal(0, 0.05, size=N*2)
        # Clip to triangle roughly? The objective will penalize.
        x_opt2, fun_val2 = run_optimization(x0_2)
        points_opt2 = x_opt2.reshape(-1, 2)
        valid2 = all(is_inside(p) for p in points_opt2)
        if valid2:
            score2 = min_area(points_opt2)
            if score2 > best_score:
                best_score = score2
                best_points = points_opt2

    # Try a third initialization: Different grid (3-4-4)
    if time.time() < deadline - 10:
        points_init3 = []
        # Row 0: 3
        y0 = h/4
        xs0 = np.linspace(0.5 - 0.75/2 * 0.8, 0.5 + 0.75/2 * 0.8, 3)
        for x in xs0:
            points_init3.append([x, y0])
        # Row 1: 4
        y1 = h/2
        xs1 = np.linspace(0.5 - 0.5/2 * 0.8, 0.5 + 0.5/2 * 0.8, 4)
        for x in xs1:
            points_init3.append([x, y1])
        # Row 2: 4
        y2 = 3*h/4
        xs2 = np.linspace(0.5 - 0.25/2 * 0.8, 0.5 + 0.25/2 * 0.8, 4)
        for x in xs2:
            points_init3.append([x, y2])
        
        points_init3 = np.array(points_init3)
        for i in range(N):
            if not is_inside(points_init3[i]):
                 points_init3[i] = 0.5 * points_init3[i] + 0.5 * np.array([0.5, h/2])
                 
        x0_3 = points_init3.flatten()
        x_opt3, fun_val3 = run_optimization(x0_3)
        points_opt3 = x_opt3.reshape(-1, 2)
        valid3 = all(is_inside(p) for p in points_opt3)
        if valid3:
            score3 = min_area(points_opt3)
            if score3 > best_score:
                best_score = score3
                best_points = points_opt3

    return best_points
# EVOLVE-BLOCK-END
