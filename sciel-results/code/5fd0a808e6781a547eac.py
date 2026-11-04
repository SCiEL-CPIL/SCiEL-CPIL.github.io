import numpy as np
import time
from scipy.optimize import differential_evolution, linprog
from scipy.spatial import ConvexHull

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon
    # We will vary the angle theta later
    
    def get_normals(theta):
        angles = theta + np.linspace(0, 2*np.pi, 7)[:-1]
        return np.array([[np.cos(a), np.sin(a)] for a in angles])

    def cost_func(params):
        # params is 1D array of 22 elements
        points = params.reshape(-1, 2)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Convex Hull Area
        try:
            hull = ConvexHull(points)
            area = hull.volume # In 2D, volume is area
        except Exception:
            # If points are collinear or duplicate, area is 0 or error
            area = 1000.0
            
        return area + 100.0 * penalty

    # Optimization
    # Bounds for positions
    bounds = [(-5, 5)] * (2 * N)
    
    # Use differential evolution
    # seed for reproducibility? No, random is fine, but let's use a fixed seed for stability in evaluation?
    # The task says "random restarts" in seed. I'll use DE which is stochastic.
    # To be safe, I'll run DE once with a decent maxiter.
    
    best_result = None
    best_cost = float('inf')
    
    # Run DE
    # maxiter=100, popsize=10 -> ~1000-2000 function evaluations?
    # DE population size is popsize * len(bounds). 10 * 22 = 220? No, popsize is multiplier.
    # Default popsize=15. 15 * 22 = 330 individuals.
    # Let's use popsize=10.
    
    res = differential_evolution(cost_func, bounds, method='differential', 
                                  maxiter=100, popsize=10, seed=42, 
                                  tol=1e-5, atol=1e-5)
    
    if res.success:
        best_params = res.x
    else:
        # Fallback to seed-like initialization if DE fails
        best_params = np.random.uniform(-3, 3, 2*N)
        
    # Extract points
    points = best_params.reshape(-1, 2)
    
    # Find optimal theta and L
    best_L = float('inf')
    best_theta = 0.0
    best_c = (0.0, 0.0)
    
    # Search for optimal rotation theta in [0, 60)
    # Step 0.1 degrees
    theta_steps = np.arange(0, np.pi/3, np.pi/1800) # 0.1 deg
    
    for theta in theta_steps:
        normals = get_normals(theta)
        
        # Setup LP
        # Minimize L
        c_obj = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        for i in range(N):
            p = points[i]
            for k in range(6):
                n = normals[k]
                proj = np.dot(p, n)
                
                # c . n - L * sqrt(3)/2 <= proj
                # c_x n_x + c_y n_y - L * sqrt(3)/2 <= proj
                A_ub.append([n[0], n[1], -np.sqrt(3)/2])
                b_ub.append(proj)
                
                # -c . n + L * sqrt(3)/2 <= -proj
                # -c_x n_x - c_y n_y + L * sqrt(3)/2 <= -proj
                A_ub.append([-n[0], -n[1], np.sqrt(3)/2])
                b_ub.append(-proj)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for c_x, c_y, L
        # c can be anywhere, but let's bound it to [-10, 10]
        # L must be positive
        bounds_lp = [(-10, 10), (-10, 10), (0, 10)]
        
        try:
            res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
            if res_lp.success:
                L_val = res_lp.x[2]
                if L_val < best_L:
                    best_L = L_val
                    best_theta = theta
                    best_c = (res_lp.x[0], res_lp.x[1])
        except Exception:
            pass
            
    if best_L == float('inf'):
        # Fallback
        best_L = 10.0
        best_c = (0.0, 0.0)
        best_theta = 0.0
        
    # Construct output
    # inner: list of (x, y, angle_degrees)
    # angle is 0.0 for all
    inner = [[float(x), float(y), 0.0] for x, y in points]
    
    # outer: center, side, angle
    # side is best_L
    # angle is best_theta (in degrees)
    outer_angle_deg = np.degrees(best_theta)
    
    return inner, (float(best_c[0]), float(best_c[1])), float(best_L), float(outer_angle_deg)

