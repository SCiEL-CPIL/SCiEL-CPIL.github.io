import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    def get_diam(points):
        # points: 8x2 array
        # Compute max distance
        # Use cdist for efficiency if available, but n=8 is small
        dists = []
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(points[i] - points[j])
                dists.append(d)
        return max(dists)
    
    def objective(params):
        points = params.reshape(8, 2)
        # Check for duplicate points to avoid log(0)
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(points[i] - points[j])
                if d < 1e-9:
                    return 1e10 # Penalty
        # Compute log product
        log_prod = 0.0
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(points[i] - points[j])
                log_prod += np.log(d)
        return -log_prod
        
    def diameter_constraint(params):
        points = params.reshape(8, 2)
        diam = get_diam(points)
        return 2.0 - diam
        
    # Initial guess: Regular Octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    # Diameter of regular octagon is 2 * R * cos(pi/8) ? No.
    # Max distance is between opposite points: 2R.
    # So if diameter is 2, R=1.
    # Wait, for regular octagon, the diameter is the distance between opposite vertices.
    # So R=1 gives diameter 2.
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    init_params = init_points.flatten()
    
    best_score = 1.0 # Regular octagon score
    best_points = init_points
    
    # Helper to evaluate a set of points
    def eval_points(points):
        # Scale to diameter 2? 
        # The problem says "after scaling the set to diameter 2".
        # So we should scale the points such that max dist is 2.
        diam = get_diam(points)
        if diam < 1e-9:
            return -np.inf
        scaled_points = points * (2.0 / diam)
        # Calculate score
        log_prod = 0.0
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(scaled_points[i] - scaled_points[j])
                if d < 1e-9:
                    return -np.inf
                log_prod += np.log(d)
        delta = np.exp(log_prod)
        score = delta / (8**8)
        return score

    # Evaluate initial
    score = eval_points(init_points)
    if score > best_score:
        best_score = score
        best_points = init_points

    # Optimization Strategy:
    # We want to maximize the score.
    # The score is invariant to scaling. So we can optimize the shape.
    # Let's fix the diameter to 2 during optimization to simplify constraints.
    # Objective: Maximize log_prod.
    # Constraint: diam <= 2.
    # But if we fix diam=2, we can just maximize log_prod subject to diam=2?
    # No, the max will always be on the boundary diam=2.
    
    # Let's use SLSQP to maximize log_prod subject to diam <= 2.
    # We will run several restarts.
    
    n_restarts = 50
    np.random.seed(42)
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial guess
        if r == 0:
            p0 = init_params
        else:
            # Perturb the regular octagon
            noise = np.random.normal(0, 0.5, 16)
            p0 = init_params + noise
            # Or random points?
            # Random points might be far from the optimum.
            # Let's try perturbing the regular octagon more aggressively.
            if r % 5 == 0:
                 # Random points in unit disk
                 p0 = np.random.uniform(-1, 1, 16)
        
        try:
            res = minimize(
                objective, 
                p0, 
                method='SLSQP', 
                constraints=[{'type': 'ineq', 'fun': diameter_constraint}],
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.success:
                points = res.x.reshape(8, 2)
                s = eval_points(points)
                if s > best_score:
                    best_score = s
                    best_points = points
        except Exception as e:
            pass

    # Final check and formatting
    # The function must return 8 distinct points [x, y]
    # We return the best_points found.
    # Ensure they are floats and distinct.
    result = best_points.tolist()
    return result
