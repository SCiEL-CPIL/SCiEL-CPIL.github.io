"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    # Helper: Calculate area of triangle formed by 3 points
    def area(p1, p2, p3):
        return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    # Helper: Calculate minimum area among all triples
    def min_area(points):
        min_a = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = area(points[i], points[j], points[k])
                    if a < min_a:
                        min_a = a
        return min_a

    # Helper: Penalty for points outside the triangle
    def penalty(points):
        pen = 0.0
        for p in points:
            x, y = p
            # y >= 0
            if y < 0: pen += y*y
            # y <= sqrt(3)*x
            if y > np.sqrt(3)*x: pen += (y - np.sqrt(3)*x)**2
            # y <= sqrt(3)*(1-x)
            if y > np.sqrt(3)*(1-x): pen += (y - np.sqrt(3)*(1-x))**2
        return pen

    # Objective function for optimizer
    # We minimize -soft_min_area + lambda * penalty
    def objective(params, alpha, lam):
        points = params.reshape((N, 2))
        # Soft min area approximation
        # Areas
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    areas.append(area(points[i], points[j], points[k]))
        areas = np.array(areas)
        # Log-sum-exp trick for stability: -1/alpha * log(sum(exp(-alpha * areas)))
        # To maximize min area, we minimize this quantity.
        # Note: large alpha -> approximates -min(areas)
        max_val = np.max(areas)
        log_sum = np.log(np.sum(np.exp(-alpha * (areas - max_val)))) + (-alpha * max_val)
        soft_min = -log_sum / alpha
        
        p_pen = penalty(points)
        
        return -soft_min + lam * p_pen

    # Generate initial guesses
    inits = []
    
    # 1. Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        # Rejection sampling for initial random points
        pts = []
        while len(pts) < N:
            u, v = rng.random(2)
            if u + v < 1:
                p = u*A + v*B + (1-u-v)*C
                pts.append(p)
        inits.append(np.array(pts).flatten())
        
    # 2. Grid-like (Triangular lattice)
    # Try to place points in a structured way
    # 11 points: maybe 3 rows?
    # Row 1: 3 points, Row 2: 4 points, Row 3: 3 points, Row 4: 1 point?
    # Let's just use a coarse grid and pick 11
    grid_pts = []
    for i in range(4):
        for j in range(4-i):
            # Barycentric coords
            u = i/3.0
            v = j/3.0
            w = 1 - u - v
            if u >= 0 and v >= 0 and w >= 0:
                p = u*A + v*B + w*C
                grid_pts.append(p)
    # grid_pts has 10 points (0..3). We need 11.
    # Add center?
    center = (A+B+C)/3.0
    grid_pts.append(center)
    # Shuffle and take 11? Or just use all 11?
    # The 4x4 grid (i=0..3, j=0..3-i) has 1+2+3+4 = 10 points.
    # So we have 10 grid points + 1 center = 11.
    inits.append(np.array(grid_pts).flatten())
    
    # 3. Symmetric "Star"
    # Center + 10 around it
    center = (A+B+C)/3.0
    pts = [center]
    angle_step = 2 * np.pi / 10
    radius = 0.3
    for i in range(10):
        angle = i * angle_step
        # Project circle to triangle? Just try small radius
        p = center + radius * np.array([np.cos(angle), np.sin(angle)])
        # Check if inside, if not, move to center
        if penalty(np.array([p])) > 1e-6:
            p = center
        pts.append(p)
    inits.append(np.array(pts).flatten())

    best_params = None
    best_score = -float('inf')
    best_points = None

    for init in inits:
        if time.time() > deadline:
            break
            
        # Try different alphas
        for alpha in [10, 50, 100]:
            if time.time() > deadline:
                break
                
            res = minimize(objective, init, args=(alpha, 10.0), method='Powell', options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8})
            
            if res.success:
                points = res.x.reshape((N, 2))
                # Verify validity
                if penalty(points) < 1e-5:
                    score = min_area(points)
                    if score > best_score:
                        best_score = score
                        best_points = points
                        best_params = res.x

    if best_points is None:
        # Fallback to first init if optimization failed
        best_points = inits[0].reshape((N, 2))
        
    return best_points
# EVOLVE-BLOCK-END
