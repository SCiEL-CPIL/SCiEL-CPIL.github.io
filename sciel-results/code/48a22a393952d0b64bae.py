import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2.0])
    
    def get_areas(pts):
        # pts is (N, 2)
        # Calculate areas of all triangles
        # Using cross product: 0.5 * |(x2-x1)(y3-y1) - (y2-y1)(x3-x1)|
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    # Vector p2-p1, p3-p1
                    v1 = p2 - p1
                    v2 = p3 - p1
                    cross = v1[0]*v2[1] - v1[1]*v2[0]
                    area = 0.5 * abs(cross)
                    areas.append(area)
        return np.array(areas)
    
    def boundary_penalty(pts):
        # pts (N, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        # Constraints:
        # 1. y >= 0
        # 2. y <= sqrt(3) * x  => sqrt(3)*x - y >= 0
        # 3. y <= sqrt(3) * (1-x) => sqrt(3)*(1-x) - y >= 0
        
        pen1 = np.maximum(0, -y)
        pen2 = np.maximum(0, y - np.sqrt(3)*x)
        pen3 = np.maximum(0, y - np.sqrt(3)*(1-x))
        
        # Quadratic penalty
        return 100.0 * (np.sum(pen1**2) + np.sum(pen2**2) + np.sum(pen3**2))
        
    def objective_flat(params, beta):
        # params is flat array of size 2N
        pts = params.reshape(N, 2)
        areas = get_areas(pts)
        
        # Soft min calculation
        # We want to maximize min(areas), so minimize -min(areas)
        # Soft min approx: -1/beta * log(sum(exp(-beta * areas)))
        
        if len(areas) == 0:
            return 0.0
            
        # Numerical stability for log-sum-exp
        # z = -beta * areas
        # log_sum_exp(z) = max(z) + log(sum(exp(z - max(z))))
        z = -beta * areas
        max_z = np.max(z)
        # If max_z is very large negative, exp(z-max_z) is small.
        # If max_z is very large positive, exp(z-max_z) is <= 1.
        # Here z is negative (areas > 0, beta > 0).
        # So max_z is the least negative (corresponding to min area).
        
        # Handle case where areas might be 0 or very small
        # z - max_z <= 0
        exp_sum = np.sum(np.exp(z - max_z))
        
        if exp_sum == 0:
            lse = max_z
        else:
            lse = max_z + np.log(exp_sum)
            
        soft_min_val = -lse / beta
        
        pen = boundary_penalty(pts)
        
        # We minimize: -soft_min_val + penalty
        # Note: soft_min_val is positive (approx min area).
        # So we minimize -soft_min_val.
        return -soft_min_val + pen

    def run_annealing(x0):
        # x0 is flat array
        current_x = x0
        betas = [10, 50, 200, 500, 1000]
        
        for beta in betas:
            if time.time() > deadline:
                break
            res = minimize(
                objective_flat, 
                current_x, 
                args=(beta,),
                method='Nelder-Mead',
                options={
                    'maxiter': 100,
                    'xatol': 1e-5,
                    'fatol': 1e-5
                }
            )
            current_x = res.x
            # Optional: project back to feasible region to help next stage?
            # The penalty should keep it close.
            
        return current_x

    def generate_initial_points(rng):
        # Generate random points in barycentric coords
        # u, v >= 0, u+v <= 1
        pts = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                # Rejection or scaling
                u /= (u+v)
                v /= (u+v)
            # Map to cartesian
            # P = u*B + v*C + (1-u-v)*A
            # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
            x = u * 1.0 + v * 0.5
            y = u * 0.0 + v * (np.sqrt(3)/2.0)
            pts.append([x, y])
        return np.array(pts).flatten()

    best_score = -np.inf
    best_points = None
    
    # Try multiple restarts
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = np.random.RandomState(42 + i)
        x0 = generate_initial_points(rng)
        
        # Ensure initial points are strictly inside to avoid huge penalties initially
        # The generator above ensures u,v >= 0 and u+v <= 1, so they are inside.
        
        res_x = run_annealing(x0)
        
        # Evaluate final score (hard min)
        pts = res_x.reshape(N, 2)
        areas = get_areas(pts)
        min_area = np.min(areas)
        
        # Check validity
        pen = boundary_penalty(pts)
        if pen > 1e-4:
            # If penalty is too high, discard
            continue
            
        if min_area > best_score:
            best_score = min_area
            best_points = pts
            
    if best_points is None:
        # Fallback to a simple symmetric pattern if optimization failed completely
        # Center + ring
        center = [0.5, np.sqrt(3)/6.0]
        ring = []
        for k in range(10):
            theta = 2 * np.pi * k / 10.0
            r = 0.2
            x = 0.5 + r * np.cos(theta)
            y = np.sqrt(3)/6.0 + r * np.sin(theta)
            ring.append([x, y])
        pts = np.array([center] + ring)
        best_points = pts
        best_score = np.min(get_areas(pts))
        
    return best_points

# EVOLVE-BLOCK-END
