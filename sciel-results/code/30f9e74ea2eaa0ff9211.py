import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 16
D = 2

def calculate_score(points):
    # points: (N, 2)
    # Calculate all pairwise distances
    # Use broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists = dists[mask]
    if len(dists) == 0:
        return float('inf')
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return float('inf')
    return (max_d / min_d) ** 2

def soft_objective(params, lambda_val=100.0):
    # params: (N*2,)
    points = params.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_sq = dists_sq[mask]
    dists = np.sqrt(dists_sq)
    
    # Soft max: log sum exp
    # To avoid overflow, subtract max
    m = np.max(dists)
    soft_max = np.log(np.sum(np.exp(lambda_val * (dists - m)))) + lambda_val * m
    
    # Soft min: -log sum exp(-lambda * dists)
    # m_min = min(dists)
    # soft_min = - (log sum exp(-lambda * (dists - m_min))) - lambda * m_min
    # Actually, log(min) approx -1/lambda log sum exp(-lambda x)
    # Let's use the same scaling
    m_min = np.min(dists)
    soft_min_term = np.log(np.sum(np.exp(-lambda_val * (dists - m_min)))) - lambda_val * m_min
    # log(min) approx -1/lambda * soft_min_term
    # We want log(max) - log(min)
    # approx 1/lambda * soft_max - (-1/lambda * soft_min_term)
    # = 1/lambda * (soft_max + soft_min_term)
    
    return (soft_max + soft_min_term) / lambda_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_score = float('inf')
    best_points = None
    
    # Initial guesses
    inits = []
    
    # 1. Hexagonal 19-point cluster, remove 3
    # Generate 19 point hexagon
    hex_pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
                # Convert axial to cartesian
                x = q + r/2.0
                y = r * np.sqrt(3)/2.0
                hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    # This has 19 points.
    # Remove 3 points. Which ones?
    # Maybe the corners?
    # Indices of corners: (2,0), (0,2), (-2,0), (0,-2), (-2,0)...
    # Let's just try removing random 3 points from the 19
    for _ in range(5):
        idx = np.random.choice(19, 16, replace=False)
        inits.append(hex_pts[idx].copy())
        
    # 2. Square grid 4x4
    side = 4
    grid = np.array(list(itertools.product(range(side), repeat=2)), dtype=float)
    inits.append(grid)
    
    # 3. Random in disk
    for _ in range(5):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.random.uniform(0, 1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
        
    # 4. Circle points
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts)

    # Optimization loop
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        # Flatten
        x0 = init.flatten()
        
        # Try L-BFGS-B
        try:
            res = minimize(soft_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                pts = res.x.reshape(N, 2)
                score = calculate_score(pts)
                if score < best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass
            
        # Try Powell (derivative free, robust)
        if time.time() < deadline - 5:
            try:
                res = minimize(soft_objective, x0, method='Powell', 
                               options={'maxiter': 1000, 'ftol': 1e-12})
                if res.success:
                    pts = res.x.reshape(N, 2)
                    score = calculate_score(pts)
                    if score < best_score:
                        best_score = score
                        best_points = pts
            except Exception:
                pass
                
    if best_points is None:
        # Fallback to grid
        best_points = inits[1]
        
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 16
D = 2

def calculate_score(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dists = dists[mask]
    if len(dists) == 0:
        return float('inf')
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return float('inf')
    return (max_d / min_d) ** 2

def soft_objective(params, lambda_val=100.0):
    points = params.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dists_sq = dists_sq[mask]
    dists = np.sqrt(dists_sq)
    
    m = np.max(dists)
    soft_max = np.log(np.sum(np.exp(lambda_val * (dists - m)))) + lambda_val * m
    
    m_min = np.min(dists)
    soft_min_term = np.log(np.sum(np.exp(-lambda_val * (dists - m_min)))) - lambda_val * m_min
    
    return (soft_max + soft_min_term) / lambda_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = float('inf')
    best_points = None
    
    inits = []
    
    # 1. Hexagonal 19-point cluster, remove 3
    hex_pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
                x = q + r/2.0
                y = r * np.sqrt(3)/2.0
                hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    for _ in range(10):
        idx = np.random.choice(19, 16, replace=False)
        inits.append(hex_pts[idx].copy())
        
    # 2. Square grid 4x4
    side = 4
    grid = np.array(list(itertools.product(range(side), repeat=2)), dtype=float)
    inits.append(grid)
    
    # 3. Random in disk
    for _ in range(10):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.random.uniform(0, 1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
        
    # 4. Circle points
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts)

    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        x0 = init.flatten()
        
        try:
            res = minimize(soft_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                pts = res.x.reshape(N, 2)
                score = calculate_score(pts)
                if score < best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass
            
        if time.time() < deadline - 5:
            try:
                res = minimize(soft_objective, x0, method='Powell', 
                               options={'maxiter': 500, 'ftol': 1e-12})
                if res.success:
                    pts = res.x.reshape(N, 2)
                    score = calculate_score(pts)
                    if score < best_score:
                        best_score = score
                        best_points = pts
            except Exception:
                pass
                
    if best_points is None:
        best_points = inits[1]
        
    return best_points
# EVOLVE-BLOCK-END
