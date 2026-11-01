import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    # Use broadcasting for speed
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the true (non-smooth) score: max/min squared distance."""
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-15:
        return np.inf
    return np.max(sq_dists) / min_d

def normalize_points(points):
    """
    Center points at origin and scale such that the sum of squared norms is 1.
    This fixes the gauge (translation and scale) to help the optimizer.
    """
    pts = points - np.mean(points, axis=0)
    norm_sq = np.sum(pts**2)
    if norm_sq < 1e-15:
        return pts
    return pts / np.sqrt(norm_sq)

def objective_smooth(x, tau):
    """
    Smooth log-ratio objective: log(soft_max(d^2)) - log(soft_min(d^2)).
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Clip to prevent log(0) or exp(overflow)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max: log(sum(exp(d^2/tau))) * tau
    # Using log-sum-exp trick for stability
    m_max = np.max(sq_dists)
    soft_max = m_max + tau * np.log(np.sum(np.exp((sq_dists - m_max) / tau)))
    
    # Soft min: -log(sum(exp(-d^2/tau))) * tau
    m_min = np.min(sq_dists)
    soft_min = m_min - tau * np.log(np.sum(np.exp((m_min - sq_dists) / tau)))
    
    # Ensure soft_min is positive for log
    soft_min = np.maximum(soft_min, 1e-15)
    soft_max = np.maximum(soft_max, 1e-15)
    
    return np.log(soft_max) - np.log(soft_min)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit, 150s kill
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Fibonacci Sphere
    def fibonacci_sphere(n_points):
        points = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_points):
            y = 1.0 - (i / (n_points - 1)) * 2.0
            radius = np.sqrt(1.0 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 2. Random points on sphere
    for seed in range(10):
        r = np.random.default_rng(seed)
        pts = r.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        candidates.append(pts.flatten())
        
    # 3. Random points in ball
    for seed in range(10):
        r = np.random.default_rng(100 + seed)
        pts = r.normal(size=(N, D))
        candidates.append(pts.flatten())
        
    # 4. Perturbed Fibonacci
    for _ in range(5):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.2
        candidates.append(perturbed.flatten())
        
    # 5. Grid-like initialization (2x2x4 grid subset)
    grid_pts = []
    for x in range(2):
        for y in range(2):
            for z in range(4):
                grid_pts.append((x, y, z))
    grid_pts_arr = np.array(grid_pts, dtype=float)
    grid_pts_arr = grid_pts_arr - np.mean(grid_pts_arr, axis=0)
    candidates.append(grid_pts_arr.flatten())
    
    # Optimization parameters
    # Tau annealing: start smooth to escape local minima, end sharp to refine
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0]
    
    # Process candidates
    for i, init_x in enumerate(candidates):
        if time.time() > deadline - 20.0:
            break
            
        x_curr = init_x.copy()
        
        try:
            for tau in tau_schedule:
                if time.time() > deadline - 10.0:
                    break
                    
                # Normalize before each stage to keep numbers well-conditioned
                pts_curr = x_curr.reshape(N, D)
                pts_curr = normalize_points(pts_curr)
                x_curr = pts_curr.flatten()
                
                # Define objective for this tau
                def obj(x):
                    return objective_smooth(x, tau)
                
                # L-BFGS-B is efficient for smooth objectives
                res = minimize(
                    obj,
                    x_curr,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            
            # Final evaluation and normalization
            pts_final = x_curr.reshape(N, D)
            pts_final = normalize_points(pts_final)
            
            # Check for distinct points
            sq_dists = get_pairwise_sq_dists(pts_final)
            if np.min(sq_dists) > 1e-10:
                s = raw_score(pts_final)
                if np.isfinite(s) and s < best_score:
                    best_score = s
                    best_points = pts_final.copy()
                    
        except Exception:
            pass

    # If no valid points found, return a random valid set
    if best_points is None:
        best_points = fib_pts
        # Ensure it's valid
        sq_dists = get_pairwise_sq_dists(best_points)
        if np.min(sq_dists) < 1e-10:
             # Fallback to simple random
             best_points = np.random.randn(N, D)
             best_points = normalize_points(best_points)

    return best_points
