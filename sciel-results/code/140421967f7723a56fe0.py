import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    """
    Center and normalize points to unit mean squared norm.
    This removes translational and scale invariance.
    """
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_smooth(x, tau=100.0):
    """
    Smooth approximation of the log ratio of max to min squared distance.
    Uses Log-Sum-Exp for soft max and soft min.
    Points are internally centered and scale-normalized to handle invariances.
    """
    points = x.reshape(N, D)
    
    # Center and normalize to handle invariances
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12) # Avoid log(0)
    
    # Soft max
    m_max = np.max(sq_dists)
    # Subtract max for numerical stability in exp
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    # Subtract min for numerical stability in exp
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # We want to minimize log(soft_max) - log(soft_min)
    # This is equivalent to minimizing log(soft_max / soft_min)
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj_val

def objective_direct(x):
    """
    Direct non-smooth objective for final polishing.
    Minimizes log(max_sq_dist / min_sq_dist) = log(max) - log(min).
    This is used only in the final step with careful initialization.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-10:
        return 1e10
    return np.log(np.max(sq_dists)) - np.log(min_d)

def fibonacci_sphere(n, seed=42):
    """Generate n points on a sphere using Fibonacci lattice."""
    rng = np.random.default_rng(seed)
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = np.zeros((n, D))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(max(0, 1 - z**2))
        theta = golden_angle * i + rng.uniform(0, 2*np.pi) * 0.01
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
        points[i, 2] = z
    return points

def construct():
    start_time = time.time()
    timeout = 145.0 # Safety margin
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        
        # Center and normalize for consistency before scoring
        pts = normalize_points(pts)
        
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # 1. Generate High-Symmetry Initial Candidates
    candidates = []
    
    # Candidate 1: Icosahedron vertices (12) + 2 random points on sphere
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    for i in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
        
    # Candidate 2: Icosahedron + Poles (Symmetric)
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    # Candidate 3: Fibonacci Sphere (Good spread for N=14)
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # Candidate 4: Random points on sphere
    for i in range(10):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # Candidate 5: Tetrahedron based (4 vertices + 6 edge midpoints + 4 face centers = 14)
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    edge_midpoints = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (tetra[i] + tetra[j]) / 2
            mid = mid / np.linalg.norm(mid)
            edge_midpoints.append(mid)
    
    face_centers = []
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                center = (tetra[i] + tetra[j] + tetra[k]) / 3
                center = center / np.linalg.norm(center)
                face_centers.append(center)
                
    tetra_full = np.vstack([tetra, np.array(edge_midpoints), np.array(face_centers[:4])])
    candidates.append(tetra_full.flatten())

    # 2. Optimization Loop
    n_restarts = len(candidates)
    
    # Annealing schedule for tau
    # Start low to explore, end high to refine
    taus = [10.0, 30.0, 100.0, 300.0, 1000.0]
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 15:
            break
            
        x0 = candidates[i].copy()
        
        # Normalize initial points
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            current_x = x0
            
            for tau in taus:
                if time.time() - start_time > timeout - 5:
                    break
                
                # Optimize with current tau
                res = minimize(
                    objective_smooth, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.x is not None:
                    current_x = res.x
                    pts_current = current_x.reshape(N, D)
                    update_best(pts_current)
                    
        except Exception:
            pass
    
    # 3. Final high-precision polish pass on best point
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            x_polish = best_points.flatten()
            
            # Step 3a: Very high tau smooth optimization
            res_smooth = minimize(
                objective_smooth, 
                x_polish, 
                method='L-BFGS-B', 
                args=(2000.0,),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_smooth.x is not None:
                pts_smooth_polish = res_smooth.x.reshape(N, D)
                update_best(pts_smooth_polish)
                x_polish = res_smooth.x
            
            # Step 3b: Direct non-smooth polish (risky but potentially better)
            # Use COBYLA or Nelder-Mead which don't require gradients and handle non-smoothness better
            # Or just L-BFGS-B on the direct log-ratio, which is differentiable almost everywhere
            res_direct = minimize(
                objective_direct, 
                x_polish, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_direct.x is not None:
                pts_direct_polish = res_direct.x.reshape(N, D)
                update_best(pts_direct_polish)
                
        except Exception:
            pass

    if best_points is None:
        # Fallback to first candidate
        best_points = normalize_points(candidates[0].reshape(N, D))
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
