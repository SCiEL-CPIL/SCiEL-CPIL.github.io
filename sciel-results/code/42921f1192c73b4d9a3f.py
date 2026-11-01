import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Returns flat array of squared pairwise distances (upper triangle)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute (max_dist / min_dist)^2 = max_sq / min_sq."""
    sq_dists = get_pairwise_sq_dists(points)
    mn = np.min(sq_dists)
    mx = np.max(sq_dists)
    if mn < 1e-14:
        return np.inf
    return mx / mn

def objective_barrier(x, mu=0.05):
    """
    Barrier objective: minimize log(max_sq) - log(min_sq) + barrier.
    Uses log-sum-exp for smooth max and min approximations.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Log-sum-exp for max: log(sum(exp(sq)))
    m = np.max(sq_dists)
    log_sum_exp_max = m + np.log(np.sum(np.exp(sq_dists - m)))
    
    # Log-sum-exp for min: log(sum(exp(-sq))) then negate
    m_neg = np.min(sq_dists)
    log_sum_exp_min = m_neg + np.log(np.sum(np.exp(sq_dists - m_neg)))
    # log(min) ≈ log_sum_exp_min - log(n_pairs) for uniform, but we want the actual min
    # Better: use the softmin directly
    # softmin = -log(sum(exp(-sq)))
    m_neg2 = np.max(-sq_dists)
    log_sum_exp_neg = m_neg2 + np.log(np.sum(np.exp(-sq_dists - m_neg2)))
    softmin_log = -log_sum_exp_neg
    
    # Objective: log(max) - log(min)
    obj = log_sum_exp_max - softmin_log
    
    # Add barrier to prevent points from coinciding
    # Barrier: -mu * log(min_sq)
    min_sq = np.min(sq_dists)
    barrier = -mu * np.log(min_sq)
    
    return obj + barrier

def objective_direct(x, tau=50.0):
    """
    Direct smooth ratio using log-sum-exp.
    Minimizes log(softmax) - log(softmin).
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-14)
    
    # Softmax: (1/tau) * log(sum(exp(tau * sq)))
    m = np.max(sq_dists)
    log_softmax = m + np.log(np.sum(np.exp(tau * (sq_dists - m)))) / tau
    
    # Softmin: -(1/tau) * log(sum(exp(-tau * sq)))
    m_neg = np.min(sq_dists)
    log_softmin = m_neg - np.log(np.sum(np.exp(-tau * (sq_dists - m_neg)))) / tau
    
    if log_softmin <= 0:
        return 1e10
    
    return np.log(log_softmax) - np.log(log_softmin)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        s = raw_score(pts)
        if s < best_score:
            best_score = s
            best_points = pts.copy()
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Grid-based start (3x3x2 minus 4 points)
    points_grid = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                points_grid.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_points_grid = [p for p in points_grid if p not in to_remove]
    init_grid = np.array(final_points_grid, dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    candidates.append(init_grid.flatten())
    
    # 2. Random starts with various distributions
    rng = np.random.default_rng(42)
    for trial in range(30):
        if time.time() - start_time > deadline - 10:
            break
        
        if trial % 3 == 0:
            # Uniform in cube
            pts = rng.uniform(-1, 1, size=(N, D))
        elif trial % 3 == 1:
            # Normal, scaled
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-6) * 2.0
        else:
            # Spherical shell
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-6) * (1.0 + 0.5 * rng.uniform(0, 1, size=(N, 1)))
        
        # Center
        pts = pts - np.mean(pts, axis=0)
        candidates.append(pts.flatten())
    
    # 3. Try some structured configurations
    # Tetrahedral-like arrangement
    for scale in [1.0, 1.5, 2.0]:
        pts = np.zeros((N, D))
        # Place points on concentric shells
        for i in range(N):
            angle1 = 2 * np.pi * i / N
            angle2 = np.pi * (i % 3) / 3
            r = scale * (1 + 0.3 * (i % 2))
            pts[i] = [r * np.sin(angle2) * np.cos(angle1),
                      r * np.sin(angle2) * np.sin(angle1),
                      r * np.cos(angle2)]
        pts = pts - np.mean(pts, axis=0)
        candidates.append(pts.flatten())
    
    # Optimize each candidate
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > deadline - 5:
            break
        
        x_curr = init_x.copy()
        
        # Try multiple optimization strategies
        for method_idx, (method, opts) in enumerate([
            ('L-BFGS-B', {'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-14}),
            ('BFGS', {'maxiter': 1000, 'gtol': 1e-10}),
        ]):
            if time.time() - start_time > deadline - 3:
                break
            
            try:
                res = minimize(
                    objective_barrier,
                    x_curr,
                    method=method,
                    options=opts
                )
                
                pts = res.x.reshape(N, D)
                update_best(pts)
                
                # Refine with direct objective
                if time.time() - start_time > deadline - 2:
                    break
                
                res2 = minimize(
                    objective_direct,
                    res.x,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                
                pts2 = res2.x.reshape(N, D)
                update_best(pts2)
                
                # Update x_curr for next method if improved
                if raw_score(pts2) < raw_score(pts):
                    x_curr = res2.x.copy()
                    
            except Exception:
                pass
        
        # Local perturbation refinement
        if time.time() - start_time > deadline - 1:
            break
        
        for perturb_trial in range(5):
            if time.time() - start_time > deadline - 1:
                break
            
            # Small random perturbation
            x_pert = x_curr + rng.normal(0, 0.01, size=x_curr.shape)
            
            try:
                res_p = minimize(
                    objective_barrier,
                    x_pert,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                pts_p = res_p.x.reshape(N, D)
                update_best(pts_p)
            except Exception:
                pass
    
    # Final fallback
    if best_points is None:
        best_points = init_grid
        best_score = raw_score(best_points)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
