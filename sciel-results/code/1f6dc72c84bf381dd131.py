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
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

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
    
    # 1. Generate Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # 2. Initial candidates
    candidates = []
    
    # Candidate 1: Ico + 2 random points on sphere
    for i in range(8):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
        
    # Candidate 2: Ico + 2 points at (0,0,1) and (0,0,-1) 
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    # Candidate 3: Fully random points
    for i in range(5):
        pts = rng.normal(size=(N, D))
        candidates.append(pts.flatten())
    
    # 3. Optimization Loop
    n_restarts = len(candidates)
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 15:
            break
            
        x0 = candidates[i].copy()
        
        # Normalize initial points
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            # Annealing schedule for tau - more aggressive
            taus = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
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
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.x is not None:
                    current_x = res.x
                    pts_current = current_x.reshape(N, D)
                    update_best(pts_current)
                    
        except Exception:
            pass
    
    # 4. Final high-precision polish pass on best point
    if best_points is not None and time.time() - start_time < timeout - 10:
        try:
            x_polish = best_points.flatten()
            # Use very high tau for precise max/min
            res = minimize(
                objective_smooth, 
                x_polish, 
                method='L-BFGS-B', 
                args=(2000.0,),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts_polish = res.x.reshape(N, D)
                update_best(pts_polish)
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
