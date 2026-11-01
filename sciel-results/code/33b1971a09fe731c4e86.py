import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_smooth(x, tau=50.0):
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def construct():
    start_time = time.time()
    timeout = 140.0
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # 1. Initial Candidates
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * 2.0
    
    p_extra1 = np.array([0.5, 0.5, 0.5])
    p_extra2 = np.array([-0.5, -0.5, -0.5])
    init_ico = np.vstack([ico, p_extra1, p_extra2])
    
    candidates = [init_ico.flatten()]
    
    for i in range(5):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 2.0
        candidates.append(pts.flatten())

    # 2. Optimization Loop with Random Restarts
    n_restarts = 100
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 5:
            break
            
        if i < len(candidates):
            x0 = candidates[i]
        else:
            x0 = rng.uniform(-2, 2, size=(N * D))
            
        try:
            # Stage 1: Smooth optimization with moderate tau
            res1 = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(50.0,),
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res1.x is not None:
                pts1 = res1.x.reshape(N, D)
                update_best(pts1)
                
                # Stage 2: High precision polish with high tau
                if time.time() - start_time < timeout - 2:
                    res2 = minimize(
                        objective_smooth, 
                        res1.x, 
                        method='L-BFGS-B', 
                        args=(500.0,),
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res2.x is not None:
                        pts2 = res2.x.reshape(N, D)
                        update_best(pts2)
                        
        except Exception:
            pass

    if best_points is None:
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
