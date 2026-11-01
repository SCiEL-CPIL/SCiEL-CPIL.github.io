import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh

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
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # Minimize log(soft_max) - log(soft_min)
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj_val

def objective_qcqp(x, t_target, lam=100.0):
    """
    QCQP objective: maximize min squared distance subject to max <= t_target.
    Formulation: minimize -log(soft_min) + lam * sum(max(0, sq_dists - t_target)^2)
    This directly targets the "minimize ratio" by fixing the upper bound and pushing the lower bound up.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft min via log-sum-exp
    m_min = np.min(sq_dists)
    exp_min = np.exp(-100.0 * (sq_dists - m_min))  # Fixed high tau for min to be sharp
    soft_min = m_min - np.log(np.sum(exp_min)) / 100.0
    
    # Penalty for exceeding t_target
    viol_high = np.maximum(sq_dists - t_target, 0)**2
    penalty = lam * np.sum(viol_high)
    
    # We want to maximize soft_min, so minimize -log(soft_min)
    obj_val = -np.log(soft_min) + penalty
    
    return obj_val

def objective_direct(x):
    """
    Direct non-smooth objective for SLSQP or other methods that can handle it.
    Minimizes log(max) - log(min) or just ratio.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return 1e10
    return np.log(np.max(sq_dists)) - np.log(np.min(sq_dists))

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
    timeout = 145.0 # Safety margin for 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Fibonacci sphere
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 2. Random points on sphere
    for _ in range(15):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Symmetric candidate: Ico + Poles
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    # 4. Tetrahedron-based initialization
    # Regular tetrahedron vertices
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    # Edge midpoints
    edge_midpoints = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (tetra[i] + tetra[j]) / 2
            mid = mid / np.linalg.norm(mid)
            edge_midpoints.append(mid)
    
    # Add 6 more points to reach 14: face centers
    face_centers = []
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                center = (tetra[i] + tetra[j] + tetra[k]) / 3
                center = center / np.linalg.norm(center)
                face_centers.append(center)
    
    # Combine: 4 tetra + 6 edge mids + 4 face centers = 14
    tetra_full = np.vstack([tetra, np.array(edge_midpoints), np.array(face_centers[:4])])
    candidates.append(tetra_full.flatten())
    
    # 5. More random symmetric candidates
    for _ in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())

    # Optimization Loop
    n_restarts = len(candidates)
    
    # Phase 1: Smooth Log-Ratio Annealing
    taus = [5.0, 20.0, 50.0, 100.0, 500.0, 2000.0]
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 25:
            break
            
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            current_x = x0
            
            # Phase 1: Standard smooth optimization
            for tau in taus:
                if time.time() - start_time > timeout - 25:
                    break
                
                res = minimize(
                    objective_smooth, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if res.x is not None:
                    current_x = res.x
                    pts_current = current_x.reshape(N, D)
                    update_best(pts_current)
            
            # Phase 2: QCQP-inspired refinement
            # Fix the upper bound at the current max distance, then maximize the min distance
            if time.time() - start_time < timeout - 15:
                pts_current = current_x.reshape(N, D)
                pts_current = normalize_points(pts_current)
                sq_dists = get_pairwise_sq_dists(pts_current)
                t_target = np.max(sq_dists)
                
                # We want to maximize min distance, so we minimize -log(soft_min) + penalty for > t_target
                # Start from current points
                x_qcqp = pts_current.flatten()
                
                res_qcqp = minimize(
                    objective_qcqp, 
                    x_qcqp, 
                    method='L-BFGS-B', 
                    args=(t_target, 100.0),
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if res_qcqp.x is not None:
                    pts_qcqp = res_qcqp.x.reshape(N, D)
                    update_best(pts_qcqp)
                    
                    # If we improved, try again with the new t_target
                    sq_dists_new = get_pairwise_sq_dists(pts_qcqp)
                    t_target_new = np.max(sq_dists_new)
                    if t_target_new < t_target:
                        res_qcqp2 = minimize(
                            objective_qcqp, 
                            pts_qcqp.flatten(), 
                            method='L-BFGS-B', 
                            args=(t_target_new, 100.0),
                            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        if res_qcqp2.x is not None:
                            pts_qcqp2 = res_qcqp2.x.reshape(N, D)
                            update_best(pts_qcqp2)

            # Phase 3: Direct non-smooth polish with SLSQP
            if time.time() - start_time < timeout - 10:
                pts_current = current_x.reshape(N, D)
                pts_current = normalize_points(pts_current)
                
                res = minimize(
                    objective_direct, 
                    pts_current.flatten(), 
                    method='SLSQP',
                    options={'maxiter': 150, 'ftol': 1e-14}
                )
                
                if res.x is not None:
                    pts_direct = res.x.reshape(N, D)
                    update_best(pts_direct)

        except Exception:
            pass
    
    # Final high-precision polish pass on best point
    if best_points is not None and time.time() - start_time < timeout - 10:
        try:
            x_polish = best_points.flatten()
            # Use very high tau for precise max/min
            res = minimize(
                objective_smooth, 
                x_polish, 
                method='L-BFGS-B', 
                args=(5000.0,),
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res.x is not None:
                pts_polish = res.x.reshape(N, D)
                update_best(pts_polish)
                
            # Another SLSQP pass on the result
            if time.time() - start_time < timeout - 5:
                res = minimize(
                    objective_direct, 
                    best_points.flatten(), 
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-16}
                )
                if res.x is not None:
                    pts_polish2 = res.x.reshape(N, D)
                    update_best(pts_polish2)
        except Exception:
            pass

    if best_points is None:
        best_points = normalize_points(candidates[0].reshape(N, D))
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
