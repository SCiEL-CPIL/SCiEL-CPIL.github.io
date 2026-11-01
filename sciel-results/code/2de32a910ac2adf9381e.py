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

def objective_smooth(x, tau=50.0):
    """
    Smooth approximation of the squared max/min pairwise distance ratio.
    Uses internal normalization to handle scale invariance robustly.
    """
    points = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Normalize to fix scale: make the mean squared distance = 1
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(mean_sq_dist)
    
    # Recompute squared distances after normalization
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max of squared distances
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Soft min of squared distances
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # Objective is the ratio of soft max to soft min (which is the squared distance ratio)
    obj = log_sum_exp_max / log_sum_exp_min
    
    return obj

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        # Check for duplicate points
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Icosahedron + 2 points (various placements)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 points at various symmetric positions
    extra_positions = [
        np.array([0, 0, 0]),  # Center
        np.array([1, 0, 0]),  # Along axis
    ]
    for i in range(6):
        pts = ico.copy()
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([pts, p1, p2])
        # Apply random rotation
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # 2. Fibonacci sphere with 14 points
    def fibonacci_sphere(n, rng):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * i / (n - 1)
            radius = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        return pts
    
    for i in range(5):
        pts = fibonacci_sphere(N, rng)
        candidates.append(pts.flatten())

    # 3. Random spherical points
    for i in range(5):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 4. Uniform random in cube
    for i in range(3):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    # Optimization Loop with aggressive tau annealing
    n_restarts = len(candidates)
    
    # Tau schedule: start low for smooth landscape, increase for precision
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0]
    
    # Phase 1: Quick optimization with low tau on all candidates
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 10:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[0],),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 2: Progressive tau increase with more iterations
    for tau_idx in range(1, len(tau_schedule)):
        if time.time() - start_time > timeout - 5:
            break
        tau = tau_schedule[tau_idx]
        max_iter = 100 + tau_idx * 50
        for i in range(n_restarts):
            if time.time() - start_time > timeout - 3:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_smooth, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 3: High-precision polish on the best candidates
    if time.time() - start_time < timeout - 5:
        # Sort candidates by their current score
        scored_candidates = []
        for i in range(n_restarts):
            pts = candidates[i].reshape(N, D)
            score = raw_score(pts)
            scored_candidates.append((score, i))
        scored_candidates.sort()
        
        # Polish the top 5 candidates with very high tau
        for score, i in scored_candidates[:5]:
            if time.time() - start_time > timeout - 2:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_smooth, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau_schedule[-1],),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 4: Additional random restarts if time remains
    restart_count = 0
    while time.time() - start_time < timeout - 3:
        x0 = rng.uniform(-1, 1, size=(N * D))
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[2],),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                
                # If this is good, do a high-precision polish
                if raw_score(pts) < best_score * 1.1 and time.time() - start_time < timeout - 2:
                    try:
                        res2 = minimize(
                            objective_smooth, 
                            res.x, 
                            method='L-BFGS-B', 
                            args=(tau_schedule[-1],),
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                        )
                        if res2.x is not None:
                            pts2 = res2.x.reshape(N, D)
                            update_best(pts2)
                    except Exception:
                        pass
        except Exception:
            pass
        restart_count += 1

    if best_points is None:
        # Fallback
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
