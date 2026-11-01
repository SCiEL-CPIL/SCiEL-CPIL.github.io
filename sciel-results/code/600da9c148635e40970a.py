import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.transform import Rotation

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

def objective_sphere(x, tau=50.0):
    """
    Objective for points constrained to the unit sphere.
    Normalizes each point to unit norm before computing log-ratio.
    """
    points = x.reshape(N, D)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 # Avoid division by zero
    points = points / norms
    
    # Center (optional, but for sphere, center is 0 if symmetric, but let's center to be safe)
    # Actually, for sphere codes, we don't center, we just fix radius.
    # But translation invariance is not an issue if we fix radius? 
    # If we fix radius, the center is fixed at origin? No, the points are on a sphere of radius 1 centered at origin.
    # This removes translation invariance.
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    log_sq_dists = np.log(sq_dists)
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    return soft_max_log - soft_min_log

def objective_log_ratio(x, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    Minimizing this minimizes the ratio of max to min distance.
    Uses normalization to fix scale.
    """
    points = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Compute squared distances
    sq_dists = get_pairwise_sq_dists(points)
    
    # Normalize to fix scale: make the mean squared distance = 1
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(mean_sq_dist)
    
    # Recompute squared distances after normalization
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max of log(squared distances)
    log_sq_dists = np.log(sq_dists)
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log(squared distances)
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    # Objective is the difference (log ratio)
    obj = soft_max_log - soft_min_log
    
    return obj

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s
    
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
    
    for i in range(8):
        pts = ico.copy()
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([pts, p1, p2])
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
    
    for i in range(6):
        pts = fibonacci_sphere(N, rng)
        candidates.append(pts.flatten())

    # 3. Random spherical points
    for i in range(6):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 4. Uniform random in cube
    for i in range(4):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    n_restarts = len(candidates)
    
    # Phase 1: Optimize on the Sphere (fixed radius) to find good angular configurations
    # This helps escape local minima in the radial direction
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 30:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_sphere, 
                x0, 
                method='L-BFGS-B', 
                args=(50.0,),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                # Normalize to unit sphere for storage
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                pts = pts / norms
                update_best(pts)
                # Store the sphere solution as a candidate for the free stage
                candidates[i] = pts.flatten()
        except Exception:
            pass

    # Phase 2: Optimize in Free Space starting from Sphere solutions
    # Use a moderate tau initially
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 20:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(100.0,),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 3: High precision polish on the best candidates
    # Sort candidates by their current score? 
    # We don't have individual scores, but we can assume the ones that improved best_score are good.
    # Just take the first 10.
    
    for i in range(min(10, n_restarts)):
        if time.time() - start_time > timeout - 10:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(1000.0,),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass

    # Phase 4: Differential Evolution if time permits
    if time.time() - start_time < timeout - 15:
        bounds = [(-1.0, 1.0)] * (N * D)
        
        def de_objective(x):
            return objective_log_ratio(x, tau=50.0)
        
        try:
            de_result = differential_evolution(
                de_objective, 
                bounds, 
                seed=42,
                maxiter=30,
                popsize=10,
                tol=1e-5,
                init='sobol'
            )
            
            if de_result.x is not None:
                pts = de_result.x.reshape(N, D)
                update_best(pts)
                
                # Refine with log-ratio
                try:
                    res = minimize(
                        objective_log_ratio, 
                        de_result.x, 
                        method='L-BFGS-B', 
                        args=(1000.0,),
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res.x is not None:
                        pts = res.x.reshape(N, D)
                        update_best(pts)
                except Exception:
                    pass
        except Exception:
            pass

    # Final polish on the best point found
    if best_points is not None and time.time() - start_time < timeout - 2:
        x0 = best_points.flatten()
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(5000.0,),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
