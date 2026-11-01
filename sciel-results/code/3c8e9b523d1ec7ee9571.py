import numpy as np
import time
from scipy.optimize import minimize, basinhopping
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
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def objective_log_ratio(x, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    Minimizing this minimizes the ratio of max to min distance.
    Includes internal normalization to fix scale and translation.
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
    
    # --- Phase 1: Global Exploration with Basin Hopping ---
    # We use a moderate tau for the basin hopping to have a smooth landscape
    # that allows the local optimizer to work well, but the stepsize is large
    # enough to jump between basins.
    
    # Generate a few diverse starting points for basin hopping
    start_points = []
    
    # 1. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    for _ in range(3):
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([ico, p1, p2])
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        start_points.append(full_pts.flatten())
        
    # 2. Fibonacci sphere
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

    start_points.append(fibonacci_sphere(N, rng).flatten())
    
    # 3. Random spherical
    for _ in range(3):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        start_points.append(pts.flatten())

    tau_bh = 20.0
    
    # Run basin hopping on the best few start points if time permits
    # We limit the number of basin hopping runs to save time for polishing
    max_bh_runs = 3
    bh_results = []
    
    for i, x0 in enumerate(start_points[:max_bh_runs]):
        if time.time() - start_time > timeout - 30:
            break
            
        def bh_objective(x):
            return objective_log_ratio(x, tau=tau_bh)
        
        # Callback to track best and enforce time limit
        bh_best_x = x0.copy()
        bh_best_f = bh_objective(x0)
        
        def bh_callback(x, f, accept):
            nonlocal bh_best_x, bh_best_f
            if f < bh_best_f:
                bh_best_f = f
                bh_best_x = x.copy()
            if time.time() - start_time > timeout - 25:
                raise Exception("Time to stop Basin Hopping")
                
        try:
            bh_res = basinhopping(
                bh_objective,
                x0,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6}},
                stepsize=0.5,
                niter=50, # Will be cut by time
                callback=bh_callback,
                seed=42 + i
            )
            # Use the best found during callback or the result
            if bh_best_f < 1e9: # Check if valid
                bh_results.append((bh_best_f, bh_best_x))
        except Exception:
            # Use whatever was found before timeout
            if bh_best_f < 1e9:
                bh_results.append((bh_best_f, bh_best_x))

    if bh_results:
        # Sort by objective value (lower is better)
        bh_results.sort(key=lambda x: x[0])
        # Take the best few for polishing
        top_bh_candidates = [x[1] for x in bh_results[:3]]
    else:
        # Fallback to original starts if BH failed
        top_bh_candidates = start_points[:3]

    # --- Phase 2: Progressive Tau Refinement ---
    # Take the best candidates and polish them with increasing tau
    tau_schedule = [50.0, 200.0, 1000.0, 5000.0]
    
    current_candidates = top_bh_candidates
    
    for tau in tau_schedule:
        if time.time() - start_time > timeout - 10:
            break
            
        next_candidates = []
        for x0 in current_candidates:
            if time.time() - start_time > timeout - 5:
                break
                
            try:
                res = minimize(
                    objective_log_ratio, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    next_candidates.append(res.x)
            except Exception:
                pass
        
        if next_candidates:
            # Keep only the best few to avoid combinatorial explosion
            scored = []
            for xc in next_candidates:
                pts = xc.reshape(N, D)
                s = raw_score(pts)
                scored.append((s, xc))
            scored.sort()
            current_candidates = [x[1] for x in scored[:3]]
        else:
            break

    # --- Phase 3: Final High-Precision Polish ---
    # Use the very best candidate found so far
    if best_points is not None:
        x_final = best_points.flatten()
        # Try a few more high-tau optimizations from slightly perturbed versions of the best
        for _ in range(3):
            if time.time() - start_time > timeout - 2:
                break
            x_perturbed = x_final + rng.normal(size=x_final.shape) * 0.01
            try:
                res = minimize(
                    objective_log_ratio, 
                    x_perturbed, 
                    method='L-BFGS-B', 
                    args=(10000.0,),
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
            except Exception:
                pass

    if best_points is None:
        # Fallback
        best_points = start_points[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
