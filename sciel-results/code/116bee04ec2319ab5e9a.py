import numpy as np
import time
from scipy.optimize import minimize, basinhopping
from scipy.spatial.transform import Rotation

N = 14
D = 3

# Precompute pairwise indices for efficiency
_pair_i, _pair_j = np.triu_indices(N, k=1)
_n_pairs = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    return sq_dists[_pair_i, _pair_j]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_log_ratio(x, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    Minimizing this minimizes the ratio of max to min distance.
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

def objective_epigraph(x, tau=100.0):
    """
    Epigraph formulation: minimize t subject to log(d_i^2) <= t for all i
    and log(d_i^2) >= -t for all i, with smooth constraints via log-sum-exp.
    
    This is equivalent to minimizing max_i log(d_i^2) - min_i log(d_i^2)
    but with a different parameterization that can be more stable.
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
    
    log_sq_dists = np.log(sq_dists)
    
    # Use log-sum-exp for smooth max and min
    m_max = np.max(log_sq_dists)
    m_min = np.min(log_sq_dists)
    
    # Smooth max: log-sum-exp of log_sq_dists
    smooth_max = m_max + np.log(np.sum(np.exp(tau * (log_sq_dists - m_max)))) / tau
    
    # Smooth min: -log-sum-exp of -log_sq_dists
    smooth_min = m_min - np.log(np.sum(np.exp(-tau * (log_sq_dists - m_min)))) / tau
    
    return smooth_max - smooth_min

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
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
    
    # Generate multiple diverse initial candidates
    def make_icosahedron():
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        return ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    def make_fibonacci_sphere(n):
        """Generate n points on a sphere using Fibonacci spiral."""
        points = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            z = 1.0 - 2.0 * i / (n - 1)
            radius = np.sqrt(1.0 - z * z)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    # Initial candidate 1: Icosahedron + 2 antipodal points
    ico = make_icosahedron()
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([-1.0, 0.0, 0.0])
    init_pts1 = np.vstack([ico, p1, p2])
    rot1 = Rotation.from_quat(rng.normal(size=4))
    init_pts1 = rot1.apply(init_pts1)
    
    # Initial candidate 2: Fibonacci sphere (14 points)
    init_pts2 = make_fibonacci_sphere(14)
    rot2 = Rotation.from_quat(rng.normal(size=4))
    init_pts2 = rot2.apply(init_pts2)
    
    # Initial candidate 3: Icosahedron + 2 random points
    p3 = rng.normal(size=D)
    p3 = p3 / np.linalg.norm(p3)
    p4 = rng.normal(size=D)
    p4 = p4 / np.linalg.norm(p4)
    init_pts3 = np.vstack([ico, p3, p4])
    rot3 = Rotation.from_quat(rng.normal(size=4))
    init_pts3 = rot3.apply(init_pts3)
    
    initial_candidates = [init_pts1, init_pts2, init_pts3]
    
    # Phase 1: Global Exploration using Basin Hopping from multiple starts
    tau_bh = 20.0
    
    def bh_objective(x):
        return objective_log_ratio(x, tau=tau_bh)
    
    bh_minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'options': {'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
    }
    
    bh_results = []
    for i, init_pts in enumerate(initial_candidates):
        if time.time() - start_time > timeout - 60:
            break
        
        x0 = init_pts.flatten()
        bh_best_x = x0.copy()
        bh_best_f = bh_objective(x0)
        
        def bh_callback(x, f, accept):
            nonlocal bh_best_x, bh_best_f
            if f < bh_best_f:
                bh_best_f = f
                bh_best_x = x.copy()
            if time.time() - start_time > timeout - 50:
                raise Exception("Time to stop Basin Hopping")
        
        try:
            bh_res = basinhopping(
                bh_objective,
                x0,
                minimizer_kwargs=bh_minimizer_kwargs,
                stepsize=0.5,
                niter=50,
                callback=bh_callback,
                seed=42 + i
            )
            if bh_res.x is not None:
                bh_results.append(bh_res.x)
        except Exception:
            pass
    
    # Update best from all basin hopping results
    for res_x in bh_results:
        update_best(res_x.reshape(N, D))
    
    # Ensure we have a best point
    if best_points is None:
        best_points = initial_candidates[0]
        best_score = raw_score(best_points)
    
    # Phase 2: Local Refinement with Multi-Stage Tau
    x_current = best_points.flatten()
    
    tau_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    
    for tau in tau_schedule:
        if time.time() - start_time > timeout - 20:
            break
        try:
            res = minimize(
                objective_log_ratio, 
                x_current, 
                method='L-BFGS-B', 
                args=(tau,),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                x_current = res.x
        except Exception:
            pass
    
    # Phase 3: Try epigraph formulation for additional refinement
    if time.time() - start_time < timeout - 15:
        try:
            res = minimize(
                objective_epigraph,
                x_current,
                method='L-BFGS-B',
                args=(200.0,),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                x_current = res.x
        except Exception:
            pass
    
    # Phase 4: Final Exact Local Search using SLSQP on true objective
    if time.time() - start_time < timeout - 5:
        def exact_objective(x):
            pts = x.reshape(N, D)
            sq_dists = get_pairwise_sq_dists(pts)
            min_sq = np.min(sq_dists)
            if min_sq < 1e-12:
                return 1e10
            return np.max(sq_dists) / min_sq
        
        try:
            res = minimize(
                exact_objective, 
                x_current, 
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass

    # Phase 5: Try a few more random restarts if time remains
    restart_count = 0
    while time.time() - start_time < timeout - 5:
        x0_rand = rng.uniform(-1, 1, size=(N * D))
        # Quick local opt
        try:
            res = minimize(
                objective_log_ratio, 
                x0_rand, 
                method='L-BFGS-B', 
                args=(50.0,),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                
                # If this is good, do a high-precision polish
                if raw_score(pts) < best_score * 1.05 and time.time() - start_time < timeout - 3:
                    try:
                        res2 = minimize(
                            objective_log_ratio, 
                            res.x, 
                            method='L-BFGS-B', 
                            args=(1000.0,),
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
        best_points = initial_candidates[0]
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
