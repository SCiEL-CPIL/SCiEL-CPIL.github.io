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

def objective_sqp(x, mu=1.0):
    """
    Sequential Quadratic Programming style objective.
    Maximizes the minimum squared distance subject to max squared distance <= 1.
    Formulation: min -t + mu * (max(0, max_sq_dist - 1))^2 + mu * (max(0, -t))^2
    where t is the minimum squared distance.
    x = [points_flat, t]
    """
    n_params = N * D
    points = x[:n_params].reshape(N, D)
    t = x[n_params]
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Compute squared distances
    sq_dists = get_pairwise_sq_dists(points)
    
    # Normalize to fix scale: make max squared distance = 1
    max_sq_dist = np.max(sq_dists)
    if max_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(max_sq_dist)
    
    # Recompute squared distances after normalization
    sq_dists = get_pairwise_sq_dists(points)
    
    # Now max_sq_dist should be approx 1
    # We want to maximize t (min sq dist) subject to max sq dist <= 1.
    # Penalty if max sq dist > 1 (should be 0 after norm, but numerical errors)
    # Penalty if t > min sq dist (t should be <= min sq dist)
    
    min_sq_dist = np.min(sq_dists)
    
    # Constraint 1: max_sq_dist <= 1 (normalized, so this is soft)
    # Actually, since we normalized by max, max is 1. 
    # The real constraint is that we want to maximize the min distance.
    # Let's use a barrier/penalty for t > min_sq_dist.
    # If t > min_sq_dist, it's infeasible.
    # If t < min_sq_dist, it's feasible but we want to increase t.
    
    # Objective: -t + penalty(t > min_sq_dist)
    # This encourages t to be as large as possible, up to min_sq_dist.
    
    violation = np.maximum(0, t - min_sq_dist)
    penalty = mu * violation**2
    
    return -t + penalty

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
    
    # Tau schedule for log-ratio objective
    tau_schedule = [10.0, 50.0, 200.0, 500.0]
    
    # Phase 1: Optimize with log-ratio objective
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 15:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[0],),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 2: Progressive tau increase for log-ratio
    for tau_idx in range(1, len(tau_schedule)):
        if time.time() - start_time > timeout - 10:
            break
        tau = tau_schedule[tau_idx]
        for i in range(n_restarts):
            if time.time() - start_time > timeout - 5:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_log_ratio, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 3: SQP-style optimization on the best candidates
    # We take the top 5 candidates based on current best_score logic (just take first few)
    # Actually, let's just run SQP on all candidates, but with a tighter time budget per candidate
    # Or just on the ones that improved.
    
    # Let's create a list of candidates for SQP
    sqp_candidates = []
    for i in range(min(10, len(candidates))):
        sqp_candidates.append(candidates[i])
    
    # Initial t guess: min sq dist of the candidate
    for i in range(len(sqp_candidates)):
        if time.time() - start_time > timeout - 5:
            break
        
        x0_pts = sqp_candidates[i].reshape(N, D)
        sq_dists = get_pairwise_sq_dists(x0_pts)
        t0 = np.min(sq_dists)
        
        # Construct x = [points, t]
        x0 = np.concatenate([x0_pts.flatten(), [t0]])
        
        try:
            # Use SLSQP which can handle constraints, but we are using penalty method in objective
            # L-BFGS-B is good for unconstrained.
            res = minimize(
                objective_sqp, 
                x0, 
                method='L-BFGS-B', 
                args=(1.0,),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x[:N*D].reshape(N, D)
                update_best(pts)
        except Exception:
            pass

    # Phase 4: Differential Evolution for global exploration
    if time.time() - start_time < timeout - 20:
        bounds = [(-1.0, 1.0)] * (N * D)
        de_tau = 10.0
        
        def de_objective(x):
            return objective_log_ratio(x, tau=de_tau)
        
        try:
            de_result = differential_evolution(
                de_objective, 
                bounds, 
                seed=42,
                maxiter=50,
                popsize=15,
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
                        args=(500.0,),
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
                args=(1000.0,),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
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
