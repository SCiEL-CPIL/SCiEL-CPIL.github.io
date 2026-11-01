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

def objective_smooth(x, tau=50.0, scale_reg=0.01):
    """
    Smooth approximation of the log ratio of max to min squared distance.
    Includes a small regularization to fix the scale (sum of squared norms).
    """
    points = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Compute squared distances
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    # Numerical stability for log-sum-exp
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # Objective is log(max) - log(min)
    obj = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    # Regularization to fix scale: penalize deviation of total squared norm from 1
    # This helps the optimizer converge by removing the flat scaling direction
    total_sq_norm = np.sum(points**2)
    reg = scale_reg * (total_sq_norm - 1.0)**2
    
    return obj + reg

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
        # Center and normalize for consistent scoring? No, raw_score is scale invariant.
        # But we should check validity.
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 points, maybe at the center? No, distinct points.
    # Try placing them symmetrically.
    p_extra1 = np.array([0, 0, 0]) # Center is a good candidate for one point
    p_extra2 = np.array([1, 0, 0]) # Or another direction
    
    # Actually, for 14 points, a known good configuration might be related to the 12-icosahedron 
    # with 2 points added. Let's just use random perturbations of ico + 2 random points on sphere.
    
    for i in range(5):
        pts = ico.copy()
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        # Scale ico slightly to make room? Or just append.
        # The optimizer will adjust positions.
        full_pts = np.vstack([pts, p1, p2])
        # Random rotation
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # 2. Random spherical points
    for i in range(10):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        # Random rotation
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 3. Uniform random in cube
    for i in range(5):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    # Optimization Loop
    n_restarts = len(candidates)
    
    # We will do multiple passes with increasing tau
    taus = [10.0, 50.0, 200.0, 500.0]
    
    # First, do a quick L-BFGS with low tau on all candidates
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 5:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(taus[0],),
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                # Save the refined point for next stages
                candidates[i] = res.x
        except Exception:
            pass

    # Second, do a more precise L-BFGS with higher tau on the best candidates found so far
    # We sort candidates by their current score and pick the top ones
    # But we don't have scores stored separately. We can just re-run on all candidates with higher tau.
    # To save time, we can just run on all again.
    
    for tau in taus[1:]:
        if time.time() - start_time > timeout - 5:
            break
        for i in range(n_restarts):
            if time.time() - start_time > timeout - 2:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_smooth, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # If we still have time, do a few random restarts with high tau
    while time.time() - start_time < timeout - 5:
        x0 = rng.uniform(-1, 1, size=(N * D))
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(taus[-1],),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
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
