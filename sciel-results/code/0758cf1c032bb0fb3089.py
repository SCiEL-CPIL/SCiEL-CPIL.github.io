import numpy as np
import time
from scipy.optimize import minimize
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
    Uses centering and normalization to remove translation and scale invariance.
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

def objective_epigraph(x, tau=50.0):
    """
    Epigraph formulation: minimize t subject to constraints.
    Uses smooth approximation of constraints.
    x = [points_flat, t] where t is the log ratio.
    """
    n_params = N * D
    points = x[:n_params].reshape(N, D)
    t = x[n_params]
    
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
    
    # Soft max of log(squared distances)
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log(squared distances)
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    # Penalty if soft_max - soft_min > t
    violation = soft_max_log - soft_min_log - t
    penalty = np.maximum(0, violation) ** 2
    
    return t + 100.0 * penalty

def fibonacci_sphere(n, rng=None):
    """Generate n points on a sphere using Fibonacci lattice."""
    if rng is None:
        rng = np.random.default_rng(42)
    points = []
    golden_angle = np.pi * (3. - np.sqrt(5.))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    pts = np.array(points)
    # Apply random rotation
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def generate_icosahedron():
    """Generate 12 icosahedron vertices on unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    return ico

def generate_initial_guesses(rng, n_guesses=15):
    """Generate diverse initial configurations."""
    guesses = []
    
    # 1. Fibonacci sphere
    fib = fibonacci_sphere(N, rng)
    guesses.append(fib.flatten())
    
    # 2. Icosahedron + center + random point
    ico = generate_icosahedron()
    center = np.array([0.0, 0.0, 0.0])
    v = rng.normal(size=3)
    v = v / np.linalg.norm(v)
    p14 = v * 1.0
    ico_plus = np.vstack([ico, center, p14])
    guesses.append(ico_plus.flatten())
    
    # 3. Icosahedron + 2 random points
    for _ in range(2):
        v1 = rng.normal(size=3)
        v1 = v1 / np.linalg.norm(v1)
        v2 = rng.normal(size=3)
        v2 = v2 / np.linalg.norm(v2)
        ico_plus2 = np.vstack([ico, v1 * 0.8, v2 * 0.8])
        guesses.append(ico_plus2.flatten())
    
    # 4. Random points on sphere
    for _ in range(n_guesses - 6):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        guesses.append(pts.flatten())
    
    # 5. Random points in cube
    for _ in range(3):
        pts = rng.uniform(-1, 1, size=(N, D))
        guesses.append(pts.flatten())
    
    return guesses

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
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
    
    def time_remaining():
        return timeout - (time.time() - start_time)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial guesses
    initial_guesses = generate_initial_guesses(rng, n_guesses=15)
    
    # Multi-stage optimization with increasing tau
    tau_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    for guess_idx, x0 in enumerate(initial_guesses):
        if time_remaining() < 10:
            break
        
        x_current = x0.copy()
        
        for tau in tau_schedule:
            if time_remaining() < 5:
                break
            
            try:
                res = minimize(
                    objective_log_ratio,
                    x_current,
                    method='L-BFGS-B',
                    args=(tau,),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    x_current = res.x
                    pts = x_current.reshape(N, D)
                    update_best(pts)
            except Exception:
                break
        
        # SLSQP polish at high tau if time permits
        if time_remaining() > 8 and x_current is not None:
            try:
                res_slsqp = minimize(
                    objective_log_ratio,
                    x_current,
                    method='SLSQP',
                    args=(10000.0,),
                    options={'maxiter': 100, 'ftol': 1e-12, 'eps': 1e-8}
                )
                if res_slsqp.x is not None:
                    pts_slsqp = res_slsqp.x.reshape(N, D)
                    update_best(pts_slsqp)
                    x_current = res_slsqp.x
            except Exception:
                pass
    
    # Final high-precision polish on the best points found
    if best_points is not None and time_remaining() > 3:
        try:
            res_final = minimize(
                objective_log_ratio,
                best_points.flatten(),
                method='L-BFGS-B',
                args=(10000.0,),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                update_best(pts_final)
        except Exception:
            pass
    
    # Another SLSQP polish if time permits
    if best_points is not None and time_remaining() > 2:
        try:
            res_final2 = minimize(
                objective_log_ratio,
                best_points.flatten(),
                method='SLSQP',
                args=(50000.0,),
                options={'maxiter': 50, 'ftol': 1e-14, 'eps': 1e-10}
            )
            if res_final2.x is not None:
                pts_final2 = res_final2.x.reshape(N, D)
                update_best(pts_final2)
        except Exception:
            pass

    if best_points is None:
        # Fallback: return Fibonacci sphere
        best_points = fibonacci_sphere(N, rng)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
