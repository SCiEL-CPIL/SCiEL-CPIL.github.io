import numpy as np
import time
from scipy.optimize import minimize
import itertools
from functools import partial

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the true (non-smooth) score."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) <= 1e-12:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def objective_log_ratio(x, tau=50.0):
    """
    Log-domain smooth approximation of log(max/min) of squared distances.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    log_dists = np.log(sq_dists)
    
    # Soft max in log domain
    m_max = np.max(log_dists)
    exp_max_terms = np.exp(tau * (log_dists - m_max))
    soft_max_log = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min in log domain
    m_min = np.min(log_dists)
    exp_min_terms = np.exp(-tau * (log_dists - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    return soft_max_log - soft_min_log

def objective_sqp(x, tau=100.0):
    """
    SQP-style objective: fix max distance to 1 (by scaling), maximize min distance.
    This removes the ratio's scale invariance.
    
    We normalize points so that the maximum pairwise squared distance is 1,
    then minimize -log(min_sq_dist) + penalty for any dist > 1.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Scale so max dist = 1
    max_d = np.max(sq_dists)
    if max_d < 1e-15:
        return 1e10
    scaled = sq_dists / max_d
    scaled = np.maximum(scaled, 1e-12)
    
    log_scaled = np.log(scaled)
    
    # Soft min of log distances (we want to maximize this)
    m_min = np.min(log_scaled)
    exp_min_terms = np.exp(-tau * (log_scaled - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # Penalty for any scaled distance > 1 (shouldn't happen after scaling, but smooth)
    # Actually after scaling max is exactly 1, so all are <= 1.
    # The objective is to maximize soft_min_log, so minimize -soft_min_log
    return -soft_min_log

def objective_epigraph(x, tau=50.0):
    """
    Epigraph-style smooth objective: minimize t subject to all sq_dists in [1/t, t].
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    log_dists = np.log(sq_dists)
    m_max = np.max(log_dists)
    m_min = np.min(log_dists)
    
    exp_max_terms = np.exp(tau * (log_dists - m_max))
    soft_max_log = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (log_dists - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    log_ratio = soft_max_log - soft_min_log
    
    min_penalty = 0.0
    if m_min < -20:
        min_penalty = (m_min + 20) ** 2
    
    return log_ratio + min_penalty

def fibonacci_sphere(n, radius=1.0):
    """Generate n points evenly distributed on a sphere using the Fibonacci lattice."""
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1)
        r = np.sqrt(1.0 - z**2)
        theta = golden_angle * i
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return points * radius

def icosahedron_points(radius=1.0):
    """Generate 12 icosahedron vertices."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    pts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms * radius
    return pts

def normalize_to_unit_max(points):
    """Center points and scale so max pairwise squared distance is 1."""
    pts = points - np.mean(points, axis=0)
    sq_dists = get_pairwise_sq_dists(pts)
    max_d = np.max(sq_dists)
    if max_d > 1e-15:
        pts = pts / np.sqrt(max_d)
    return pts

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    def try_update(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        s = raw_score(pts)
        if s < best_score:
            best_score = s
            best_points = pts.copy()
    
    # Initial candidates
    # 1. Grid
    points_grid = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                points_grid.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_points_grid = [p for p in points_grid if p not in to_remove]
    init_grid = np.array(final_points_grid, dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    
    # 2. Fibonacci sphere
    fib_pts = fibonacci_sphere(N, radius=2.0)
    
    # 3. Icosahedron + 2 extras
    ico_pts = icosahedron_points(radius=2.0)
    extra1 = np.array([0, 0, 2.0])
    extra2 = np.array([0, 0, -2.0])
    ico_extended = np.vstack([ico_pts, extra1.reshape(1,3), extra2.reshape(1,3)])
    ico_extended = ico_extended - np.mean(ico_extended, axis=0)
    
    # 4. Random starts
    rng = np.random.default_rng(42)
    
    candidates = [init_grid.flatten(), fib_pts.flatten(), ico_extended.flatten()]
    
    # Random sphere starts
    for i in range(15):
        if time.time() - start_time > timeout:
            break
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * rng.uniform(1.5, 3.0)
        candidates.append(pts.flatten())
    
    # Random ball starts
    for i in range(10):
        if time.time() - start_time > timeout:
            break
        pts = rng.normal(size=(N, D)) * 2.0
        candidates.append(pts.flatten())
    
    # Perturbed versions
    for base in [init_grid, fib_pts, ico_extended]:
        for _ in range(3):
            if time.time() - start_time > timeout:
                break
            perturbed = base + rng.normal(size=(N, D)) * 0.15
            candidates.append(perturbed.flatten())
    
    # Evaluate initial candidates
    for cand in candidates:
        try_update(cand.reshape(N, D))
    
    # Two optimization strategies:
    # Strategy A: Log-ratio (original)
    # Strategy B: SQP-style (normalize to unit max, maximize min)
    
    tau_schedule_A = [10.0, 30.0, 100.0, 300.0, 800.0]
    maxiter_schedule_A = [150, 200, 300, 400, 500]
    
    tau_schedule_B = [20.0, 50.0, 200.0, 500.0, 1500.0]
    maxiter_schedule_B = [150, 200, 300, 400, 500]
    
    # Optimize each candidate with both strategies
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 5:
            break
            
        # --- Strategy A: Log-ratio ---
        x_curr_a = init_x.copy()
        try:
            for idx, tau in enumerate(tau_schedule_A):
                if time.time() - start_time > timeout - 5:
                    break
                obj = partial(objective_log_ratio, tau=tau)
                res = minimize(
                    obj, x_curr_a,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter_schedule_A[idx], 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                if res.x is not None and np.isfinite(res.fun):
                    x_curr_a = res.x
            try_update(x_curr_a.reshape(N, D))
        except Exception:
            pass
        
        # --- Strategy B: SQP-style (normalize, maximize min) ---
        if time.time() - start_time > timeout - 10:
            continue
            
        # Start from normalized version
        pts_init_b = normalize_to_unit_max(init_x.reshape(N, D))
        x_curr_b = pts_init_b.flatten()
        try:
            for idx, tau in enumerate(tau_schedule_B):
                if time.time() - start_time > timeout - 10:
                    break
                obj = partial(objective_sqp, tau=tau)
                res = minimize(
                    obj, x_curr_b,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter_schedule_B[idx], 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                if res.x is not None and np.isfinite(res.fun):
                    x_curr_b = res.x
            # After SQP optimization, re-normalize and evaluate
            pts_b = normalize_to_unit_max(x_curr_b.reshape(N, D))
            try_update(pts_b)
        except Exception:
            pass
    
    # Final polish on the best point found
    if best_points is not None and time.time() - start_time < timeout - 10:
        x_polish = best_points.flatten().copy()
        try:
            # High-tau log-ratio polish
            obj_final = partial(objective_log_ratio, tau=2000.0)
            res_polish = minimize(
                obj_final, x_polish,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_polish.x is not None and np.isfinite(res_polish.fun):
                try_update(res_polish.x.reshape(N, D))
        except Exception:
            pass
        
        # SLSQP polish
        if time.time() - start_time < timeout - 5:
            try:
                x_sl = best_points.flatten().copy()
                obj_slsqp = partial(objective_log_ratio, tau=3000.0)
                res_slsqp = minimize(
                    obj_slsqp, x_sl,
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                if res_slsqp.x is not None and np.isfinite(res_slsqp.fun):
                    try_update(res_slsqp.x.reshape(N, D))
            except Exception:
                pass
    
    # Fallback
    if best_points is None:
        best_points = init_grid
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
