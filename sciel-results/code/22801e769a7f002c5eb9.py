import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Calculate squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def objective_smooth(x, tau=100.0):
    """
    Smoothed objective function using log-sum-exp for max and min.
    Minimizes log(max_sq_dist) - log(min_sq_dist).
    Includes centering and normalization to remove translation and scale invariance.
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

def d3d_symmetric_points(r1, r2, r3):
    """
    Generate 14 points using a symmetric configuration:
    - 6 points on a hexagonal ring 1 at height -r3
    - 6 points on a hexagonal ring 2 at height +r3 (rotated by 30 degrees)
    - 2 points on the z-axis at heights ±r3*1.5
    
    This structure enforces 6-fold rotational symmetry (modulo the offset) and reflection symmetry.
    """
    # 6 points on hexagonal ring 1 at height -r3
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    pts1 = np.column_stack([
        r1 * np.cos(angles1),
        r1 * np.sin(angles1),
        np.full(6, -r3)
    ])
    
    # 6 points on hexagonal ring 2 at height +r3
    # Offset by pi/6 (30 degrees) to interlock
    angles2 = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False)
    pts2 = np.column_stack([
        r2 * np.cos(angles2),
        r2 * np.sin(angles2),
        np.full(6, r3)
    ])
    
    # 2 points on z-axis
    pts3 = np.array([[0, 0, -r3*1.5], [0, 0, r3*1.5]])
    
    return np.vstack([pts1, pts2, pts3])

def d3d_objective(params):
    """Objective for D_{3d} symmetric configuration."""
    r1, r2, r3 = params
    if r1 < 0 or r2 < 0 or r3 < 0:
        return 1e10
    pts = d3d_symmetric_points(r1, r2, r3)
    return raw_score(pts)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
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
    
    # --- Strategy 0: D_{3d} symmetry-constrained optimization (3D search space) ---
    # This is the primary strategy per the directive
    
    # Grid search over the 3D parameter space
    r1_vals = np.linspace(0.5, 2.0, 15)
    r2_vals = np.linspace(0.5, 2.0, 15)
    r3_vals = np.linspace(0.2, 1.5, 12)
    
    best_sym_score = np.inf
    best_sym_params = None
    
    for r1 in r1_vals:
        if time.time() - start_time > timeout - 60:
            break
        for r2 in r2_vals:
            for r3 in r3_vals:
                try:
                    score = d3d_objective([r1, r2, r3])
                    if score < best_sym_score:
                        best_sym_score = score
                        best_sym_params = [r1, r2, r3]
                except Exception:
                    pass
    
    # Refine with Nelder-Mead around the best grid point
    if best_sym_params is not None:
        try:
            res_sym = minimize(
                d3d_objective,
                best_sym_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_sym.x is not None:
                pts_sym = d3d_symmetric_points(*res_sym.x)
                update_best(pts_sym)
        except Exception:
            pass
    
    # Also try several random initial points for Nelder-Mead
    for _ in range(20):
        if time.time() - start_time > timeout - 40:
            break
        r1_init = rng.uniform(0.5, 2.0)
        r2_init = rng.uniform(0.5, 2.0)
        r3_init = rng.uniform(0.2, 1.5)
        try:
            res_rand = minimize(
                d3d_objective,
                [r1_init, r2_init, r3_init],
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_rand.x is not None:
                pts_rand = d3d_symmetric_points(*res_rand.x)
                update_best(pts_rand)
        except Exception:
            pass

    # --- Strategy 1: Coordinate-space optimization (original approach) ---
    
    # Initial candidates
    # 1. Icosahedron + center + 1 point
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    center = np.array([0.0, 0.0, 0.0])
    v = rng.normal(size=3)
    v = v / np.linalg.norm(v)
    p14 = v * 1.0
    
    init_ico = np.vstack([ico, center, p14])
    coord_candidates = [init_ico.flatten()]
    
    # 2. Random points on a sphere
    for i in range(3):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        coord_candidates.append(pts.flatten())
        
    # 3. Random points in a cube
    for i in range(3):
        pts = rng.uniform(-1, 1, size=(N, D))
        coord_candidates.append(pts.flatten())

    # 4. Fibonacci sphere
    def fibonacci_sphere(n):
        points = []
        golden_angle = np.pi * (3. - np.sqrt(5.))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    fib = fibonacci_sphere(N)
    coord_candidates.append(fib.flatten())
    
    # 5. Two rings of 7 points each (from inspiration)
    for i in range(3):
        ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        ring2_angles = ring1_angles + np.pi/7
        ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.zeros(7)])
        ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, 0.5)])
        pts = np.vstack([ring1, ring2])
        coord_candidates.append(pts.flatten())

    # Coordinate-space optimization
    # Use a tau schedule for progressive refinement
    tau_schedule = [10.0, 50.0, 200.0, 500.0]
    
    n_coord_restarts = 50
    for i in range(n_coord_restarts):
        if time.time() - start_time > timeout - 20:
            break
            
        if i < len(coord_candidates):
            x0 = coord_candidates[i]
        else:
            x0 = rng.uniform(-1, 1, size=(N * D))
            
        try:
            # Multi-stage optimization
            # Stage 1: Low tau
            res1 = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[0],),
                options={'maxiter': 80, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            if res1.x is not None:
                pts1 = res1.x.reshape(N, D)
                update_best(pts1)
                
                if time.time() - start_time < timeout - 15:
                    # Stage 2: Medium tau
                    res2 = minimize(
                        objective_smooth, 
                        res1.x, 
                        method='L-BFGS-B', 
                        args=(tau_schedule[1],),
                        options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    if res2.x is not None:
                        pts2 = res2.x.reshape(N, D)
                        update_best(pts2)
                        
                        if time.time() - start_time < timeout - 10:
                            # Stage 3: High tau
                            res3 = minimize(
                                objective_smooth, 
                                res2.x, 
                                method='L-BFGS-B', 
                                args=(tau_schedule[2],),
                                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                            )
                            if res3.x is not None:
                                pts3 = res3.x.reshape(N, D)
                                update_best(pts3)
                                
        except Exception:
            pass

    # --- Final polish: use the best points found and do a high-precision coordinate optimization ---
    if best_points is not None and time.time() - start_time < timeout - 2:
        try:
            res_final = minimize(
                objective_smooth, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                args=(tau_schedule[3],),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                update_best(pts_final)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = coord_candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
