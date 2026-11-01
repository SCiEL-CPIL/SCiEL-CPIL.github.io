import numpy as np
import time
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the true (non-smooth) score: max/min squared distance."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def objective_smooth(x, tau=10.0):
    """
    Smooth log-ratio objective using soft-max and soft-min.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max and soft min via log-sum-exp
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    soft_min = np.maximum(soft_min, 1e-15)
    
    return np.log(soft_max) - np.log(soft_min)

def fibonacci_sphere(n_points, radius=2.0):
    """Generate n_points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        r = np.sqrt(1.0 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])
    return np.array(points)

def generate_4d_polytope_vertices():
    """
    Generate vertices of a 4D polytope with good distance distribution.
    Returns an array of shape (n_vertices, 4) where n_vertices >= N.
    We'll select N vertices from these.
    """
    # 16-cell (cross-polytope) vertices: ±e_i in 4D → 8 vertices
    # Not enough. Let's use 24-cell or combine.
    
    # 24-cell vertices: all permutations of (±1, ±1, 0, 0) → 24 vertices
    verts_24 = []
    for i in range(4):
        for j in range(i+1, 4):
            for si in [1.0, -1.0]:
                for sj in [1.0, -1.0]:
                    v = [0.0, 0.0, 0.0, 0.0]
                    v[i] = si
                    v[j] = sj
                    verts_24.append(v)
    
    # Also add 16-cell vertices (±e_i) for more options
    verts_16 = []
    for i in range(4):
        for s in [1.0, -1.0]:
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            verts_16.append(v)
    
    # Combine and normalize
    all_verts = np.array(verts_24 + verts_16, dtype=float)
    norms = np.linalg.norm(all_verts, axis=1, keepdims=True)
    all_verts = all_verts / norms
    return all_verts

def optimize_projection(vertices_4d, indices, init_matrix=None):
    """
    Optimize a 3x4 projection matrix P such that P @ vertices_4d[indices] 
    minimizes the max/min squared distance ratio.
    Returns (best_points_3d, best_score).
    """
    N_sel = len(indices)
    
    def proj_objective(params):
        P = params.reshape(3, 4)
        pts = vertices_4d[indices] @ P.T  # (N_sel, 3)
        # Center
        pts = pts - pts.mean(axis=0)
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-15:
            return 1e10
        return np.max(sq_dists) / np.min(sq_dists)
    
    def proj_objective_smooth(params, tau=50.0):
        P = params.reshape(3, 4)
        pts = vertices_4d[indices] @ P.T
        pts = pts - pts.mean(axis=0)
        sq_dists = get_pairwise_sq_dists(pts)
        sq_dists = np.maximum(sq_dists, 1e-15)
        m_max = np.max(sq_dists)
        m_min = np.min(sq_dists)
        exp_max_terms = np.exp(tau * (sq_dists - m_max))
        soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
        exp_min_terms = np.exp(-tau * (sq_dists - m_min))
        soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
        soft_min = np.maximum(soft_min, 1e-15)
        return np.log(soft_max) - np.log(soft_min)
    
    best_score = np.inf
    best_pts = None
    
    # Multiple random initializations for the projection
    rng = np.random.default_rng(12345)
    for trial in range(5):
        if init_matrix is not None and trial == 0:
            x0 = init_matrix.flatten().copy()
        else:
            x0 = rng.normal(size=(3, 4))
            x0 = x0 / np.linalg.norm(x0)
        
        try:
            # Progressive tau
            for tau in [10.0, 50.0, 200.0, 1000.0]:
                obj = lambda p, t=tau: proj_objective_smooth(p, tau=t)
                res = minimize(obj, x0, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None and np.isfinite(res.fun):
                    x0 = res.x
            
            # Evaluate true score
            P = x0.reshape(3, 4)
            pts = vertices_4d[indices] @ P.T
            pts = pts - pts.mean(axis=0)
            s = raw_score(pts)
            if np.isfinite(s) and s < best_score:
                best_score = s
                best_pts = pts.copy()
                
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Grid-based initialization
    grid_pts = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                grid_pts.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_grid = [p for p in grid_pts if p not in to_remove]
    grid_pts_arr = np.array(final_grid, dtype=float)
    grid_pts_arr = grid_pts_arr - np.mean(grid_pts_arr, axis=0)
    candidates.append(grid_pts_arr.flatten())
    
    # 2. Fibonacci sphere points
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 3. 4D polytope projection approach
    try:
        verts_4d = generate_4d_polytope_vertices()
        n_4d = len(verts_4d)
        
        # Try different subsets of 14 vertices from the 4D polytope
        # Use combinations or random subsets
        all_indices = list(range(n_4d))
        
        # Try several random subsets
        for subset_seed in range(8):
            if time.time() - start_time > timeout - 30:
                break
            r_sub = np.random.default_rng(500 + subset_seed)
            indices = r_sub.choice(n_4d, size=N, replace=False)
            
            proj_pts, proj_score = optimize_projection(verts_4d, indices)
            if proj_pts is not None:
                candidates.append(proj_pts.flatten())
                if np.isfinite(proj_score) and proj_score < best_score:
                    best_score = proj_score
                    best_points = proj_pts.copy()
        
        # Also try all 24 vertices of the 24-cell (first 14)
        indices_24 = list(range(24))  # First 24 are 24-cell
        if len(indices_24) >= N:
            proj_pts, proj_score = optimize_projection(verts_4d, indices_24[:N])
            if proj_pts is not None:
                candidates.append(proj_pts.flatten())
                if np.isfinite(proj_score) and proj_score < best_score:
                    best_score = proj_score
                    best_points = proj_pts.copy()
                    
    except Exception:
        pass
    
    # 4. Random points on sphere with different seeds
    for seed in range(10):
        r = np.random.default_rng(seed)
        pts = r.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout - 40:
            break
    
    # 5. Random points in ball
    for seed in range(5):
        r = np.random.default_rng(100 + seed)
        pts = r.normal(size=(N, D)) * 2.0
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout - 40:
            break
    
    # 6. Perturbed Fibonacci
    for _ in range(3):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.2
        candidates.append(perturbed.flatten())
    
    # 7. Perturbed grid
    for _ in range(3):
        perturbed = grid_pts_arr + rng.normal(size=(N, D)) * 0.1
        candidates.append(perturbed.flatten())
    
    # Optimization with progressive tau schedule
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0]
    maxiter_schedule = [200, 300, 400, 500, 600, 800]
    
    # Optimization loop over candidates
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 15:
            break
        
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout - 10:
                    break
                
                obj = partial(objective_smooth, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                # Use L-BFGS-B for the main optimization
                res = minimize(
                    obj,
                    x_curr,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
                )
                
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            
            # Final high-precision polish with SLSQP at very high tau
            if time.time() - start_time < timeout - 5.0:
                try:
                    obj_final = partial(objective_smooth, tau=10000.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='SLSQP',
                        options={'maxiter': 300, 'ftol': 1e-16, 'maxls': 50}
                    )
                    if res_polish.x is not None and np.isfinite(res_polish.fun):
                        x_curr = res_polish.x
                except Exception:
                    pass
            
            # Evaluate the true non-smooth score
            pts = x_curr.reshape(N, D)
            true_s = raw_score(pts)
            if np.isfinite(true_s) and true_s < best_score:
                best_score = true_s
                best_points = pts.copy()
                
        except Exception:
            pass
    
    # If optimization didn't find anything better, fall back to the best candidate
    if best_points is None:
        for init_x in candidates:
            pts = init_x.reshape(N, D)
            s = raw_score(pts)
            if np.isfinite(s) and s < best_score:
                best_score = s
                best_points = pts.copy()
        
        if best_points is None:
            best_points = grid_pts_arr
            best_score = raw_score(best_points)

    # Final validation: ensure all points are distinct
    sq_dists = get_pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-8:
        best_points = grid_pts_arr
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
