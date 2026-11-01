import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_smooth(x, tau=20.0):
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def objective_log_ratio(x, tau=50.0):
    """
    Direct log-ratio objective with soft min/max.
    Returns log(max/min) in squared distance space.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m = np.max(sq_dists)
    log_soft_max = m + np.log(np.sum(np.exp(tau * (sq_dists - m)))) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    log_soft_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    return log_soft_max - log_soft_min

def anneal_optimize(x0, start_time, timeout, tau_start=10.0, tau_end=100.0, n_stages=5):
    """
    Multi-stage optimization with tau annealing.
    """
    elapsed = time.time() - start_time
    remaining = timeout - elapsed
    if remaining < 2.0:
        return x0
    
    x = x0.copy()
    # Log-spaced tau values
    taus = np.logspace(np.log10(tau_start), np.log10(tau_end), n_stages)
    
    for tau in taus:
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        if remaining < 2.0:
            break
        
        # Allocate iterations based on remaining time
        maxiter = max(100, int(remaining * 50))
        
        try:
            res = minimize(
                objective_log_ratio,
                x,
                method='L-BFGS-B',
                args=(tau,),
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                x = res.x
        except Exception:
            pass
    
    return x

def generate_candidates():
    """
    Generate diverse initial candidates.
    """
    candidates = []
    rng = np.random.default_rng(42)
    
    # Candidate 1: Icosahedron-based (12 vertices) + 2 interior points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * 2.0
    # Add 2 points: one near center, one slightly off
    p1 = np.array([0.3, 0.3, 0.3])
    p2 = np.array([-0.2, -0.2, -0.2])
    pts1 = np.vstack([ico, p1, p2])
    candidates.append(pts1.flatten())
    
    # Candidate 2: Random on sphere
    for seed in [1, 2, 3, 4, 5]:
        rng_tmp = np.random.default_rng(seed)
        pts = rng_tmp.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
    
    # Candidate 3: Cube face centers + corners
    # 6 face centers of cube [-1,1]^3
    face_centers = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ], dtype=float) * 1.5
    # 8 corners
    corners = np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]], dtype=float) * 1.2
    # Pick 6 from corners to get 14 total
    corners_subset = corners[:8]  # all 8 corners
    pts3 = np.vstack([face_centers, corners_subset[:8]])
    candidates.append(pts3.flatten())
    
    # Candidate 4: Two tetrahedra (dual arrangement)
    # Regular tetrahedron vertices
    tet1 = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float) / np.sqrt(3) * 2.0
    
    # Dual tetrahedron (rotated)
    tet2 = np.array([
        [-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]
    ], dtype=float) / np.sqrt(3) * 2.0
    
    # 8 points from two tetrahedra, add 6 more
    # Add points at mid-edges or other symmetric positions
    extra = np.array([
        [2.0, 0, 0], [-2.0, 0, 0],
        [0, 2.0, 0], [0, -2.0, 0],
        [0, 0, 2.0], [0, 0, -2.0]
    ], dtype=float)
    pts4 = np.vstack([tet1, tet2, extra])
    candidates.append(pts4.flatten())
    
    # Candidate 5: Random with varying radii
    for seed in [10, 20, 30]:
        rng_tmp = np.random.default_rng(seed)
        pts = rng_tmp.normal(size=(N, D))
        # Vary radii to create shell structure
        radii = rng_tmp.uniform(1.5, 2.5, size=(N, 1))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * radii
        candidates.append(pts.flatten())
    
    # Candidate 6: Grid-like structure
    grid_pts = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if len(grid_pts) < N:
                    grid_pts.append([x-1, y-1, z-1])
    pts_grid = np.array(grid_pts, dtype=float) * 1.5
    candidates.append(pts_grid.flatten())
    
    return candidates

def construct():
    start_time = time.time()
    timeout = 142.0  # Safety margin: 8s before 150s limit
    
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
    
    # Generate initial candidates
    candidates = generate_candidates()
    
    # Phase 1: Anneal-optimize each candidate
    for i, x0 in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > timeout - 30:
            break
        
        x_opt = anneal_optimize(x0, start_time, timeout, tau_start=5.0, tau_end=80.0, n_stages=4)
        update_best(x_opt.reshape(N, D))
    
    # Phase 2: Differential Evolution with time check
    elapsed = time.time() - start_time
    if elapsed < timeout - 25:
        bounds = [(-4, 4)] * (N * D)
        try:
            # Use fewer iterations but larger population for better exploration
            res_de = differential_evolution(
                raw_score,
                bounds,
                maxiter=30,
                popsize=12,
                seed=42,
                tol=1e-5,
                polish=False,
                init='sobol'
            )
            if res_de.x is not None:
                # Polish with annealing
                x_polished = anneal_optimize(res_de.x, start_time, timeout, tau_start=10.0, tau_end=100.0, n_stages=3)
                update_best(x_polished.reshape(N, D))
        except Exception:
            pass
    
    # Phase 3: Basin hopping from best point
    elapsed = time.time() - start_time
    if elapsed < timeout - 10 and best_points is not None:
        try:
            rng_bh = np.random.default_rng(123)
            
            def step(x):
                delta = rng_bh.normal(size=x.shape) * 0.15
                return x + delta
            
            bh = basinhopping(
                objective_log_ratio,
                best_points.flatten(),
                step=step,
                minimizer_kwargs={'method': 'L-BFGS-B', 'args': (50.0,), 'options': {'maxiter': 200}},
                niter=15,
                seed=123,
                T=1.0,
                stepsize=0.5
            )
            if bh.x is not None:
                # Final polish
                x_final = anneal_optimize(bh.x, start_time, timeout, tau_start=20.0, tau_end=120.0, n_stages=3)
                update_best(x_final.reshape(N, D))
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish if time remains
    elapsed = time.time() - start_time
    if elapsed < timeout - 3 and best_points is not None:
        try:
            x_final = anneal_optimize(best_points.flatten(), start_time, timeout, tau_start=50.0, tau_end=200.0, n_stages=3)
            update_best(x_final.reshape(N, D))
        except Exception:
            pass
    
    if best_points is None:
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
