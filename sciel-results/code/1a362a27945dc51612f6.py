import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin for 630s limit

    def score(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_log_range(pts, beta):
        """
        Soft approximation of -log(d_max/d_min) = log(d_min) - log(d_max).
        We minimize this, which is equivalent to maximizing d_min/d_max.
        Using soft-min and soft-max for smoothness.
        """
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        # Avoid log(0)
        dists = np.maximum(dists, 1e-12)
        
        # Soft min: log-sum-exp trick for min
        # min(x) = -logsumexp(-x)
        m = np.min(dists)
        soft_min = m - np.log(np.sum(np.exp(-(dists - m))))
        
        # Soft max: logsumexp(x)
        M = np.max(dists)
        soft_max = M + np.log(np.sum(np.exp(dists - M)))
        
        # We want to minimize (soft_max - soft_min) / beta ?
        # Actually, let's minimize (soft_max - soft_min). 
        # As beta increases, we sharpen.
        # A common soft-min is: -1/beta * log(sum(exp(-beta*x)))
        # Let's use that.
        
        # Soft min of dists
        bm = beta * dists
        bm_max = np.max(bm)
        soft_min_val = -(1/beta) * np.log(np.sum(np.exp(-(bm - bm_max))))
        
        # Soft max of dists
        bm_max2 = np.max(bm)
        soft_max_val = (1/beta) * np.log(np.sum(np.exp(bm - bm_max2)))
        
        # Objective: minimize soft_max - soft_min
        # This is proportional to log(d_max/d_min)
        return soft_max_val - soft_min_val

    def optimize_from_seed(pts_init, beta_start=10, beta_end=1000, steps=200):
        pts = pts_init.copy()
        pts = pts - np.mean(pts, axis=0)
        
        # Normalize to unit max dist initially? 
        # The objective is scale invariant, so normalization helps with numerical stability.
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max > 0:
            pts = pts / d_max
        
        best_pts = pts.copy()
        best_score = score(pts)
        
        # L-BFGS-B optimization with annealed beta
        # We will run a few iterations, increasing beta
        num_annealing_steps = 10
        betas = np.geomspace(beta_start, beta_end, num_annealing_steps)
        
        current_pts = pts.copy()
        
        for b in betas:
            if time.time() > deadline:
                break
            
            # Define objective for scipy
            def obj(x):
                p = x.reshape(n, d)
                return soft_log_range(p, b)
            
            # Gradient is not provided, so L-BFGS-B will use numerical differentiation
            # Or we can provide a gradient? Numerical is okay for 42 dims.
            
            res = minimize(
                obj, 
                current_pts.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                new_pts = res.x.reshape(n, d)
                new_pts = new_pts - np.mean(new_pts, axis=0)
                new_score = score(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                
                # Update current position for next annealing step
                # Sometimes it's better to keep the best, sometimes the last
                current_pts = new_pts
        
        return best_pts, best_score

    best_overall_pts = None
    best_overall_score = 0

    # Generate diverse seeds
    seeds = []
    
    # 1. Icosahedron vertices (12) + 2 extra
    phi = (1 + np.sqrt(5)) / 2
    icosa = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ])
    # Add 2 points near center or perturbed
    seeds.append(np.vstack([icosa, [[0, 0, 0]], [[0.5, 0.5, 0.5]]]))
    
    # 2. Cube vertices (8) + Face centers (6)
    verts = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], dtype=float)
    faces = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    seeds.append(np.vstack([verts, faces]))
    
    # 3. Random points on sphere
    np.random.seed(42)
    for i in range(5):
        pts = np.random.randn(n, d)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        seeds.append(pts)
        
    # 4. Random points in cube
    for i in range(5):
        pts = np.random.rand(n, d) * 2 - 1
        seeds.append(pts)

    # Run optimization on each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
        
        pts, sc = optimize_from_seed(seed)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall_pts = pts
        
        # If we have time, do a local search refinement with simulated annealing
        # on the best points found so far?
        # Or just run more L-BFGS-B steps?
        
        # Let's add a simple local jitter refinement if time permits
        if time.time() < deadline - 30:
            # Simple hill climbing
            cur_pts = pts.copy()
            cur_sc = sc
            step_size = 0.05
            for _ in range(100):
                if time.time() > deadline:
                    break
                new_pts = cur_pts + np.random.randn(n, d) * step_size
                new_sc = score(new_pts)
                if new_sc > cur_sc:
                    cur_pts = new_pts
                    cur_sc = new_sc
                    step_size *= 0.9
                else:
                    step_size *= 0.95
                    
            if cur_sc > best_overall_score:
                best_overall_score = cur_sc
                best_overall_pts = cur_pts

    # Final check
    if best_overall_pts is None:
        # Fallback
        best_overall_pts = seeds[0]
        
    return best_overall_pts
