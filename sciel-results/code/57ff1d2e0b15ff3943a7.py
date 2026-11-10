import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a Model-and-Optimize approach:
    1. Initializes with a hexagonal lattice pattern (strong geometric prior).
    2. Optimizes using D2 symmetry (8 parameters) to quickly find a good basin.
    3. Refines with full 32-parameter optimization.
    4. Uses multiple restarts and methods to avoid local minima.
    5. Includes a final high-precision polish pass.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    def score(points):
        """Compute R = (d_min / d_max)^2"""
        if points is None or len(points) != n:
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_pts(pts):
        """Center and scale points to unit diameter (max pairwise distance)."""
        pts = pts - np.mean(pts, axis=0)
        # Scale so that the maximum pairwise distance is 1
        # This is more natural for the ratio d_min/d_max
        if np.all(pts == 0):
            return pts
        # Use a simple scaling: divide by max norm from center
        # But for better conditioning, let's use the actual max distance
        # However, computing pdist just for normalization is expensive.
        # Let's stick to max norm but be aware it's an approximation.
        # Actually, for the objective d_min/d_max, scaling by any constant cancels out.
        # The issue is translation. Centering is sufficient for invariance.
        # But numerical stability might benefit from scaling.
        # Let's use max norm for speed.
        norms = np.linalg.norm(pts, axis=1)
        max_norm = np.max(norms)
        if max_norm < 1e-10:
            return pts
        return pts / max_norm

    def objective(params):
        """
        Objective function to minimize: -d_min / d_max
        Uses normalization to remove scale/translation invariance.
        """
        pts = params.reshape((n, 2))
        pts = normalize_pts(pts)
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return -ratio

    def smooth_objective(params, beta=100.0):
        """
        Smooth surrogate using soft-min and soft-max via log-sum-exp.
        Helps L-BFGS-B navigate non-smooth landscape.
        Uses normalization.
        """
        pts = params.reshape((n, 2))
        pts = normalize_pts(pts)
        dists = pdist(pts, 'euclidean')
        
        # Soft min: approximate min(dists)
        # min(x) = -1/beta * log(sum(exp(-beta*x)))
        neg_dists = -beta * dists
        max_neg_dists = np.max(neg_dists)
        log_sum_exp_neg = max_neg_dists + np.log(np.sum(np.exp(neg_dists - max_neg_dists)))
        d_min_soft = -log_sum_exp_neg / beta
        
        # Soft max: approximate max(dists)
        # max(x) = 1/beta * log(sum(exp(beta*x)))
        pos_dists = beta * dists
        max_pos_dists = np.max(pos_dists)
        log_sum_exp_pos = max_pos_dists + np.log(np.sum(np.exp(pos_dists - max_pos_dists)))
        d_max_soft = log_sum_exp_pos / beta
        
        if d_max_soft < 1e-10:
            return 0.0
            
        ratio = d_min_soft / d_max_soft
        return -ratio

    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        if not np.all(np.isfinite(pts)):
            return
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    def generate_hexagonal():
        """Generate 16 points in a hexagonal pattern."""
        # 4 rows, 4 columns
        h = np.sqrt(3) / 2
        pts = np.zeros((16, 2))
        for i in range(4):
            y = i * 2 * h
            for j in range(4):
                idx = i * 4 + j
                if i % 2 == 0:
                    x = j
                else:
                    x = j + 0.5
                pts[idx] = [x, y]
        # Center the points
        pts = pts - np.mean(pts, axis=0)
        return pts

    def generate_d2_symmetric(params):
        """
        Map 8 parameters (x1,y1, x2,y2, x3,y3, x4,y4) to 16 points with D2 symmetry.
        Points are (±xi, ±yi) for i=1..4.
        """
        pts = np.zeros((16, 2))
        for i in range(4):
            x = params[i*2]
            y = params[i*2+1]
            pts[i*4] = [x, y]
            pts[i*4+1] = [-x, y]
            pts[i*4+2] = [-x, -y]
            pts[i*4+3] = [x, -y]
        return pts

    def objective_d2(params):
        """Objective for D2 symmetric parameters."""
        pts = generate_d2_symmetric(params)
        return objective(pts.flatten())

    def smooth_objective_d2(params, beta=50.0):
        """Smooth objective for D2 symmetric parameters."""
        pts = generate_d2_symmetric(params)
        return smooth_objective(pts.flatten(), beta)

    def refine(pts, label=""):
        """
        Multi-stage refinement.
        """
        if time.time() > deadline - 10:
            return

        current = pts.copy()
        
        # Stage 1: Smooth L-BFGS-B on full 32 dims
        # Start with lower beta for broad basin, then increase
        betas = [20.0, 50.0, 100.0, 200.0]
        for beta in betas:
            if time.time() > deadline - 10:
                break
            try:
                res1 = minimize(
                    lambda x: smooth_objective(x, beta=beta), 
                    current.flatten(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res1.x is not None and np.all(np.isfinite(res1.x)):
                    current = res1.x.reshape((n, 2))
            except Exception:
                pass

        if time.time() > deadline - 10:
            update_best(current)
            return

        # Stage 2: Raw L-BFGS-B on full 32 dims
        try:
            res2 = minimize(
                objective, 
                current.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                current = res2.x.reshape((n, 2))
        except Exception:
            pass

        if time.time() > deadline - 10:
            update_best(current)
            return

        # Stage 3: Powell (Derivative-free)
        try:
            res3 = minimize(
                objective, 
                current.flatten(), 
                method='Powell', 
                options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            if res3.x is not None and np.all(np.isfinite(res3.x)):
                current = res3.x.reshape((n, 2))
        except Exception:
            pass

        update_best(current)

    # 1. Hexagonal Initialization
    hex_pts = generate_hexagonal()
    refine(hex_pts, "hex")

    # 2. D2 Symmetric Initialization
    np.random.seed(42)
    
    # Try a few symmetric configurations
    for _ in range(8):
        if time.time() > deadline - 30:
            break
        # Random points in first quadrant, with smaller range for denser packing
        params_init = np.random.rand(8) * 3 + 0.5
        try:
            # Use smooth objective with moderate beta
            res_sym = minimize(
                lambda x: smooth_objective_d2(x, beta=50.0), 
                params_init, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_sym.x is not None and np.all(np.isfinite(res_sym.x)):
                pts_sym = generate_d2_symmetric(res_sym.x)
                refine(pts_sym, "d2_sym")
        except Exception:
            pass

    # 3. Random Restarts (Full 32D)
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        # Random points in a square, scaled to roughly match hexagonal extent
        # Use smaller spread to encourage denser packing
        rand_pts = np.random.rand(n, 2) * 4 - 2
        refine(rand_pts, f"rand_{i}")

    # 4. Final Polish on best found
    if best_points is not None and time.time() < deadline - 5:
        # Try multiple final polish strategies
        current_best = best_points.copy()
        
        # Strategy A: L-BFGS-B with very tight tolerance
        try:
            res_final = minimize(
                objective, 
                current_best.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res_final.x is not None and np.all(np.isfinite(res_final.x)):
                cand = res_final.x.reshape((n, 2))
                update_best(cand)
        except Exception:
            pass
        
        # Strategy B: Powell with tight tolerance
        if time.time() < deadline - 5:
            try:
                res_powell = minimize(
                    objective, 
                    best_points.flatten(), 
                    method='Powell', 
                    options={'maxiter': 500, 'ftol': 1e-14, 'xtol': 1e-14}
                )
                if res_powell.x is not None and np.all(np.isfinite(res_powell.x)):
                    cand = res_powell.x.reshape((n, 2))
                    update_best(cand)
            except Exception:
                pass

    if best_points is None:
        return hex_pts

    return best_points
