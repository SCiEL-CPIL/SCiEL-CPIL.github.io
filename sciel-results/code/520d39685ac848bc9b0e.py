# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin

    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    num_pairs = len(i_idx)

    def compute_score(pts):
        """Compute R = (d_min / d_max)^2"""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_pts(pts):
        """Normalize points so that d_max = 1 (scale by 1/d_max)"""
        d = pdist(pts)
        d_max = np.max(d)
        if d_max < 1e-15:
            return pts
        return pts / d_max

    def objective_and_grad(x):
        """
        Objective: minimize -d_min (i.e., maximize d_min)
        Subject to: d_max <= 1 (handled by normalization in objective)
        Actually, we'll use a different approach:
        Minimize: -log(d_min) subject to d_max = 1 (via normalization)
        """
        pts = x.reshape(n, 2)
        pts_norm = normalize_pts(pts)
        d = pdist(pts_norm)
        d_min = np.min(d)
        d_max = np.max(d)
        
        if d_min < 1e-15:
            return 1e10, np.zeros_like(x)
        
        # We want to maximize d_min, so minimize -d_min
        # But we need to be careful about the normalization
        # Actually, let's use a smooth approach: minimize -log(d_min)
        obj_val = -np.log(d_min)
        
        # Gradient: d/dx [-log(d_min)] = -1/d_min * d(d_min)/dx
        # d_min is the minimum of all pairwise distances
        idx_min = np.argmin(d)
        i, j = i_idx[idx_min], j_idx[idx_min]
        diff = pts_norm[i] - pts_norm[j]
        d_min_val = d[idx_min]
        
        # Gradient of distance w.r.t. points
        grad_pts = np.zeros_like(pts_norm)
        grad_pts[i] = diff / d_min_val
        grad_pts[j] = -diff / d_min_val
        
        # Now account for the normalization: pts_norm = pts / d_max
        # This is complex, so let's use a simpler formulation
        
        return obj_val, grad_pts.flatten()

    # Better approach: Use trust-constr with proper constraints
    def _d_min(pts):
        return np.min(pdist(pts))
    
    def _d_max(pts):
        return np.max(pdist(pts))
    
    def _objective(x):
        """Maximize d_min subject to d_max = 1"""
        pts = x.reshape(n, 2)
        # Normalize so d_max = 1
        d_max_val = _d_max(pts)
        if d_max_val < 1e-15:
            return 1e10
        pts_norm = pts / d_max_val
        return -_d_min(pts_norm)
    
    def _objective_grad(x):
        """Gradient of objective"""
        pts = x.reshape(n, 2)
        d_max_val = _d_max(pts)
        if d_max_val < 1e-15:
            return np.zeros_like(x)
        
        pts_norm = pts / d_max_val
        d = pdist(pts_norm)
        d_min_val = np.min(d)
        if d_min_val < 1e-15:
            return np.zeros_like(x)
        
        idx_min = np.argmin(d)
        i, j = i_idx[idx_min], j_idx[idx_min]
        diff = pts_norm[i] - pts_norm[j]
        
        grad_pts = np.zeros_like(pts_norm)
        grad_pts[i] = -diff / d_min_val
        grad_pts[j] = diff / d_min_val
        
        # Chain rule for normalization: d(pts_norm)/d(pts) = 1/d_max
        # But d_max also depends on pts, so this is complex
        # For now, use the simple gradient
        return grad_pts.flatten()
    
    def _constraint_dmax(x):
        """Constraint: d_max <= 1"""
        pts = x.reshape(n, 2)
        return 1.0 - _d_max(pts)
    
    def _constraint_dmax_grad(x):
        """Gradient of d_max constraint"""
        pts = x.reshape(n, 2)
        d = pdist(pts)
        idx_max = np.argmax(d)
        i, j = i_idx[idx_max], j_idx[idx_max]
        diff = pts[i] - pts[j]
        d_max_val = d[idx_max]
        
        grad_pts = np.zeros_like(pts)
        if d_max_val > 1e-15:
            grad_pts[i] = -diff / d_max_val
            grad_pts[j] = diff / d_max_val
        
        return grad_pts.flatten()

    def generate_hexagonal():
        """Generate a hexagonal-like pattern for 16 points"""
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Ring 1: 6 points
        for k in range(6):
            angle = k * np.pi / 3
            pts.append([np.cos(angle), np.sin(angle)])
        # Ring 2: 9 points (approximate hexagonal)
        for k in range(9):
            angle = k * 2 * np.pi / 9 + np.pi/9
            r = 2.0
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        return np.array(pts[:16])

    def generate_random(seed):
        """Generate random points"""
        np.random.seed(seed)
        return np.random.randn(n, 2)

    def generate_grid():
        """Generate a 4x4 grid"""
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))

    def generate_circle(seed):
        """Generate points on a circle with some perturbation"""
        np.random.seed(seed)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.random.randn(n) * 0.1
        radii = 1.0 + np.random.randn(n) * 0.1
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts

    # Initial points: diverse configurations
    initial_configs = [
        generate_hexagonal().flatten(),
        generate_random(42).flatten(),
        generate_grid().flatten(),
        generate_random(123).flatten(),
        generate_random(999).flatten(),
        generate_circle(7).flatten(),
        generate_circle(42).flatten(),
    ]

    best_pts = None
    best_score = -1.0

    # Evaluate initial configs
    for cfg in initial_configs:
        pts = cfg.reshape(n, 2)
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # Run optimization from each initial config using trust-constr
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        
        try:
            from scipy.optimize import LinearConstraint
            from scipy.optimize import NonlinearConstraint
            
            # Constraint: d_max <= 1
            cons = NonlinearConstraint(_constraint_dmax, 0, np.inf, 
                                       jac=_constraint_dmax_grad)
            
            result = minimize(
                _objective,
                cfg,
                jac=_objective_grad,
                method='trust-constr',
                constraints=[cons],
                options={'maxiter': 200, 'gtol': 1e-6, 'xtol': 1e-8}
            )
            
            if time.time() > deadline:
                break
            
            pts = result.x.reshape(n, 2)
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue

    # If we have time, do more random restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = 1000 + num_restarts
        cfg = generate_random(seed).flatten()
        try:
            from scipy.optimize import NonlinearConstraint
            
            cons = NonlinearConstraint(_constraint_dmax, 0, np.inf,
                                       jac=_constraint_dmax_grad)
            
            result = minimize(
                _objective,
                cfg,
                jac=_objective_grad,
                method='trust-constr',
                constraints=[cons],
                options={'maxiter': 100, 'gtol': 1e-5, 'xtol': 1e-7}
            )
            
            if time.time() > deadline:
                break
            
            pts = result.x.reshape(n, 2)
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass
        num_restarts += 1

    if best_pts is None:
        # Fallback to random
        np.random.seed(42)
        best_pts = np.random.randn(n, 2)

    return best_pts

# EVOLVE-BLOCK-END
