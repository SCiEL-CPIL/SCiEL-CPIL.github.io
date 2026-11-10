# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint
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

    def objective_and_grad(x):
        """
        x = [points (32), t (1)]
        Objective: minimize -t
        """
        pts = x[:2*n].reshape(n, 2)
        t = x[2*n]
        
        # Objective value
        obj_val = -t
        
        # Gradient
        grad = np.zeros_like(x)
        grad[2*n] = -1.0
        
        return obj_val, grad

    def constraint_fun(x):
        """
        Constraints:
        1. t <= d_ij for all i<j  =>  t - d_ij <= 0
        2. d_ij <= 1 for all i<j  =>  d_ij - 1 <= 0
        Returns array of shape (240,)
        """
        pts = x[:2*n].reshape(n, 2)
        t = x[2*n]
        
        d = pdist(pts)
        
        # Constraint 1: t - d <= 0
        c1 = t - d
        # Constraint 2: d - 1 <= 0
        c2 = d - 1.0
        
        return np.concatenate([c1, c2])

    def constraint_grad(x):
        """
        Gradient of constraint functions.
        Shape: (240, 33)
        """
        pts = x[:2*n].reshape(n, 2)
        t = x[2*n]
        
        d = pdist(pts)
        
        # Gradient of d_ij w.r.t pts
        grad_d = np.zeros((num_pairs, 2*n))
        
        # Avoid division by zero
        safe_d = np.where(d > 1e-12, d, 1e-12)
        
        for k in range(num_pairs):
            i = i_idx[k]
            j = j_idx[k]
            diff = pts[i] - pts[j]
            grad_vec = diff / safe_d[k]
            
            # d(d_k)/d(p_i) = grad_vec
            # d(d_k)/d(p_j) = -grad_vec
            grad_d[k, 2*i:2*i+2] = grad_vec
            grad_d[k, 2*j:2*j+2] = -grad_vec
            
        # Gradient of c1 = t - d
        # d(c1)/d(t) = 1
        # d(c1)/d(pts) = -grad_d
        grad_c1 = np.zeros((num_pairs, 2*n + 1))
        grad_c1[:, 2*n] = 1.0
        grad_c1[:, :2*n] = -grad_d
        
        # Gradient of c2 = d - 1
        # d(c2)/d(t) = 0
        # d(c2)/d(pts) = grad_d
        grad_c2 = np.zeros((num_pairs, 2*n + 1))
        grad_c2[:, :2*n] = grad_d
        
        # Combine
        return np.vstack([grad_c1, grad_c2])

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

    def normalize_for_opt(pts):
        """Scale points so d_max < 1 to ensure feasibility for d_ij <= 1 constraint"""
        d_max = np.max(pdist(pts))
        if d_max > 1e-9:
            return pts * (0.95 / d_max)
        return pts

    # Initial points: diverse configurations
    initial_configs = [
        generate_hexagonal(),
        generate_random(42),
        generate_grid(),
        generate_random(123),
        generate_random(999),
        generate_circle(7),
        generate_circle(42),
    ]

    best_pts = None
    best_score = -1.0

    # Evaluate initial configs
    for pts_init in initial_configs:
        score = compute_score(pts_init)
        if score > best_score:
            best_score = score
            best_pts = pts_init.copy()

    # Define constraints
    # c <= 0
    constraints = [NonlinearConstraint(constraint_fun, -np.inf, 0, jac=constraint_grad)]
    
    # Bounds for x: points are free, t >= 0
    bounds = [(None, None)] * (2*n) + [(0, None)]

    # Run optimization from each initial config
    for pts_init in initial_configs:
        if time.time() > deadline:
            break
        
        pts_opt_init = normalize_for_opt(pts_init)
        
        # Initial t: slightly less than min distance of init
        d_init = pdist(pts_opt_init)
        t_init = np.min(d_init) * 0.9
        
        x0 = np.concatenate([pts_opt_init.flatten(), [t_init]])
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='trust-constr',
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-10, 'verbose': 0}
            )
            
            if time.time() > deadline:
                break
            
            x_opt = result.x
            pts_opt = x_opt[:2*n].reshape(n, 2)
            score = compute_score(pts_opt)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
                
        except Exception:
            continue

    # If we have time, do more random restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = 1000 + num_restarts
        pts_init = generate_random(seed)
        pts_opt_init = normalize_for_opt(pts_init)
        
        d_init = pdist(pts_opt_init)
        t_init = np.min(d_init) * 0.9
        
        x0 = np.concatenate([pts_opt_init.flatten(), [t_init]])
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='trust-constr',
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 300, 'gtol': 1e-7, 'xtol': 1e-9, 'verbose': 0}
            )
            
            if time.time() > deadline:
                break
            
            x_opt = result.x
            pts_opt = x_opt[:2*n].reshape(n, 2)
            score = compute_score(pts_opt)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
                
        except Exception:
            pass
        num_restarts += 1

    if best_pts is None:
        # Fallback to random
        np.random.seed(42)
        best_pts = np.random.randn(n, 2)

    return best_pts

# EVOLVE-BLOCK-END
