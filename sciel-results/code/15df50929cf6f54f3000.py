"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _initial_guess(n):
    """Generate an initial grid-like configuration for n circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.9
                cells.append([x, y, r])
    return np.array(cells)

def _objective_and_grad(params, n, boundary_margin=1e-6):
    """
    Objective: minimize negative sum of radii + penalty terms for constraint violations.
    Constraints:
      1. Each circle inside [0,1]^2: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
      2. Pairwise non-overlap: dist_ij >= r_i + r_j
    We use a smooth penalty approach.
    """
    params = np.asarray(params)
    n = len(params) // 3
    pts = params.reshape(n, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = np.maximum(pts[:, 2], 1e-8)  # Ensure positive radii
    
    # Sum of radii (we maximize this, so minimize negative)
    obj = -np.sum(r)
    
    # Boundary violations (smooth penalty)
    # For each circle, distance to each boundary
    d_left = x - r
    d_right = 1.0 - x - r
    d_bottom = y - r
    d_top = 1.0 - y - r
    
    # Use smooth max(0, -d) via softplus-like function
    def smooth_neg_min(d, k=50):
        # Approximate max(0, -d) smoothly
        # Using: (log(1 + exp(k*(-d))) / k) for the positive part
        # But we want penalty when d < 0, i.e., -d > 0
        # max(0, -d) ≈ softplus(k*(-d))/k
        return np.log1p(np.exp(k * (-d))) / k
    
    pen_left = smooth_neg_min(d_left)
    pen_right = smooth_neg_min(d_right)
    pen_bottom = smooth_neg_min(d_bottom)
    pen_top = smooth_neg_min(d_top)
    
    penalty_boundary = np.sum(pen_left) + np.sum(pen_right) + np.sum(pen_bottom) + np.sum(pen_top)
    
    # Pairwise overlap penalty
    penalty_pair = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_dist = r[i] + r[j]
            violation = min_dist - dist
            # Smooth penalty for violation
            pen_pair_i = smooth_neg_min(-violation)  # penalty when violation > 0
            penalty_pair += pen_pair_i
    
    # Weight penalties
    w_boundary = 10.0
    w_pair = 10.0
    
    total_obj = obj + w_boundary * penalty_boundary + w_pair * penalty_pair
    
    # Gradient computation (numerical for simplicity, or analytical)
    # For efficiency, we'll use numerical gradient
    eps = 1e-7
    grad = np.zeros_like(params)
    for k in range(len(params)):
        params_plus = params.copy()
        params_plus[k] += eps
        f_plus = _objective_no_grad(params_plus, n, boundary_margin)
        grad[k] = (f_plus - total_obj) / eps
    
    return total_obj, grad

def _objective_no_grad(params, n, boundary_margin=1e-6):
    """Objective function without gradient, used for numerical gradient computation."""
    pts = params.reshape(n, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = np.maximum(pts[:, 2], 1e-8)
    
    obj = -np.sum(r)
    
    d_left = x - r
    d_right = 1.0 - x - r
    d_bottom = y - r
    d_top = 1.0 - y - r
    
    def smooth_neg_min(d, k=50):
        return np.log1p(np.exp(k * (-d))) / k
    
    pen_boundary = (np.sum(smooth_neg_min(d_left)) + np.sum(smooth_neg_min(d_right)) +
                    np.sum(smooth_neg_min(d_bottom)) + np.sum(smooth_neg_min(d_top)))
    
    penalty_pair = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_dist = r[i] + r[j]
            violation = min_dist - dist
            penalty_pair += smooth_neg_min(-violation)
    
    w_boundary = 10.0
    w_pair = 10.0
    
    return obj + w_boundary * pen_boundary + w_pair * penalty_pair

def _is_valid(pts, tol=1e-7):
    """Check if all constraints are satisfied."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    
    # Check boundaries
    if np.any(x - r < -tol) or np.any(x + r > 1.0 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1.0 + tol):
        return False
    
    # Check pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True

def _project_to_valid(pts, n):
    """Project points to be valid (inside square, non-overlapping) via simple iterations."""
    pts = pts.copy()
    for _ in range(100):
        # Clamp to square
        for i in range(n):
            pts[i, 2] = max(pts[i, 2], 1e-8)
            pts[i, 0] = np.clip(pts[i, 0], pts[i, 2], 1.0 - pts[i, 2])
            pts[i, 1] = np.clip(pts[i, 1], pts[i, 2], 1.0 - pts[i, 2])
        
        # Resolve overlaps
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = pts[i, 2] + pts[j, 2]
                if dist < min_dist and dist > 1e-12:
                    # Push apart
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    pts[i, 0] += 0.5 * overlap * nx
                    pts[i, 1] += 0.5 * overlap * ny
                    pts[j, 0] -= 0.5 * overlap * nx
                    pts[j, 1] -= 0.5 * overlap * ny
                    changed = True
                elif dist <= 1e-12:
                    # Separate randomly
                    pts[i, 0] += 1e-4
                    changed = True
        if not changed:
            break
    
    # Final clamp
    for i in range(n):
        pts[i, 2] = max(pts[i, 2], 1e-8)
        pts[i, 0] = np.clip(pts[i, 0], pts[i, 2], 1.0 - pts[i, 2])
        pts[i, 1] = np.clip(pts[i, 1], pts[i, 2], 1.0 - pts[i, 2])
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_value = -np.inf
    
    # Try multiple random restarts
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial guess
        if restart == 0:
            pts = _initial_guess(N)
        else:
            # Perturb the initial guess
            pts = _initial_guess(N)
            np.random.seed(restart * 12345 + 67890)
            pts[:, 0] += np.random.uniform(-0.1, 0.1, N)
            pts[:, 1] += np.random.uniform(-0.1, 0.1, N)
            pts[:, 2] *= np.random.uniform(0.5, 1.5, N)
            pts = _project_to_valid(pts, N)
        
        # Run optimization
        try:
            params = pts.flatten()
            res = minimize(
                _objective_and_grad,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.status == 0:
                result_pts = res.x.reshape(N, 3)
                result_pts = _project_to_valid(result_pts, N)
                
                if _is_valid(result_pts, tol=1e-6):
                    value = np.sum(result_pts[:, 2])
                    if value > best_value:
                        best_value = value
                        best_result = result_pts.copy()
        except Exception:
            continue
    
    # If no valid result found, use the initial grid
    if best_result is None:
        pts = _initial_guess(N)
        best_result = _project_to_valid(pts, N)
        # Ensure it's valid
        if not _is_valid(best_result, tol=1e-5):
            # Fallback: use even smaller circles
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) < N:
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        r = 0.45 / side
                        cells.append([x, y, r])
            best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
