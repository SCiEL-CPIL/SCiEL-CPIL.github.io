"""Rewritten program: multi-start stochastic local search with jitter-and-expand for circle packing."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
# Deadline guard: must finish within 120s (evaluator kills at 150s, keep safety margin)
DEADLINE = 115.0  # seconds
START_TIME = time.time()

def _check_deadline():
    return (time.time() - START_TIME) < DEADLINE

def _compute_value(circles):
    """Compute sum of radii, return 0 if invalid."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check boundaries
    if np.any(r < 0) or np.any(x - r < -1e-10) or np.any(x + r > 1 + 1e-10) or \
       np.any(y - r < -1e-10) or np.any(y + r > 1 + 1e-10):
        return 0.0
    
    # Check pairwise non-overlap
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-10:
                return 0.0
    
    return float(np.sum(r))

def _expand_radii(circles, max_iter=200, tol=1e-10):
    """
    Given fixed centers, maximize sum of radii subject to:
      - x_i - r_i >= 0, x_i + r_i <= 1
      - y_i - r_i >= 0, y_i + r_i <= 1
      - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    
    We solve this as a convex QP using scipy.optimize.minimize with SLSQP.
    Variables: r_0, ..., r_{n-1}
    Objective: minimize -sum(r)
    Constraints:
      - r_i >= 0
      - x_i - r_i >= 0  =>  r_i <= x_i
      - 1 - x_i - r_i >= 0  =>  r_i <= 1 - x_i
      - y_i - r_i >= 0  =>  r_i <= y_i
      - 1 - y_i - r_i >= 0  =>  r_i <= 1 - y_i
      - For each pair (i,j): sqrt((x_i-x_j)^2 + (y_i-y_j)^2) >= r_i + r_j
        => r_i + r_j <= d_ij
        This is a linear constraint in r!
    
    So the problem is actually a LP in r! Let's use linear programming.
    """
    from scipy.optimize import linprog
    
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    
    # Objective: minimize -sum(r)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Boundary constraints:
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    A_ub = np.zeros((4 * n, n))
    b_ub = np.zeros(4 * n)
    for i in range(n):
        A_ub[4*i, i] = 1
        b_ub[4*i] = x[i]
        A_ub[4*i+1, i] = 1
        b_ub[4*i+1] = 1 - x[i]
        A_ub[4*i+2, i] = 1
        b_ub[4*i+2] = y[i]
        b_ub[4*i+2] = y[i]
        A_ub[4*i+3, i] = 1
        b_ub[4*i+3] = 1 - y[i]
    
    # Pairwise constraints: r_i + r_j <= d_ij
    # Add these to A_ub and b_ub
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            pairs.append((i, j, d))
    
    if pairs:
        n_pairs = len(pairs)
        A_pair = np.zeros((n_pairs, n))
        b_pair = np.zeros(n_pairs)
        for k, (i, j, d) in enumerate(pairs):
            A_pair[k, i] = 1
            A_pair[k, j] = 1
            b_pair[k] = d
        A_ub = np.vstack([A_ub, A_pair])
        b_ub = np.concatenate([b_ub, b_pair])
    
    # Bounds: r_i >= 0
    bounds = [(0, None)] * n
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            new_r = result.x
            # Ensure no negative due to numerical issues
            new_r = np.maximum(new_r, 0)
            new_circles = circles.copy()
            new_circles[:, 2] = new_r
            return new_circles
        else:
            return circles
    except Exception:
        return circles

def _jitter_and_expand(circles, jitter_scale=0.05, max_iter=100):
    """
    Iteratively apply small random perturbations to centers, then expand radii.
    Keep the best configuration found.
    """
    best_circles = circles.copy()
    best_value = _compute_value(best_circles)
    
    for it in range(max_iter):
        if not _check_deadline():
            break
        
        # Randomly perturb a subset of centers
        n = len(circles)
        # Perturb all centers with small random offsets
        new_circles = best_circles.copy()
        offsets = np.random.normal(0, jitter_scale, size=(n, 2))
        new_circles[:, 0] += offsets[:, 0]
        new_circles[:, 1] += offsets[:, 1]
        
        # Clip centers to [0,1] to stay in bounds
        new_circles[:, 0] = np.clip(new_circles[:, 0], 0, 1)
        new_circles[:, 1] = np.clip(new_circles[:, 1], 0, 1)
        
        # Expand radii
        expanded = _expand_radii(new_circles)
        val = _compute_value(expanded)
        
        if val > best_value + 1e-12:
            best_circles = expanded
            best_value = val
    
    return best_circles

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Multi-start: try several random initial configurations
    best_circles = None
    best_value = -np.inf
    
    n_starts = 30  # Number of random restarts
    
    for start in range(n_starts):
        if not _check_deadline():
            break
        
        # Random initial positions
        np.random.seed(start * 1000 + 42)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        # Small initial radii
        r = np.full(N, 0.01)
        circles = np.column_stack([x, y, r])
        
        # Expand radii first
        circles = _expand_radii(circles)
        val = _compute_value(circles)
        
        if val > best_value:
            best_value = val
            best_circles = circles.copy()
        
        # Jitter and expand
        circles = _jitter_and_expand(circles, jitter_scale=0.03, max_iter=50)
        val = _compute_value(circles)
        
        if val > best_value:
            best_value = val
            best_circles = circles.copy()
    
    # Final polish: tighter jitter
    if best_circles is not None and _check_deadline():
        best_circles = _jitter_and_expand(best_circles, jitter_scale=0.01, max_iter=100)
        best_value = _compute_value(best_circles)
    
    if best_circles is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
