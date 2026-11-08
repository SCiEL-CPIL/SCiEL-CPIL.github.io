import numpy as np
import time

N = 26
DEADLINE = 140.0  # safety margin before 150s

def _compute_loss_and_grad(arr):
    """
    arr: (N, 3) array of (x, y, r)
    Returns loss (to minimize) and gradient.
    Loss = -sum(r) + penalty for overlaps + penalty for boundary violations
    """
    n = arr.shape[0]
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    
    # Base objective: maximize sum of radii -> minimize -sum(r)
    base_loss = -np.sum(r)
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = -np.ones(n)
    
    # Penalty weight for constraint violations
    lam = 50.0
    
    # Pairwise overlap constraints: dist(i,j) >= r_i + r_j
    # gap_ij = dist_ij - r_i - r_j, should be >= 0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-12
            gap = dist - r[i] - r[j]
            if gap < 0:
                # Penalty: lam * gap^2 (quadratic penalty for violation)
                pen = lam * gap * gap
                base_loss += pen
                # Gradient of pen w.r.t. x_i, y_i, r_i, x_j, y_j, r_j
                dpen_dgap = 2 * lam * gap
                # d gap / d x_i = dx/dist, etc.
                ddx = dx / dist
                ddy = dy / dist
                # d gap / d r_i = -1, d gap / d r_j = -1
                grad_x[i] += dpen_dgap * ddx
                grad_y[i] += dpen_dgap * ddy
                grad_r[i] += dpen_dgap * (-1)
                grad_x[j] += dpen_dgap * (-ddx)
                grad_y[j] += dpen_dgap * (-ddy)
                grad_r[j] += dpen_dgap * (-1)
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # gap = x_i - r_i >= 0, etc.
    for i in range(n):
        # Left boundary: x_i - r_i >= 0
        gap_l = x[i] - r[i]
        if gap_l < 0:
            pen = lam * gap_l * gap_l
            base_loss += pen
            dpen = 2 * lam * gap_l
            grad_x[i] += dpen * 1
            grad_r[i] += dpen * (-1)
        
        # Right boundary: 1 - x_i - r_i >= 0
        gap_r = 1.0 - x[i] - r[i]
        if gap_r < 0:
            pen = lam * gap_r * gap_r
            base_loss += pen
            dpen = 2 * lam * gap_r
            grad_x[i] += dpen * (-1)
            grad_r[i] += dpen * (-1)
        
        # Bottom boundary: y_i - r_i >= 0
        gap_b = y[i] - r[i]
        if gap_b < 0:
            pen = lam * gap_b * gap_b
            base_loss += pen
            dpen = 2 * lam * gap_b
            grad_y[i] += dpen * 1
            grad_r[i] += dpen * (-1)
        
        # Top boundary: 1 - y_i - r_i >= 0
        gap_t = 1.0 - y[i] - r[i]
        if gap_t < 0:
            pen = lam * gap_t * gap_t
            base_loss += pen
            dpen = 2 * lam * gap_t
            grad_y[i] += dpen * (-1)
            grad_r[i] += dpen * (-1)
    
    grad = np.column_stack([grad_x, grad_y, grad_r])
    return base_loss, grad

def _is_valid(arr, tol=1e-9):
    """Check if all circles are disjoint and inside the unit square."""
    n = arr.shape[0]
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    
    # Check radii positive
    if np.any(r <= 0):
        return False
    
    # Check boundary
    if np.any(x - r < -tol) or np.any(1.0 - x - r < -tol):
        return False
    if np.any(y - r < -tol) or np.any(1.0 - y - r < -tol):
        return False
    
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True

def _score(arr):
    """Return sum of radii if valid, else -inf."""
    if _is_valid(arr):
        return np.sum(arr[:, 2])
    return -np.inf

def _optimize_from_start(start, max_iters=20000, lr0=0.01, momentum=0.9):
    """
    Run gradient descent with momentum from a starting configuration.
    Returns the best valid configuration found.
    """
    arr = start.copy()
    best_arr = arr.copy()
    best_score = _score(arr)
    
    # Adaptive learning rate
    lr = lr0
    vel = np.zeros_like(arr)
    
    prev_loss = None
    
    for it in range(max_iters):
        loss, grad = _compute_loss_and_grad(arr)
        
        # Check for NaN
        if not np.isfinite(loss):
            break
        
        # Momentum update
        vel = momentum * vel - lr * grad
        new_arr = arr + vel
        
        # Project radii to be positive and not too large
        new_arr[:, 2] = np.clip(new_arr[:, 2], 1e-6, 0.5)
        # Project positions to [0, 1]
        new_arr[:, 0] = np.clip(new_arr[:, 0], 0.0, 1.0)
        new_arr[:, 1] = np.clip(new_arr[:, 1], 0.0, 1.0)
        
        new_score = _score(new_arr)
        
        if new_score > best_score:
            best_score = new_score
            best_arr = new_arr.copy()
        
        # Adaptive learning rate: reduce if loss increases
        if prev_loss is not None and loss > prev_loss * 1.1 + 1e-6:
            lr *= 0.8
            vel *= 0.5
        elif it % 500 == 0:
            lr = max(lr * 0.95, 1e-5)
        
        arr = new_arr
        prev_loss = loss
        
        # Early stopping if loss is very small
        if abs(loss) < 1e-10:
            break
    
    return best_arr, best_score

def construct():
    start_time = time.time()
    best_result = None
    best_score = -np.inf
    
    # Multiple random restarts
    np.random.seed(42)
    
    # Start with a grid-like initial configuration for stability
    side = 6  # 6x6 grid gives 36 cells, we use 26
    for restart in range(30):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Random perturbation of a grid
        positions = []
        for i in range(side):
            for j in range(side):
                positions.append(((i + 0.5) / side, (j + 0.5) / side))
        
        # Shuffle and take N
        np.random.shuffle(positions)
        positions = positions[:N]
        
        # Add small random perturbation
        for p in positions:
            p[0] += np.random.uniform(-0.02, 0.02)
            p[1] += np.random.uniform(-0.02, 0.02)
            p[0] = np.clip(p[0], 0.05, 0.95)
            p[1] = np.clip(p[1], 0.05, 0.95)
        
        # Initial radii: small
        r0 = 0.05
        arr = np.array([[p[0], p[1], r0] for p in positions])
        
        # Run optimization
        result, score = _optimize_from_start(arr)
        
        if score > best_score:
            best_score = score
            best_result = result
        
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
    
    if best_result is None:
        # Fallback: return a valid grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
