"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def _hex_init(n, aspect):
    """Generate hexagonal packing initialization for n circles in a 2D unit-width strip."""
    # For perimeter 4: w + h = 2, so h = 2 - w
    # We parameterize by aspect = h/w, so w = 2/(1+aspect), h = 2*aspect/(1+aspect)
    w = 2.0 / (1.0 + aspect)
    h = 2.0 * aspect / (1.0 + aspect)
    
    # Hexagonal grid spacing
    # Try different density levels
    best_init = None
    for cols in range(3, 8):
        rows = int(np.ceil(n / cols))
        if rows < 1:
            continue
        dx = w / (cols + 0.5)
        dy = h / (rows + 0.5)
        spacing = min(dx, dy * 2.0 / np.sqrt(3))
        r = spacing * 0.5 * 0.95
        
        centers = []
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n:
                    break
                x = (col + 1.0) * dx
                if row % 2 == 1:
                    x += dx * 0.5
                y = (row + 1.0) * dy
                centers.append((x, y, r))
                count += 1
        if len(centers) >= n:
            init = np.array(centers[:n])
            if best_init is None or np.sum(init[:, 2]) > np.sum(best_init[:, 2]):
                best_init = init
    if best_init is None:
        # Fallback: uniform grid
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n:
                    break
                cells.append([w * (i + 0.5) / side, h * (j + 0.5) / side, min(w, h) / (2 * side) * 0.9])
        best_init = np.array(cells)
    return best_init

def _objective(params, n, aspect):
    """Objective: maximize sum of radii, penalize violations."""
    w = 2.0 / (1.0 + aspect)
    h = 2.0 * aspect / (1.0 + aspect)
    
    centers = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    # Penalty for circles going outside bounds
    penalty = 0.0
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        if x - r < 0: penalty += (x - r) ** 2
        if x + r > w: penalty += (x + r - w) ** 2
        if y - r < 0: penalty += (y - r) ** 2
        if y + r > h: penalty += (y + r - h) ** 2
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    # Maximize sum of radii -> minimize negative sum
    return -np.sum(radii) + 100.0 * penalty

def _objective_and_grad(params, n, aspect):
    """Objective with analytical gradient."""
    w = 2.0 / (1.0 + aspect)
    h = 2.0 * aspect / (1.0 + aspect)
    
    centers = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:].copy()
    
    obj = -np.sum(radii)
    grad = np.zeros_like(params)
    grad[2 * n:] = -1.0  # d(-sum(r))/dr = -1
    
    # Boundary constraints (soft)
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        # Left wall
        if x - r < 0:
            diff = x - r
            obj += 100.0 * diff ** 2
            grad[2 * i] += 200.0 * diff
            grad[2 * n + i] += -200.0 * diff
        # Right wall
        if x + r > w:
            diff = x + r - w
            obj += 100.0 * diff ** 2
            grad[2 * i] += 200.0 * diff
            grad[2 * n + i] += 200.0 * diff
        # Bottom wall
        if y - r < 0:
            diff = y - r
            obj += 100.0 * diff ** 2
            grad[2 * i + 1] += 200.0 * diff
            grad[2 * n + i] += -200.0 * diff
        # Top wall
        if y + r > h:
            diff = y + r - h
            obj += 100.0 * diff ** 2
            grad[2 * i + 1] += 200.0 * diff
            grad[2 * n + i] += 200.0 * diff
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            min_dist = radii[i] + radii[j]
            if dist < min_dist and dist > 1e-10:
                diff = min_dist - dist
                obj += 100.0 * diff ** 2
                factor = 200.0 * diff / dist
                grad[2 * i] += factor * dx
                grad[2 * i + 1] += factor * dy
                grad[2 * j] -= factor * dx
                grad[2 * j + 1] -= factor * dy
                grad[2 * n + i] += 200.0 * diff
                grad[2 * n + j] += 200.0 * diff
    
    return obj, grad

def _is_valid(result, n, tol=1e-6):
    """Check if result is valid: all circles inside, no overlaps."""
    if len(result) != n:
        return False
    w = 2.0
    h = 2.0
    # Determine actual rectangle from bounding box
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    
    # Check bounds
    for i in range(n):
        if xs[i] - rs[i] < -tol or xs[i] + rs[i] > w + tol:
            return False
        if ys[i] - rs[i] < -tol or ys[i] + rs[i] > h + tol:
            return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def _polish(result, n, max_iter=500, tol=1e-8):
    """Gradient-based polish to improve packing."""
    w = 2.0
    h = 2.0
    params = result.flatten().copy()
    lr = 0.01
    best_params = params.copy()
    best_val = -np.sum(params[2 * n:])
    
    for it in range(max_iter):
        obj, grad = _objective_and_grad(params, n, 1.0)
        val = -np.sum(params[2 * n:])
        if val > best_val:
            best_val = val
            best_params = params.copy()
        
        # Gradient descent with adaptive step
        norm = np.linalg.norm(grad)
        if norm < tol:
            break
        params -= lr * grad / (norm + 1e-10)
        
        # Project radii to be positive
        params[2 * n:] = np.maximum(params[2 * n:], 1e-6)
        
        # Keep centers in bounds
        for i in range(n):
            r = params[2 * n + i]
            params[2 * i] = np.clip(params[2 * i], r, w - r)
            params[2 * i + 1] = np.clip(params[2 * i + 1], r, h - r)
        
        if it % 50 == 0:
            lr = max(0.001, lr * 0.95)
    
    result_polished = best_params.reshape(n, 3)
    return result_polished

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple aspect ratios and initializations
    aspects = [0.8, 1.0, 1.2, 1.5, 0.7, 1.3]
    np.random.seed(42)
    
    for attempt in range(30):
        if time.time() > deadline - 5.0:
            break
        
        aspect = aspects[attempt % len(aspects)]
        w = 2.0 / (1.0 + aspect)
        h = 2.0 * aspect / (1.0 + aspect)
        
        # Get initial configuration
        if attempt % 3 == 0:
            init = _hex_init(N, aspect)
        elif attempt % 3 == 1:
            # Random init
            r_base = min(w, h) / (2 * int(np.ceil(np.sqrt(N)))) * 0.8
            centers = np.random.uniform(0, 1, (N, 2)) * np.array([w, h])
            radii = np.full(N, r_base)
            init = np.column_stack([centers, radii])
        else:
            # Slightly perturbed hex
            init = _hex_init(N, aspect)
            init[:, :2] += np.random.normal(0, 0.02, (N, 2))
            init[:, 2] *= 0.9
        
        # Scale to fit
        init[:, 0] = np.clip(init[:, 0], 0, w)
        init[:, 1] = np.clip(init[:, 1], 0, h)
        init[:, 2] = np.clip(init[:, 2], 1e-6, min(w, h) / 2)
        
        # SLSQP optimization
        params0 = init.flatten()
        
        # Bounds: centers in [0, w] x [0, h], radii > 0
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, min(w, h) / 2))
        
        try:
            res = minimize(
                _objective,
                params0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
        except Exception:
            continue
        
        if res.success and res.fun < _objective(params0, N, aspect):
            result = res.x.reshape(N, 3)
            # Polish
            result = _polish(result, N, max_iter=200)
            
            # Validate
            if _is_valid(result, N):
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result.copy()
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
