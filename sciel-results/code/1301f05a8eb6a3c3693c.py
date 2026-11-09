"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validate(arr):
    """Check if the arrangement is valid."""
    if arr is None or len(arr) != N:
        return False
    xs = arr[:, 0]
    ys = arr[:, 1]
    rs = arr[:, 2]
    if np.any(rs <= 0):
        return False
    # Bounding box
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    w = xmax - xmin
    h = ymax - ymin
    if 2 * (w + h) > 4 + 1e-6:
        return False
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - 1e-6:
                return False
    return True

def _objective(x, w, h):
    """Objective to minimize: negative sum of radii + penalty for violations."""
    # x = [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    penalty = 0.0
    # Boundary constraints
    for i in range(N):
        if xs[i] - rs[i] < 0:
            penalty += (xs[i] - rs[i])**2
        if xs[i] + rs[i] > w:
            penalty += (xs[i] + rs[i] - w)**2
        if ys[i] - rs[i] < 0:
            penalty += (ys[i] - rs[i])**2
        if ys[i] + rs[i] > h:
            penalty += (ys[i] + rs[i] - h)**2
        if rs[i] <= 0:
            penalty += rs[i]**2
    
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    return -np.sum(rs) + 10.0 * penalty

def _pack_for_aspect(w, h, max_time=30.0):
    """Try to pack circles in rectangle w x h."""
    start_time = time.time()
    
    # Grid initialization
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    # Adjust to get at least N cells
    while cols * rows < N:
        if cols < rows:
            cols += 1
        else:
            rows += 1
    
    # Initial radius estimate
    cell_w = w / cols
    cell_h = h / rows
    r0 = min(cell_w, cell_h) / 2 * 0.9
    
    # Create initial positions
    x_init = []
    for i in range(N):
        col = i % cols
        row = i // cols
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        x_init.extend([cx, cy, r0])
    
    x0 = np.array(x_init)
    
    # Try several optimization attempts with different initial conditions
    best_result = None
    best_score = -np.inf
    
    for attempt in range(5):
        if time.time() - start_time > max_time - 5:
            break
        
        # Add some randomness
        if attempt > 0:
            x_try = x0.copy()
            noise_scale = 0.05 * min(w, h)
            x_try[0::3] += np.random.normal(0, noise_scale, N)
            x_try[1::3] += np.random.normal(0, noise_scale, N)
            # Clamp positions
            x_try[0::3] = np.clip(x_try[0::3], 0, w)
            x_try[1::3] = np.clip(x_try[1::3], 0, h)
        else:
            x_try = x0.copy()
        
        # Use SLSQP
        try:
            res = minimize(
                _objective, x_try, args=(w, h),
                method='SLSQP',
                bounds=[(0, None)] * (3 * N),
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if res.success or res.fun < best_score:
                # Check validity
                arr = res.x.reshape(N, 3)
                if _validate(arr):
                    score = np.sum(arr[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = arr
        except Exception:
            pass
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Try various aspect ratios
    aspect_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        result = _pack_for_aspect(w, h, max_time=5.0)
        
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
    
    # If no valid result, fall back to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
