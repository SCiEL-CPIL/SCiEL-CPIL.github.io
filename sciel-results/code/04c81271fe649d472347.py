"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    # Check radii positive
    if np.any(r <= 0):
        return None
    # Check inside unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    # Check disjoint interiors (vectorized)
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
        return None
    return float(np.sum(r))


def _grid_initial(n=N):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])


def _random_initial(n=N, seed=42):
    """Create a random initial configuration with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    # Start with small radii and grow
    circles[:, 2] = 0.01 + 0.01 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _structured_initial(n=N, seed=0):
    """Create a structured initial configuration based on known good packings.
    
    Uses a hexagonal-like arrangement which is efficient for circle packing.
    """
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # Hexagonal grid arrangement
    # For 26 circles, we can use roughly a 5x6 grid with hexagonal offset
    cols = 5
    rows = 6  # 5*6 = 30, we'll use 26
    
    # Hexagonal spacing
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    r_base = min(dx, dy) * 0.48
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            # Hexagonal offset for odd rows
            x_offset = 0.5 * dx if row % 2 == 1 else 0
            cx = (col + 0.5 + x_offset) * dx
            cy = (row + 0.5) * dy
            # Add small random perturbation
            cx += 0.01 * rng.randn()
            cy += 0.01 * rng.randn()
            # Clamp to valid range
            cx = np.clip(cx, r_base, 1 - r_base)
            cy = np.clip(cy, r_base, 1 - r_base)
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r_base
            idx += 1
    
    return circles


def _penalty_objective(params, n=N, penalty_weight=100.0):
    """Objective to minimize: -sum(r) + penalty for violations."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Objective: maximize sum of radii (minimize negative)
    obj = -np.sum(r)
    
    # Penalty for boundary violations
    boundary_viol = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                    np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
    obj += penalty_weight * np.sum(boundary_viol)
    
    # Penalty for circle overlaps (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0, r[i_idx] + r[j_idx] - dist)
    obj += penalty_weight * np.sum(overlap)
    
    return obj


def _local_optimize(circles, time_limit=10, seed=0):
    """Fast local optimization using vectorized pair computations.
    
    Uses a combination of:
    1. Repulsion-based gradient for overlaps
    2. Attraction toward center for boundary
    3. Radius growth when feasible
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    lr = 0.005
    decay = 0.99995
    it = 0
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Compute all pairwise distances
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        # Overlap amount
        overlap = r[i_idx] + r[j_idx] - dist
        
        # Gradient for positions: push apart when overlapping
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        # Repulsion from overlaps
        repulsion = np.where(overlap > 0, 50.0 * overlap / dist, 0.0)
        gx[i_idx] -= repulsion * dx / dist
        gy[i_idx] -= repulsion * dy / dist
        gx[j_idx] += repulsion * dx / dist
        gy[j_idx] += repulsion * dy / dist
        
        # Boundary repulsion
        # Push away from walls
        gx += np.where(x < r, 100.0, 0.0) - np.where(1 - x < r, 100.0, 0.0)
        gy += np.where(y < r, 100.0, 0.0) - np.where(1 - y < r, 100.0, 0.0)
        
        # Radius gradient: want to grow, but penalize overlaps
        gr += 1.0  # Want to grow
        # Penalty for boundary violations
        gr -= np.where(x < r, 100.0, 0.0)
        gr -= np.where(1 - x < r, 100.0, 0.0)
        gr -= np.where(y < r, 100.0, 0.0)
        gr -= np.where(1 - y < r, 100.0, 0.0)
        # Penalty for overlaps (each circle contributes to its overlaps)
        overlap_pen_i = np.zeros(n)
        overlap_pen_j = np.zeros(n)
        np.add.at(overlap_pen_i, i_idx, np.where(overlap > 0, 50.0, 0.0))
        np.add.at(overlap_pen_j, j_idx, np.where(overlap > 0, 50.0, 0.0))
        gr -= overlap_pen_i + overlap_pen_j
        
        # Update
        x += lr * gx
        y += lr * gy
        r += lr * gr
        
        # Clamp
        r = np.maximum(r, 0.001)
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        
        # Ensure within bounds
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        it += 1
        
        # Check score periodically
        if it % 200 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score:
                best_score = score
                best_circles = current_circles.copy()
    
    # Final check
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _lbfgs_optimize(circles, time_limit=15, seed=0):
    """Use L-BFGS-B with penalty method for more robust local optimization."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    
    # Pack into single vector
    params = np.concatenate([x, y, r])
    
    # Define objective
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        return _penalty_objective(p, n=n, penalty_weight=100.0)
    
    # Define gradient (numerical or analytical)
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        # Boundary gradients
        gx += np.where(xx < rr, 100.0, 0.0) - np.where(1 - xx < rr, 100.0, 0.0)
        gy += np.where(yy < rr, 100.0, 0.0) - np.where(1 - yy < rr, 100.0, 0.0)
        gr += np.where(xx < rr, 100.0, 0.0) + np.where(1 - xx < rr, 100.0, 0.0)
        gr += np.where(yy < rr, 100.0, 0.0) + np.where(1 - yy < rr, 100.0, 0.0)
        
        # Overlap gradients
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[i_idx] + rr[j_idx] - dist
        
        mask = overlap > 0
        if np.any(mask):
            rep = 100.0 * overlap[mask] / dist[mask]
            gx[i_idx[mask]] -= rep * dx[mask] / dist[mask]
            gy[i_idx[mask]] -= rep * dy[mask] / dist[mask]
            gx[j_idx[mask]] += rep * dx[mask] / dist[mask]
            gy[j_idx[mask]] += rep * dy[mask] / dist[mask]
            gr[i_idx[mask]] += 100.0
            gr[j_idx[mask]] += 100.0
        
        # Objective gradient: -1 for each radius
        gr -= 1.0
        
        return np.concatenate([gx, gy, gr])
    
    # Bounds: x, y in [0,1], r in [0.001, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
    for i in range(n):
        bounds.append((0.0, 1.0))  # y
    for i in range(n):
        bounds.append((0.001, 0.5))  # r
    
    # Run L-BFGS-B
    result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                      bounds=bounds, options={'maxiter': 1000, 'maxfun': 5000})
    
    if time.time() - start_time > time_limit:
        # Return best from input
        best_score = _validate_and_score(circles)
        if best_score is None:
            best_score = 0.0
        return circles.copy(), best_score
    
    # Extract results
    new_params = result.x
    new_x = new_params[:n]
    new_y = new_params[n:2*n]
    new_r = new_params[2*n:3*n]
    
    # Clamp to ensure validity
    new_r = np.maximum(new_r, 0.001)
    new_x = np.clip(new_x, 0, 1)
    new_y = np.clip(new_y, 0, 1)
    max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
    new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
    
    new_circles = np.column_stack([new_x, new_y, new_r])
    new_score = _validate_and_score(new_circles)
    
    # Compare with input
    input_score = _validate_and_score(circles)
    if input_score is None:
        input_score = 0.0
    if new_score is None:
        new_score = 0.0
    
    if new_score >= input_score:
        return new_circles, new_score
    else:
        return circles.copy(), input_score


def _simulated_annealing(circles, max_iter=50000, t_init=0.05, t_min=0.001,
                          time_limit=80, seed=0):
    """Simulated annealing with fast scoring to optimize circle packing."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _fast_score(cx, cy, cr):
        """Fast validity check returning score or None."""
        if np.any(cr <= 0):
            return None
        if np.any(cx - cr < -1e-9) or np.any(cx + cr > 1 + 1e-9) or np.any(cy - cr < -1e-9) or np.any(cy + cr > 1 + 1e-9):
            return None
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < cr[i_idx] + cr[j_idx] - 1e-9):
            return None
        return float(np.sum(cr))
    
    best_score = _fast_score(x, y, r)
    best_circles = np.column_stack([x, y, r]).copy()
    current_score = best_score
    if current_score is None:
        current_score = 0.0
    
    t = t_init
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb a random circle
        i = rng.randint(n)
        move_type = rng.rand()
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.5:
            # Move position
            delta = 0.015 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.8:
            # Change radius
            new_r[i] *= (1 + 0.08 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        else:
            # Swap-like: adjust two circles
            j = rng.randint(n)
            if j == i:
                j = (i + 1) % n
            delta_r = 0.01 * rng.randn()
            new_r[i] += delta_r
            new_r[j] -= delta_r
            new_r[i] = max(0.005, new_r[i])
            new_r[j] = max(0.005, new_r[j])
        
        new_score = _fast_score(new_x, new_y, new_r)
        
        if new_score is not None:
            if new_score > current_score:
                x, y, r = new_x, new_y, new_r
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_circles = np.column_stack([x, y, r]).copy()
            else:
                # Accept with probability
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / max(t, 1e-10))
                    if rng.rand() < prob:
                        x, y, r = new_x, new_y, new_r
                        current_score = new_score
        else:
            # Invalid configuration, reject
            pass
        
        t *= cooling
    
    return best_circles, best_score


def _resolve_overlaps(circles, max_iter=1000):
    """Iteratively resolve overlaps by pushing circles apart and shrinking radii."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iter):
        # Check for overlaps
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i_idx] + r[j_idx] - dist
        
        if not np.any(overlap > 1e-9):
            break
        
        # Resolve overlaps
        for k in range(len(i_idx)):
            if overlap[k] > 1e-9:
                i = i_idx[k]
                j = j_idx[k]
                if dist[k] < 1e-12:
                    dx_k = 1e-6
                    dy_k = 0
                else:
                    dx_k = dx[k] / dist[k]
                    dy_k = dy[k] / dist[k]
                
                # Push apart
                push = overlap[k] * 0.5
                x[i] -= push * dx_k
                y[i] -= push * dy_k
                x[j] += push * dx_k
                y[j] += push * dy_k
                
                # Also shrink radii slightly if needed
                shrink = overlap[k] * 0.1
                r[i] = max(r[i] - shrink, 0.001)
                r[j] = max(r[j] - shrink, 0.001)
        
        # Clamp positions
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Ensure within bounds
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
    
    return np.column_stack([x, y, r])


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 120  # seconds
    safety_margin = 15  # seconds
    
    best_circles = None
    best_score = 0.0
    
    # Try multiple initializations and optimization strategies
    initializations = [
        ('grid', _grid_initial(N)),
        ('random1', _random_initial(N, seed=1)),
        ('random2', _random_initial(N, seed=2)),
        ('structured1', _structured_initial(N, seed=1)),
        ('structured2', _structured_initial(N, seed=2)),
    ]
    
    for name, init_circles in initializations:
        elapsed = time.time() - start_time
        if elapsed > time_budget - safety_margin:
            break
        
        # First, resolve any initial overlaps
        resolved = _resolve_overlaps(init_circles, max_iter=200)
        
        # Try local optimization with gradient descent
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 10:
            loc_circles, loc_score = _local_optimize(resolved, time_limit=min(remaining * 0.3, 15), seed=hash(name) % 1000)
            if loc_score > best_score:
                best_score = loc_score
                best_circles = loc_circles
        
        # Try L-BFGS-B optimization
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 10:
            lbfgs_circles, lbfgs_score = _lbfgs_optimize(best_circles if best_circles is not None else resolved, 
                                                          time_limit=min(remaining * 0.3, 15), seed=hash(name) % 1000)
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_circles = lbfgs_circles
        
        # Try simulated annealing
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 15:
            sa_circles, sa_score = _simulated_annealing(best_circles if best_circles is not None else resolved,
                                                         max_iter=30000, t_init=0.03, t_min=0.0005,
                                                         time_limit=min(remaining * 0.4, 20), seed=hash(name) % 1000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
    
    # Final polish with local optimization
    if best_circles is not None:
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 5:
            final_circles, final_score = _local_optimize(best_circles, time_limit=min(remaining * 0.5, 10), seed=999)
            if final_score > best_score:
                best_score = final_score
                best_circles = final_circles
    
    # Final validation
    if best_circles is None:
        # Fallback to grid
        return _grid_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        # Fallback to grid
        return _grid_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
