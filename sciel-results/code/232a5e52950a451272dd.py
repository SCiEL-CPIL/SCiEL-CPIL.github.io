"""Circle packing in perimeter-4 rectangle: maximize sum of 21 radii."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 21
DEADLINE = 140.0  # safety margin before 150s kill


def _validate(circles):
    """Check if circles are valid: non-overlapping, inside rectangle, perimeter <= 4."""
    if circles is None or len(circles) != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Check radii are positive
    if np.any(rs <= 0):
        return False
    
    # Check circles are inside the bounding box
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    
    # Check each circle is within [0, W] x [0, H] where W = max_x - min_x, H = max_y - min_y
    W = max_x - min_x
    H = max_y - min_y
    
    if W <= 0 or H <= 0:
        return False
    
    # Perimeter constraint
    if 2 * (W + H) > 4.0 + 1e-6:
        return False
    
    # Check each circle is inside the rectangle
    for i in range(N):
        if xs[i] - rs[i] < -1e-6 or xs[i] + rs[i] > W + 1e-6:
            return False
        if ys[i] - rs[i] < -1e-6 or ys[i] + rs[i] > H + 1e-6:
            return False
    
    # Check no overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-6:
                return False
    
    return True


def _compute_score(circles):
    """Compute the sum of radii if valid, else 0."""
    if not _validate(circles):
        return 0.0
    return np.sum(circles[:, 2])


def _penalty_objective(params):
    """
    Penalty-based objective for L-BFGS-B.
    params: 2N+2 array: [x1..xN, y1..yN, W, H]
    We optimize to maximize sum of radii, but radii are determined by the configuration.
    Actually, let's use a different approach: optimize positions and radii directly.
    
    New approach: params = [x1..xN, y1..yN, r1..rN, W, H]
    Objective: minimize -sum(r) + penalty terms
    """
    N_c = N
    x = params[:N_c]
    y = params[N_c:2*N_c]
    r = params[2*N_c:3*N_c]
    W = params[3*N_c]
    H = params[3*N_c + 1]
    
    penalty = 0.0
    
    # Perimeter constraint: 2(W+H) <= 4
    perim = 2 * (W + H)
    if perim > 4.0:
        penalty += 100.0 * (perim - 4.0) ** 2
    
    # Circles inside rectangle
    for i in range(N_c):
        if x[i] - r[i] < 0:
            penalty += 10.0 * (x[i] - r[i]) ** 2
        if x[i] + r[i] > W:
            penalty += 10.0 * (x[i] + r[i] - W) ** 2
        if y[i] - r[i] < 0:
            penalty += 10.0 * (y[i] - r[i]) ** 2
        if y[i] + r[i] > H:
            penalty += 10.0 * (y[i] + r[i] - H) ** 2
    
    # Non-overlap
    for i in range(N_c):
        for j in range(i + 1, N_c):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += 10.0 * (min_dist - dist) ** 2
    
    return -np.sum(r) + penalty


def _make_initial_config(W, H, n_circles):
    """Create an initial grid-like configuration of n_circles in a W x H rectangle."""
    # Use a grid pattern
    side = int(np.ceil(np.sqrt(n_circles)))
    xs = np.zeros(n_circles)
    ys = np.zeros(n_circles)
    rs = np.zeros(n_circles)
    
    for i in range(n_circles):
        row = i // side
        col = i % side
        xs[i] = (col + 0.5) * W / side
        ys[i] = (row + 0.5) * H / side
        # Initial radius: half the cell size minus a small margin
        cell_w = W / side
        cell_h = H / side
        rs[i] = 0.45 * min(cell_w, cell_h)
    
    # Ensure perimeter constraint
    if 2 * (W + H) > 4.0:
        scale = 2.0 / (W + H)
        W *= scale
        H *= scale
        xs *= scale
        ys *= scale
        rs *= scale
    
    return xs, ys, rs, W, H


def _optimize_from_initial(xs, ys, rs, W, H, time_limit):
    """Run L-BFGS-B optimization from initial configuration."""
    N_c = N
    params0 = np.concatenate([xs, ys, rs, [W, H]])
    
    bounds = []
    for i in range(N_c):
        bounds.append((0.01, 1.0))  # x
    for i in range(N_c):
        bounds.append((0.01, 1.0))  # y
    for i in range(N_c):
        bounds.append((0.01, 1.0))  # r
    bounds.append((0.5, 2.0))  # W
    bounds.append((0.5, 2.0))  # H
    
    start_time = time.time()
    
    try:
        result = minimize(
            _penalty_objective,
            params0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-8, 'gtol': 1e-6}
        )
    except Exception:
        return None
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return None
    
    params = result.x
    x = params[:N_c]
    y = params[N_c:2*N_c]
    r = params[2*N_c:3*N_c]
    W_opt = params[3*N_c]
    H_opt = params[3*N_c + 1]
    
    # Check if the solution is valid
    circles = np.column_stack([x, y, r])
    if not _validate(circles):
        # Try to repair: shrink radii slightly
        r *= 0.95
        circles = np.column_stack([x, y, r])
        if not _validate(circles):
            return None
    
    return circles


def construct():
    start_time = time.time()
    best_circles = None
    best_score = 0.0
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE:
            break
        
        # For perimeter 4: 2(W+H) = 4 => W+H = 2
        # If W/H = ar, then W = ar*H, so ar*H + H = 2 => H = 2/(ar+1), W = 2*ar/(ar+1)
        H = 2.0 / (ar + 1.0)
        W = 2.0 * ar / (ar + 1.0)
        
        time_left = DEADLINE - (time.time() - start_time)
        if time_left < 5.0:
            break
        
        # Allocate time for this aspect ratio
        time_per_ar = min(15.0, time_left * 0.8 / max(1, len(aspect_ratios) - aspect_ratios.index(ar)))
        
        # Try multiple initial configurations
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Add some randomness to the initial configuration
            if restart == 0:
                xs, ys, rs, W_init, H_init = _make_initial_config(W, H, N)
            else:
                xs, ys, rs, W_init, H_init = _make_initial_config(W, H, N)
                # Perturb
                xs += np.random.uniform(-0.1, 0.1, size=N)
                ys += np.random.uniform(-0.1, 0.1, size=N)
                rs *= np.random.uniform(0.8, 1.2, size=N)
                # Clip to valid range
                xs = np.clip(xs, 0.05, W_init - 0.05)
                ys = np.clip(ys, 0.05, H_init - 0.05)
                rs = np.clip(rs, 0.02, min(W_init, H_init) * 0.4)
            
            circles = _optimize_from_initial(xs, ys, rs, W_init, H_init, time_per_ar)
            
            if circles is not None:
                score = _compute_score(circles)
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
    
    # If no valid solution found, return the seed solution
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles


# Test
if __name__ == '__main__':
    np.random.seed(42)
    circles = construct()
    print(f"Number of circles: {len(circles)}")
    print(f"Sum of radii: {np.sum(circles[:, 2]):.6f}")
    print(f"Valid: {_validate(circles)}")
