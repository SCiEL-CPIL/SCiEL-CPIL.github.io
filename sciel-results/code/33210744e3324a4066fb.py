"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # safety margin before 150s kill


def _check_validity(circles, W, H, tol=1e-7):
    """Check if circles are valid: inside rectangle, no overlaps, positive radii."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _smooth_overlap_penalty(circles, W, H, mu=1.0):
    """Smooth penalty for overlaps and boundary violations using log-sum-exp style."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Boundary violations (smooth)
    boundary = np.maximum(0, -x + r) + np.maximum(0, -y + r) + np.maximum(0, x - W + r) + np.maximum(0, y - H + r)
    boundary_pen = np.sum(np.maximum(0, boundary)) ** 2
    
    # Overlap penalties (smooth)
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                overlap_pen += overlap * overlap
    
    # Negative radius penalty
    neg_r_pen = np.sum(np.maximum(0, -r)) ** 2
    
    return 100.0 * (boundary_pen + overlap_pen + neg_r_pen)


def _objective(params, W, H, mu=1.0):
    """Objective: minimize -sum(radii) + smooth penalties."""
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Main objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Smooth penalties
    obj += _smooth_overlap_penalty(circles, W, H, mu)
    
    return obj


def _objective_and_grad(params, W, H):
    """Objective and analytical gradient for trust-constr."""
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    obj = -np.sum(r)
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0
    
    # Boundary penalty gradients
    for i in range(N):
        xi, yi, ri = x[i], y[i], r[i]
        
        # Left boundary: max(0, -xi + ri)
        if -xi + ri > 0:
            grad[3*i] += 100.0 * 2.0 * (-xi + ri) * (-1.0)
            grad[3*i+2] += 100.0 * 2.0 * (-xi + ri) * (1.0)
        
        # Bottom boundary: max(0, -yi + ri)
        if -yi + ri > 0:
            grad[3*i+1] += 100.0 * 2.0 * (-yi + ri) * (-1.0)
            grad[3*i+2] += 100.0 * 2.0 * (-yi + ri) * (1.0)
        
        # Right boundary: max(0, xi - W + ri)
        if xi - W + ri > 0:
            grad[3*i] += 100.0 * 2.0 * (xi - W + ri) * (1.0)
            grad[3*i+2] += 100.0 * 2.0 * (xi - W + ri) * (1.0)
        
        # Top boundary: max(0, yi - H + ri)
        if yi - H + ri > 0:
            grad[3*i+1] += 100.0 * 2.0 * (yi - H + ri) * (1.0)
            grad[3*i+2] += 100.0 * 2.0 * (yi - H + ri) * (1.0)
    
    # Overlap penalty gradients
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-10:
                dist = 1e-10
            
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                # Gradient of overlap^2 w.r.t. positions
                factor = 2.0 * overlap / dist
                grad[3*i] += 100.0 * factor * dx
                grad[3*i+1] += 100.0 * factor * dy
                grad[3*j] -= 100.0 * factor * dx
                grad[3*j+1] -= 100.0 * factor * dy
                # Gradient w.r.t. radii
                grad[3*i+2] += 100.0 * 2.0 * overlap
                grad[3*j+2] += 100.0 * 2.0 * overlap
    
    # Negative radius penalty
    for i in range(N):
        if r[i] < 0:
            grad[3*i+2] += 100.0 * 2.0 * (-r[i])
            obj += 100.0 * r[i] * r[i]
    
    return obj, grad


def _local_optimize(circles, W, H, max_iter=5000, method='Nelder-Mead'):
    """Locally optimize circle positions and radii."""
    params = circles.flatten()
    
    if method == 'trust-constr':
        from scipy.optimize import LinearConstraint
        # Bounds: 0 < r, r <= min(W,H)/2, r <= x <= W-r, r <= y <= H-r
        # For simplicity, use bounds on r and soft constraints on positions
        bounds = [(0.001, min(W, H) / 2)] * N + [(0.001, W - 0.001)] * N + [(0.001, H - 0.001)] * N
        # Reorder: params are [x0,y0,r0, x1,y1,r1, ...]
        bounds = []
        for i in range(N):
            bounds.append((0.001, W - 0.001))  # x
            bounds.append((0.001, H - 0.001))  # y
            bounds.append((0.001, min(W, H) / 2))  # r
        
        try:
            result = minimize(
                _objective_and_grad, params, args=(W, H),
                method='trust-constr',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'gtol': 1e-8, 'xtol': 1e-10}
            )
        except Exception:
            # Fallback to Nelder-Mead
            result = minimize(
                _objective, params, args=(W, H),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
    else:
        result = minimize(
            _objective, params, args=(W, H),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
    
    return result.x.reshape(N, 3)


def _grid_initialization(W, H, n_cols=None, n_rows=None):
    """Create a grid-based initial configuration."""
    if n_cols is None or n_rows is None:
        # Determine grid dimensions based on aspect ratio
        aspect = W / H
        if aspect > 1.2:
            n_cols = int(np.ceil(np.sqrt(N * aspect)))
            n_rows = int(np.ceil(N / n_cols))
        elif aspect < 0.8:
            n_rows = int(np.ceil(np.sqrt(N / aspect)))
            n_cols = int(np.ceil(N / n_rows))
        else:
            n_cols = int(np.ceil(np.sqrt(N)))
            n_rows = int(np.ceil(N / n_cols))
    
    cells = []
    for i in range(n_rows):
        for j in range(n_cols):
            if len(cells) >= N:
                break
            cx = (j + 0.5) * W / n_cols
            cy = (i + 0.5) * H / n_rows
            r = min(W / n_cols, H / n_rows) / 2.0 - 1e-4
            r = max(r, 0.001)
            cells.append([cx, cy, r])
    
    if len(cells) < N:
        # Fill remaining with small circles in random positions
        rng = np.random.default_rng(0)
        while len(cells) < N:
            r = 0.01
            x = rng.uniform(r, W - r)
            y = rng.uniform(r, H - r)
            cells.append([x, y, r])
    
    return np.array(cells[:N])


def _random_initialization(W, H, rng, min_r=0.01, max_r_frac=0.25):
    """Create a random initial configuration with small circles."""
    circles = []
    max_r = min(W, H) * max_r_frac
    for _ in range(N):
        r = rng.uniform(min_r, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _polish(circles, W, H, n_iter=100):
    """Iteratively push overlapping circles apart to fix small violations."""
    circles = circles.copy()
    for _ in range(n_iter):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    dx, dy = 1e-5, 1e-5
                    dist = np.sqrt(dx*dx + dy*dy)
                
                overlap = circles[i][2] + circles[j][2] - dist
                if overlap > 1e-9:
                    # Push apart along the line connecting centers
                    push = overlap / 2.0 + 1e-6
                    ux, uy = dx / dist, dy / dist
                    circles[i][0] += push * ux
                    circles[i][1] += push * uy
                    circles[j][0] -= push * ux
                    circles[j][1] -= push * uy
                    moved = True
        
        # Push circles back into bounds
        for i in range(N):
            r = circles[i][2]
            circles[i][0] = np.clip(circles[i][0], r, W - r)
            circles[i][1] = np.clip(circles[i][1], r, H - r)
        
        if not moved:
            break
    
    return circles


def _best_known(W, H, rng, time_budget=30.0):
    """Find the best configuration for given W, H using multiple strategies."""
    start = time.time()
    best_circles = None
    best_score = -np.inf
    
    # Strategy 1: Grid initialization + local optimization
    circles = _grid_initialization(W, H)
    circles_opt = _local_optimize(circles, W, H, max_iter=3000, method='Nelder-Mead')
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
    circles_opt = _polish(circles_opt, W, H)
    
    if _check_validity(circles_opt, W, H):
        score = np.sum(circles_opt[:, 2])
        if score > best_score:
            best_score = score
            best_circles = circles_opt.copy()
    
    # Strategy 2: Trust-constr with analytical gradients
    if time.time() - start < time_budget - 5:
        circles_tc = _local_optimize(circles_opt, W, H, max_iter=2000, method='trust-constr')
        circles_tc[:, 0] = np.clip(circles_tc[:, 0], circles_tc[:, 2], W - circles_tc[:, 2])
        circles_tc[:, 1] = np.clip(circles_tc[:, 1], circles_tc[:, 2], H - circles_tc[:, 2])
        circles_tc[:, 2] = np.clip(circles_tc[:, 2], 0.001, min(W, H) / 2)
        circles_tc = _polish(circles_tc, W, H)
        
        if _check_validity(circles_tc, W, H):
            score = np.sum(circles_tc[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles_tc.copy()
    
    # Strategy 3: Random restarts with local optimization
    n_restarts = 5
    for _ in range(n_restarts):
        if time.time() - start > time_budget - 5:
            break
        circ_rand = _random_initialization(W, H, rng)
        circ_opt = _local_optimize(circ_rand, W, H, max_iter=2000, method='Nelder-Mead')
        circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
        circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
        circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)
        circ_opt = _polish(circ_opt, W, H)
        
        if _check_validity(circ_opt, W, H):
            score = np.sum(circ_opt[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circ_opt.copy()
    
    if best_circles is None:
        # Fallback: return grid
        best_circles = _grid_initialization(W, H)
        best_score = np.sum(best_circles[:, 2])
    
    return best_circles, best_score


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    # Perimeter = 2(W+H) = 4, so W+H = 2
    # Try various aspect ratios
    aspect_ratios = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
    
    rng = np.random.default_rng(42)
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)
        
        # Allocate time budget based on remaining time
        remaining = DEADLINE - (time.time() - start_time)
        n_ar_remaining = len(aspect_ratios[aspect_ratios.index(ar):])
        time_budget = max(10.0, (remaining - 10.0) / n_ar_remaining)
        
        circles, score = _best_known(W, H, rng, time_budget=min(time_budget, 40.0))
        if score > best_score:
            best_score = score
            best_result = circles.copy()
    
    # Final fallback: simple grid in square
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        best_score = np.sum(best_result[:, 2])
    
    # Ensure the result is valid
    if best_result is not None:
        # Determine W, H for validation
        # Find the aspect ratio that gives W+H=2 and matches the configuration
        # For simplicity, check with the square (W=1, H=1) as fallback
        if not _check_validity(best_result, 1.0, 1.0):
            # Try to find valid W, H
            found_valid = False
            for ar in aspect_ratios:
                W = 2.0 * ar / (ar + 1.0)
                H = 2.0 / (ar + 1.0)
                if _check_validity(best_result, W, H):
                    found_valid = True
                    break
            
            if not found_valid:
                # Scale down radii to fit
                for i in range(N):
                    best_result[i][2] *= 0.95
                # Check again
                if not _check_validity(best_result, 1.0, 1.0):
                    # Last resort: tiny circles
                    side = int(np.ceil(np.sqrt(N)))
                    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.4 / side)
                             for i in range(side) for j in range(side)]
                    best_result = np.array(cells[:N])
                    best_score = np.sum(best_result[:, 2])
    
    return best_result
# EVOLVE-BLOCK-END
