"""
Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.
Combines softplus-penalty L-BFGS-B with vectorized evaluation and multi-start search.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# Precompute pair indices for vectorized overlap checks
_IDX_I, _IDX_J = np.triu_indices(N, k=1)

def _get_dims(t):
    """Get width and height from parameter t (w=2t, h=2(1-t))."""
    t = np.clip(t, 0.05, 0.95)
    w = 2.0 * t
    h = 2.0 * (1.0 - t)
    return w, h

def _evaluate(state, beta=50.0):
    """
    Returns (score, penalty). Score = sum_r - penalty.
    Higher score is better. Valid solutions have penalty=0.
    state: [t, x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    t = state[0]
    w, h = _get_dims(t)
    
    x = state[1::3]
    y = state[2::3]
    radii = state[3::3]
    
    penalty = 0.0
    
    if np.any(radii < 1e-8):
        penalty += 100.0 * np.sum(np.maximum(0, 1e-8 - radii))
        radii_safe = np.maximum(radii, 1e-8)
    else:
        radii_safe = radii
    
    sum_r = np.sum(radii)
    
    # Boundary violations (vectorized)
    viol = 0.0
    viol += np.sum(np.maximum(0, radii_safe - x))       # left
    viol += np.sum(np.maximum(0, x + radii_safe - w))   # right
    viol += np.sum(np.maximum(0, radii_safe - y))       # bottom
    viol += np.sum(np.maximum(0, y + radii_safe - h))   # top
    
    # Overlap violations (vectorized)
    dx = x[_IDX_I] - x[_IDX_J]
    dy = y[_IDX_I] - y[_IDX_J]
    dist_sq = dx*dx + dy*dy
    min_dist = radii_safe[_IDX_I] + radii_safe[_IDX_J]
    
    overlap_mask = dist_sq < min_dist * min_dist
    if np.any(overlap_mask):
        dist = np.sqrt(np.maximum(dist_sq[overlap_mask], 1e-15))
        overlaps = min_dist[overlap_mask] - dist
        viol += np.sum(overlaps)
    
    penalty = 50.0 * viol
    score = sum_r - penalty
    return score, penalty

def _check_valid(state, eps=1e-9):
    """Check if state is valid: all circles inside rectangle, no overlaps."""
    t = state[0]
    w, h = _get_dims(t)
    
    x = state[1::3]
    y = state[2::3]
    radii = state[3::3]
    
    if np.any(radii < 1e-7):
        return False
    
    if np.any(x - radii < -eps) or np.any(x + radii > w + eps) or \
       np.any(y - radii < -eps) or np.any(y + radii > h + eps):
        return False
    
    dx = x[_IDX_I] - x[_IDX_J]
    dy = y[_IDX_I] - y[_IDX_J]
    dist_sq = dx*dx + dy*dy
    min_dist = radii[_IDX_I] + radii[_IDX_J]
    if np.any(dist_sq < (min_dist - eps)**2):
        return False
    
    return True

def _make_hexagonal(t_init=0.5):
    """Create initial state using hexagonal lattice pattern."""
    w, h = _get_dims(t_init)
    state = np.zeros(1 + 3*N)
    state[0] = t_init
    
    # Estimate radius for hex packing
    r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
    
    positions = []
    r_current = r_est
    for attempt in range(30):
        positions = []
        y = r_current
        row_idx = 0
        while y <= h - r_current + 1e-9:
            if row_idx % 2 == 0:
                x_start = r_current
            else:
                x_start = r_current * 2
            x = x_start
            while x <= w - r_current + 1e-9:
                positions.append((x, y, r_current))
                x += 2.0 * r_current
            y += np.sqrt(3) * r_current
            row_idx += 1
        if len(positions) >= N:
            break
        r_current *= 0.93
    
    if len(positions) < N:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        idx = 0
        positions = []
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                gx = (i + 0.5) / side * w
                gy = (j + 0.5) / side * h
                r = 0.5 / side * min(w, h) * 0.92
                positions.append((gx, gy, r))
                idx += 1
    
    for i in range(N):
        if i < len(positions):
            state[1 + 3*i] = positions[i][0]
            state[1 + 3*i + 1] = positions[i][1]
            state[1 + 3*i + 2] = positions[i][2]
        else:
            state[1 + 3*i] = w/2
            state[1 + 3*i + 1] = h/2
            state[1 + 3*i + 2] = 0.01
    return state

def _make_grid(t_init=0.5):
    """Create initial state using square grid pattern."""
    w, h = _get_dims(t_init)
    state = np.zeros(1 + 3*N)
    state[0] = t_init
    
    best_positions = None
    best_avg_r = 0
    for rows in range(2, 10):
        cols = int(np.ceil(N / rows))
        if rows * cols < N:
            continue
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= N:
                    break
                gx = (j + 0.5) / cols * w
                gy = (i + 0.5) / rows * h
                r = 0.5 * min(w / cols, h / rows) * 0.88
                positions.append((gx, gy, r))
        if len(positions) >= N:
            avg_r = np.mean([p[2] for p in positions])
            if avg_r > best_avg_r:
                best_avg_r = avg_r
                best_positions = positions
    
    if best_positions is None:
        side = 5
        best_positions = []
        for i in range(side):
            for j in range(side):
                if len(best_positions) >= N:
                    break
                gx = (i + 0.5) / side * w
                gy = (j + 0.5) / side * h
                r = 0.5 / side * min(w, h) * 0.92
                best_positions.append((gx, gy, r))
    
    for i in range(N):
        state[1 + 3*i] = best_positions[i][0]
        state[1 + 3*i + 1] = best_positions[i][1]
        state[1 + 3*i + 2] = best_positions[i][2]
    return state

def _make_random(t_init=0.5, rng=None):
    """Create random initial state with non-overlapping circles."""
    if rng is None:
        rng = np.random.default_rng()
    w, h = _get_dims(t_init)
    state = np.zeros(1 + 3*N)
    state[0] = t_init
    
    r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
    positions = []
    
    for i in range(N):
        placed = False
        for attempt in range(500):
            r = r_est * rng.uniform(0.7, 1.3)
            x = rng.uniform(r, w - r)
            y = rng.uniform(r, h - r)
            
            valid = True
            for (px, py, pr) in positions:
                dx = x - px
                dy = y - py
                if dx*dx + dy*dy < (r + pr)**2:
                    valid = False
                    break
            
            if valid:
                positions.append((x, y, r))
                placed = True
                break
        
        if not placed:
            positions.append((w/2, h/2, 0.01))
    
    for i in range(N):
        state[1 + 3*i] = positions[i][0]
        state[1 + 3*i + 1] = positions[i][1]
        state[1 + 3*i + 2] = positions[i][2]
    return state

def _optimize_lbfgsb(state, deadline, max_iter=300):
    """Run L-BFGS-B optimization on the state."""
    def objective(s):
        score, _ = _evaluate(s)
        return -score  # minimize negative score
    
    bounds = [(0.05, 0.95)] + [(0, 2.0)]*(2*N) + [(0.001, 1.0)]*N
    
    result = minimize(
        objective, state,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10,
                 'maxls': 50, 'maxfun': 10000}
    )
    
    # Clip result to valid range
    res = result.x.copy()
    res[0] = np.clip(res[0], 0.05, 0.95)
    for i in range(N):
        res[1 + 3*i] = np.clip(res[1 + 3*i], 0, 2.0)
        res[2 + 3*i] = np.clip(res[2 + 3*i], 0, 2.0)
        res[3 + 3*i] = np.clip(res[3 + 3*i], 0.001, 1.0)
    
    return res

def _polish(state, deadline):
    """High-precision gradient polish using small steps."""
    best_state = state.copy()
    best_score, _ = _evaluate(state)
    
    step_size = 0.005
    patience = 0
    max_patience = 100
    
    while time.time() < deadline - 3.0 and patience < max_patience:
        t = state[0]
        w, h = _get_dims(t)
        
        # Compute gradient numerically
        grad = np.zeros_like(state)
        eps = 1e-6
        for k in range(len(state)):
            state_plus = state.copy()
            state_plus[k] += eps
            _, pen_plus = _evaluate(state_plus)
            state_minus = state.copy()
            state_minus[k] -= eps
            _, pen_minus = _evaluate(state_minus)
            # We want to maximize score, so gradient of -score
            grad[k] = (pen_plus - pen_minus) / (2 * eps)
        
        # Take step
        new_state = state - step_size * grad
        new_state[0] = np.clip(new_state[0], 0.05, 0.95)
        for i in range(N):
            new_state[1 + 3*i] = np.clip(new_state[1 + 3*i], 0, 2.0)
            new_state[2 + 3*i] = np.clip(new_state[2 + 3*i], 0, 2.0)
            new_state[3 + 3*i] = np.clip(new_state[3 + 3*i], 0.001, 1.0)
        
        new_score, new_pen = _evaluate(new_state)
        
        if new_score > best_score and new_pen < 1e-6:
            best_score = new_score
            best_state = new_state.copy()
            state = new_state
            step_size = min(step_size * 1.1, 0.02)
            patience = 0
        else:
            step_size *= 0.85
            patience += 1
    
    return best_state, best_score

def construct():
    """Pack 21 circles with disjoint interiors in a rectangle of perimeter 4,
    maximizing the sum of their radii.
    Returns array of shape (n, 3): [x, y, r] for each circle.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_state = None
    
    # Generate diverse initializations
    initializations = []
    
    # Hexagonal at various aspect ratios
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        initializations.append(_make_hexagonal(t))
    
    # Grid at various aspect ratios
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        initializations.append(_make_grid(t))
    
    # Random at various aspect ratios
    for t in [0.40, 0.45, 0.50, 0.55, 0.60]:
        initializations.append(_make_random(t, rng))
    
    # Shuffle for diversity
    idx = np.arange(len(initializations))
    rng.shuffle(idx)
    initializations = [initializations[i] for i in idx]
    
    # Phase 1: Quick L-BFGS-B on all initializations
    time_per_opt = max(2.0, (deadline - start_time - 30.0) / len(initializations))
    
    candidates = []
    for state in initializations:
        if time.time() > deadline - 35.0:
            break
        
        try:
            opt_state = _optimize_lbfgsb(state, deadline, max_iter=200)
            score, pen = _evaluate(opt_state)
            
            if pen < 1e-6 and _check_valid(opt_state):
                if score > best_score:
                    best_score = score
                    best_state = opt_state.copy()
                candidates.append((score, opt_state))
        except Exception:
            continue
    
    # Sort candidates by score
    candidates.sort(key=lambda x: -x[0])
    
    # Phase 2: Refine top candidates with more iterations
    top_k = min(8, len(candidates))
    remaining = deadline - time.time() - 20.0
    if remaining > 10.0 and top_k > 0:
        time_per_refine = max(3.0, remaining / top_k)
        
        for score, state in candidates[:top_k]:
            if time.time() > deadline - 25.0:
                break
            
            try:
                opt_state = _optimize_lbfgsb(state, deadline, max_iter=400)
                new_score, new_pen = _evaluate(opt_state)
                
                if new_pen < 1e-6 and _check_valid(opt_state):
                    if new_score > best_score:
                        best_score = new_score
                        best_state = opt_state.copy()
            except Exception:
                continue
    
    # Phase 3: Polish the best solution
    if best_state is not None and time.time() < deadline - 8.0:
        try:
            polished_state, polished_score = _polish(best_state, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_state = polished_state
        except Exception:
            pass
    
    # Fallback if no valid solution found
    if best_state is None:
        state = _make_grid(0.5)
        best_state = state
        best_score, _ = _evaluate(state)
    
    # Extract final result
    x = best_state[1::3]
    y = best_state[2::3]
    radii = best_state[3::3]
    
    # Final repair: ensure all constraints satisfied
    t = best_state[0]
    w, h = _get_dims(t)
    
    for i in range(N):
        radii[i] = max(radii[i], 0.001)
        radii[i] = min(radii[i], x[i], w - x[i], y[i], h - y[i])
        radii[i] = max(radii[i], 0.001)
        x[i] = np.clip(x[i], radii[i], w - radii[i])
        y[i] = np.clip(y[i], radii[i], h - y[i])
    
    result = np.column_stack((x, y, radii))
    return result
