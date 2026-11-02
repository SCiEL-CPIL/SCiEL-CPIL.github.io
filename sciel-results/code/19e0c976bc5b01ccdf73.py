"""Circle packing in perimeter-4 rectangle: Simulated Annealing with adaptive moves."""
import numpy as np
import time

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _compute_score(cx, cy, r, W, H):
    """Compute a score for a (possibly invalid) configuration. Lower is better (negative sum radii)."""
    if W <= 0.05 or H <= 0.05:
        return 1e6
    score = -np.sum(r)
    # Boundary violations
    for i in range(N):
        if cx[i] - r[i] < 0:
            score += 100 * (r[i] - cx[i]) ** 2
        if cx[i] + r[i] > W:
            score += 100 * (cx[i] + r[i] - W) ** 2
        if cy[i] - r[i] < 0:
            score += 100 * (r[i] - cy[i]) ** 2
        if cy[i] + r[i] > H:
            score += 100 * (cy[i] + r[i] - H) ** 2
    # Overlap violations
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx * dx + dy * dy
            rs = r[i] + r[j]
            if d2 < rs * rs:
                d = np.sqrt(max(d2, 1e-12))
                overlap = rs - d
                score += 100 * overlap * overlap
    return score


def _clamp_to_bounds(cx, cy, r, W, H):
    """Clamp circles to stay within bounds."""
    for i in range(N):
        cx[i] = np.clip(cx[i], r[i], W - r[i])
        cy[i] = np.clip(cy[i], r[i], H - r[i])


def _local_search(cx, cy, r, W, H, max_iter=2000, rng=None):
    """Local search to improve a valid packing by small perturbations and radius growth."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_val = np.sum(r)
    best_state = (cx.copy(), cy.copy(), r.copy(), W, H)
    
    for it in range(max_iter):
        improved = False
        
        # Try to grow all radii slightly
        factor = 1.0 + rng.uniform(0.0001, 0.002)
        r_trial = r * factor
        if _is_valid(cx, cy, r_trial, W, H):
            r = r_trial
            improved = True
        
        # Try small random moves
        for _ in range(3):
            idx = rng.integers(0, N)
            dx = rng.normal(0, 0.005)
            dy = rng.normal(0, 0.005)
            cx_trial = cx.copy()
            cy_trial = cy.copy()
            cx_trial[idx] += dx
            cy_trial[idx] += dy
            _clamp_to_bounds(cx_trial, cy_trial, r, W, H)
            if _is_valid(cx_trial, cy_trial, r, W, H):
                # Check if this allows radius growth
                r_trial = r.copy()
                grow = 1.001
                for _ in range(10):
                    r_trial_new = r_trial * grow
                    if _is_valid(cx_trial, cy_trial, r_trial_new, W, H):
                        r_trial = r_trial_new
                    else:
                        break
                if np.sum(r_trial) > np.sum(r):
                    cx = cx_trial
                    cy = cy_trial
                    r = r_trial
                    improved = True
                    break
        
        if not improved and it > 100:
            break
    
    if np.sum(r) > best_val:
        best_state = (cx.copy(), cy.copy(), r.copy(), W, H)
        best_val = np.sum(r)
    
    return best_state


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_val = 0.0
    best_result = None
    
    rng = np.random.default_rng(12345)
    
    # Simulated Annealing
    T_start = 0.1
    T_end = 1e-6
    T_current = T_start
    cooling_rate = 0.9995
    
    # Initial configuration: grid-like
    side = int(np.ceil(np.sqrt(N)))
    W0 = 1.0
    H0 = 2.0 - W0
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        row, col = divmod(i, side)
        cx[i] = (col + 0.5) / side * W0
        cy[i] = (row + 0.5) / side * H0
        r[i] = min(W0 / side, H0 / side) * 0.4
    
    W = W0
    H = H0
    
    # Make sure initial state is valid
    _clamp_to_bounds(cx, cy, r, W, H)
    if not _is_valid(cx, cy, r, W, H):
        # Shrink radii until valid
        scale = 1.0
        while not _is_valid(cx, cy, r * scale, W, H) and scale > 0.01:
            scale *= 0.95
        r = r * scale
    
    current_score = _compute_score(cx, cy, r, W, H)
    
    # Track best
    if _is_valid(cx, cy, r, W, H):
        best_val = np.sum(r)
        best_result = (cx.copy(), cy.copy(), r.copy(), W, H)
    
    max_iter = 50000
    for it in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        # Generate a random move
        move_type = rng.integers(0, 5)
        
        cx_trial = cx.copy()
        cy_trial = cy.copy()
        r_trial = r.copy()
        W_trial = W
        H_trial = H
        
        if move_type == 0:
            # Move one circle
            idx = rng.integers(0, N)
            cx_trial[idx] += rng.normal(0, 0.02)
            cy_trial[idx] += rng.normal(0, 0.02)
        elif move_type == 1:
            # Change radius of one circle
            idx = rng.integers(0, N)
            r_trial[idx] *= (1.0 + rng.normal(0, 0.05))
            r_trial[idx] = max(0.005, r_trial[idx])
        elif move_type == 2:
            # Change aspect ratio
            W_trial = W + rng.normal(0, 0.05)
            W_trial = np.clip(W_trial, 0.3, 1.7)
            H_trial = 2.0 - W_trial
            # Rescale positions
            scale_x = W_trial / W
            scale_y = H_trial / H
            cx_trial = cx * scale_x
            cy_trial = cy * scale_y
            r_trial = r * min(scale_x, scale_y)
        elif move_type == 3:
            # Move two circles
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            cx_trial[idx1] += rng.normal(0, 0.015)
            cy_trial[idx1] += rng.normal(0, 0.015)
            cx_trial[idx2] += rng.normal(0, 0.015)
            cy_trial[idx2] += rng.normal(0, 0.015)
        else:
            # Grow all radii slightly
            factor = 1.0 + rng.uniform(0.001, 0.01)
            r_trial = r * factor
        
        _clamp_to_bounds(cx_trial, cy_trial, r_trial, W_trial, H_trial)
        
        trial_score = _compute_score(cx_trial, cy_trial, r_trial, W_trial, H_trial)
        delta = trial_score - current_score
        
        # Acceptance criterion
        if delta < 0 or rng.random() < np.exp(-delta / max(T_current, 1e-10)):
            cx, cy, r, W, H = cx_trial, cy_trial, r_trial, W_trial, H_trial
            current_score = trial_score
            
            if _is_valid(cx, cy, r, W, H):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_result = (cx.copy(), cy.copy(), r.copy(), W, H)
        
        T_current = max(T_current * cooling_rate, T_end)
    
    # Local search on best result
    if best_result is not None and time.time() < deadline - 15:
        cx_ls, cy_ls, r_ls, W_ls, H_ls = best_result
        cx_ls, cy_ls, r_ls, W_ls, H_ls = _local_search(cx_ls, cy_ls, r_ls, W_ls, H_ls, max_iter=5000, rng=rng)
        if _is_valid(cx_ls, cy_ls, r_ls, W_ls, H_ls):
            val = np.sum(r_ls)
            if val > best_val:
                best_val = val
                best_result = (cx_ls.copy(), cy_ls.copy(), r_ls.copy(), W_ls, H_ls)
    
    # Final radius growth
    if best_result is not None and time.time() < deadline - 5:
        cx_f, cy_f, r_f, W_f, H_f = best_result
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            r_trial = r_f * 1.0005
            if _is_valid(cx_f, cy_f, r_trial, W_f, H_f):
                r_f = r_trial
            else:
                break
        if _is_valid(cx_f, cy_f, r_f, W_f, H_f):
            val = np.sum(r_f)
            if val > best_val:
                best_val = val
                best_result = (cx_f.copy(), cy_f.copy(), r_f.copy(), W_f, H_f)
    
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    cx, cy, r, W, H = best_result
    
    # Final safety: ensure strict validity
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            break
    
    return np.column_stack((cx, cy, r))
