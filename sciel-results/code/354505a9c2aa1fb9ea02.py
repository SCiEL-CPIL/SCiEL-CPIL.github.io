import numpy as np
import time

N = 21
DEADLINE = 145.0  # 5s margin before 150s kill

def _is_valid(circles, tol=1e-9):
    if circles.shape != (N, 3):
        return False
    if np.any(circles[:, 2] <= 0):
        return False
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    xmin, xmax = np.min(xs - rs), np.max(xs + rs)
    ymin, ymax = np.min(ys - rs), np.max(ys + rs)
    w, h = xmax - xmin, ymax - ymin
    if 2 * (w + h) > 4.0 + 1e-6:
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))

def _hex_positions(w, h, target_count=N):
    """Generate hexagonal lattice positions in w x h rectangle."""
    positions = []
    # Try multiple spacings to get at least N positions
    best_positions = None
    best_score = -1
    
    for spacing in np.linspace(0.08, 0.6, 40):
        dx = spacing
        dy = spacing * np.sqrt(3) / 2.0
        positions = []
        col = 0
        x = 0.0
        while x <= w:
            y_offset = 0.0 if col % 2 == 0 else dy / 2.0
            y = y_offset
            while y <= h:
                if 0 <= x <= w and 0 <= y <= h:
                    positions.append((x, y))
                y += dy
            x += dx * 0.75
            col += 1
        
        if len(positions) >= target_count:
            # Score by how well-spaced they are (prefer larger spacing)
            score = spacing * min(len(positions), target_count) / target_count
            if score > best_score:
                best_score = score
                best_positions = positions[:target_count]
    
    return best_positions

def _init_hex(w, h, seed=0):
    """Initialize circles on hexagonal lattice with estimated radii."""
    positions = _hex_positions(w, h)
    if positions is None or len(positions) < N:
        return None
    
    rng = np.random.RandomState(seed)
    circles = np.zeros((N, 3))
    
    # Estimate radius from local spacing
    # Use average nearest-neighbor distance
    pos_arr = np.array(positions)
    nn_dists = []
    for i in range(len(pos_arr)):
        d = np.sqrt(np.sum((pos_arr - pos_arr[i])**2, axis=1))
        d[i] = np.inf
        nn_dists.append(np.min(d))
    avg_nn = np.mean(nn_dists)
    r_est = avg_nn * 0.48
    
    for i in range(N):
        x, y = positions[i]
        r = min(r_est, x, w - x, y, h - y) * 0.95
        r = max(r, 0.01)
        circles[i] = [x, y, r]
    
    return circles

def _init_grid(w, h, m, n, seed=0):
    """Grid initialization."""
    if m * n < N:
        return None
    circles = np.zeros((N, 3))
    count = 0
    for i in range(m):
        for j in range(n):
            if count >= N:
                break
            x = (i + 0.5) * w / m
            y = (j + 0.5) * h / n
            r = min(w / m, h / n) / 2.0 * 0.95
            circles[count] = [x, y, r]
            count += 1
        if count >= N:
            break
    return circles

def _init_random(w, h, seed):
    """Random initialization with separation."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((N, 3))
    for i in range(N):
        r = rng.uniform(0.05, 0.15)
        x = rng.uniform(r, w - r)
        y = rng.uniform(r, h - r)
        circles[i] = [x, y, r]
    return _separate(circles, w, h, iters=200)

def _separate(circles, w, h, iters=200):
    """Separate overlapping circles and keep in bounds."""
    circles = circles.copy()
    for _ in range(iters):
        moved = False
        for i in range(N):
            x, y, r = circles[i]
            if x - r < 0:
                circles[i, 0] = r
                moved = True
            if x + r > w:
                circles[i, 0] = w - r
                moved = True
            if y - r < 0:
                circles[i, 1] = r
                moved = True
            if y + r > h:
                circles[i, 1] = h - r
                moved = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[j, 0] - circles[i, 0]
                dy = circles[j, 1] - circles[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = circles[i, 2] + circles[j, 2]
                if dist < min_d and dist > 1e-12:
                    overlap = min_d - dist
                    nx, ny = dx/dist, dy/dist
                    circles[i, 0] -= nx * overlap * 0.5
                    circles[i, 1] -= ny * overlap * 0.5
                    circles[j, 0] += nx * overlap * 0.5
                    circles[j, 1] += ny * overlap * 0.5
                    moved = True
                elif dist <= 1e-12:
                    circles[i, 0] -= 0.005
                    circles[j, 0] += 0.005
                    moved = True
        if not moved:
            break
    return circles

def _hill_climb(circles, w, h, max_iters=500, step=0.02):
    """
    Gradient-free coordinate ascent hill climbing.
    Tries to increase each radius, move positions, and adjust aspect ratio.
    """
    best = circles.copy()
    best_score = _sum_radii(best)
    current_w, current_h = w, h
    
    for it in range(max_iters):
        improved = False
        
        # Try increasing each radius
        for i in range(N):
            old_r = best[i, 2]
            for frac in [1.03, 1.015, 1.005]:
                new_r = old_r * frac
                # Check bounds
                if new_r > best[i, 0] or new_r > current_w - best[i, 0]:
                    continue
                if new_r > best[i, 1] or new_r > current_h - best[i, 1]:
                    continue
                best[i, 2] = new_r
                sep = _separate(best, current_w, current_h, iters=50)
                if _is_valid(sep):
                    score = _sum_radii(sep)
                    if score > best_score:
                        best_score = score
                        best = sep
                        improved = True
                        break
                else:
                    best[i, 2] = old_r
        
        # Try moving each circle
        for i in range(N):
            for dx in [-step, step]:
                for dy in [-step, step]:
                    old_x, old_y = best[i, 0], best[i, 1]
                    new_x = np.clip(old_x + dx, best[i, 2], current_w - best[i, 2])
                    new_y = np.clip(old_y + dy, best[i, 2], current_h - best[i, 2])
                    best[i, 0] = new_x
                    best[i, 1] = new_y
                    sep = _separate(best, current_w, current_h, iters=30)
                    if _is_valid(sep):
                        score = _sum_radii(sep)
                        if score > best_score:
                            best_score = score
                            best = sep
                            improved = True
                            break
                    else:
                        best[i, 0] = old_x
                        best[i, 1] = old_y
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best, best_score

def _optimize_aspect(circles, max_rounds=20):
    """
    Optimize aspect ratio: given circle positions relative to a unit square,
    find optimal w, h with w+h=2 that maximizes sum of radii.
    """
    # Normalize positions to [0,1] x [0,1]
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    xmin, xmax = np.min(xs - rs), np.max(xs + rs)
    ymin, ymax = np.min(ys - rs), np.max(ys + rs)
    w0, h0 = xmax - xmin, ymax - ymin
    
    # Normalize to unit square
    nx = (xs - xmin) / w0
    ny = (ys - ymin) / h0
    nr = rs / np.array([w0, w0, h0, h0]).repeat(N)  # This is wrong, let me fix
    
    # Actually, let's work with normalized positions and optimize w, h
    # Given normalized positions (nx, ny) in [0,1]^2, and normalized radii (nr)
    # Actual: x = nx * w, y = ny * h, r = min(nr_x * w, nr_y * h)
    # This is complex. Let's just try different aspect ratios directly.
    
    best = circles.copy()
    best_score = _sum_radii(best)
    
    # Try aspect ratios
    for ar in np.linspace(0.5, 2.0, 20):
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Scale positions to fit in w x h
        scale_x = w / w0
        scale_y = h / h0
        scale = min(scale_x, scale_y)
        
        new_circles = circles.copy()
        new_circles[:, 0] = (xs - xmin) * scale
        new_circles[:, 1] = (ys - ymin) * scale
        new_circles[:, 2] = rs * scale
        
        # Center in rectangle
        cx = np.mean(new_circles[:, 0] + new_circles[:, 2])
        cy = np.mean(new_circles[:, 1] + new_circles[:, 2])
        new_circles[:, 0] += (w/2 - cx)
        new_circles[:, 1] += (h/2 - cy)
        
        # Separate
        sep = _separate(new_circles, w, h, iters=100)
        if _is_valid(sep):
            score = _sum_radii(sep)
            if score > best_score:
                best_score = score
                best = sep
    
    return best, best_score

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8]
    
    for ar in aspect_ratios:
        if time_left() < 15.0:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Multiple initializations
        inits = []
        
        # Hex lattice (primary)
        for seed in range(3):
            if time_left() < 10.0:
                break
            hex_c = _init_hex(w, h, seed)
            if hex_c is not None and _is_valid(hex_c):
                inits.append(hex_c)
        
        # Grid
        for m, n in [(3, 7), (4, 5), (5, 4), (7, 3)]:
            if m * n < N:
                continue
            grid_c = _init_grid(w, h, m, n)
            if grid_c is not None and _is_valid(grid_c):
                inits.append(grid_c)
        
        # Random
        for seed in range(3):
            if time_left() < 5.0:
                break
            rand_c = _init_random(w, h, seed + int(ar * 100))
            if _is_valid(rand_c):
                inits.append(rand_c)
        
        # Optimize each initialization
        for init in inits:
            if time_left() < 10.0:
                break
            
            # Hill climbing
            hc_result, hc_score = _hill_climb(init, w, h, max_iters=300, step=0.02)
            if hc_score > best_score:
                best_score = hc_score
                best_result = hc_result.copy()
            
            # Aspect optimization
            if time_left() > 20.0:
                ao_result, ao_score = _optimize_aspect(hc_result, max_rounds=15)
                if ao_score > best_score:
                    best_score = ao_score
                    best_result = ao_result.copy()
            
            # Final hill climb on best
            if time_left() > 15.0:
                fc_result, fc_score = _hill_climb(best_result, w, h, max_iters=200, step=0.01)
                if fc_score > best_score:
                    best_score = fc_score
                    best_result = fc_result.copy()
    
    # Final polish
    if best_result is not None and time_left() > 10.0:
        xs, ys, rs = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        xmin, xmax = np.min(xs - rs), np.max(xs + rs)
        ymin, ymax = np.min(ys - rs), np.max(ys + rs)
        w = xmax - xmin
        h = ymax - ymin
        best_result[:, 0] -= xmin
        best_result[:, 1] -= ymin
        
        polished, pol_score = _hill_climb(best_result, w, h, max_iters=100, step=0.005)
        if pol_score > best_score:
            best_score = pol_score
            best_result = polished
    
    if best_result is not None and _is_valid(best_result):
        return best_result
    
    # Fallback
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
