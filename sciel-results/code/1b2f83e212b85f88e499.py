"""Circle packing in perimeter-4 rectangle: structured grid initialization + force-directed relaxation + greedy expansion."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START

def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid: all circles inside [0,W]x[0,H] and non-overlapping."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    n = len(r)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _force_relax(x, y, r, w, h, time_budget, max_iter=3000):
    """Force-directed relaxation to resolve overlaps and push circles to walls."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    dt = 0.002
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion for overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    if d < 1e-10:
                        dx = np.random.uniform(-0.01, 0.01)
                        dy = np.random.uniform(-0.01, 0.01)
                        d = np.sqrt(dx * dx + dy * dy)
                    # Force proportional to overlap
                    f = overlap * 2.0
                    fx[i] += (dx / d) * f
                    fy[i] += (dy / d) * f
                    fx[j] -= (dx / d) * f
                    fy[j] -= (dy / d) * f
        
        # Wall repulsion
        for i in range(N):
            # Left wall
            if x[i] - r[i] < 0:
                fx[i] += (r[i] - x[i]) * 5.0
            # Right wall
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 5.0
            # Bottom wall
            if y[i] - r[i] < 0:
                fy[i] += (r[i] - y[i]) * 5.0
            # Top wall
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 5.0
        
        # Apply forces with damping
        x += dt * fx
        y += dt * fy
        
        # Clamp to valid region (allow slight penetration for relaxation)
        for i in range(N):
            x[i] = max(r[i] * 0.5, min(w - r[i] * 0.5, x[i]))
            y[i] = max(r[i] * 0.5, min(h - r[i] * 0.5, y[i]))
        
        # Decay step size
        dt *= 0.999
        
        # Check if converged
        max_force = np.max(np.sqrt(fx * fx + fy * fy))
        if max_force < 1e-6:
            break
    
    return x, y, r


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 100:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            # Max radius from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
    return x, y, r


def _local_search(x, y, r, w, h, time_budget, n_iter=2000):
    """Local search: try small perturbations of positions and re-expand."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    best_sum = np.sum(r)
    rng = np.random.default_rng(42)
    
    for _ in range(n_iter):
        if time.time() > time_budget:
            break
        # Pick a random circle and nudge it
        i = rng.integers(0, N)
        scale = 0.01
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        nx = x[i] + dx
        ny = y[i] + dy
        
        # Check if new position is roughly valid (allow slight overlap for now)
        if nx < 0 or nx > w or ny < 0 or ny > h:
            continue
        
        # Save state
        old_x, old_y, old_r = x[i], y[i], r[i]
        x[i] = nx
        y[i] = ny
        
        # Try to expand
        x2, y2, r2 = _greedy_expand(x, y, r, w, h, time_budget)
        
        if np.sum(r2) > best_sum + 1e-10:
            best_sum = np.sum(r2)
            x = x2
            y = y2
            r = r2
        else:
            # Revert
            x[i] = old_x
            y[i] = old_y
            r[i] = old_r
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_value = 0.0
    best_result = None
    
    # Generate diverse initial configurations
    def get_initial_configs():
        configs = []
        
        # Grid-based configurations with various aspect ratios
        # For a grid of r rows and c cols, we need r*c >= N
        # Optimal aspect ratio for grid: w/h = c/r
        # Perimeter constraint: w + h = 2
        
        grid_configs = [
            (5, 5),   # 25 cells, use 21
            (4, 6),   # 24 cells, use 21
            (3, 7),   # 21 cells, use 21
            (6, 4),   # 24 cells
            (7, 3),   # 21 cells
            (5, 4),   # 20 cells - not enough, skip
            (4, 5),   # 20 cells - not enough, skip
            (6, 3),   # 18 cells - not enough
            (3, 8),   # 24 cells
            (8, 3),   # 24 cells
            (4, 7),   # 28 cells
            (7, 4),   # 28 cells
            (5, 6),   # 30 cells
            (6, 5),   # 30 cells
        ]
        
        for r, c in grid_configs:
            if r * c < N:
                continue
            # Optimal w/h ratio for this grid
            # We want w/c = h/r for uniform cells
            # w + h = 2
            # w = 2*c/(r+c), h = 2*r/(r+c)
            w = 2.0 * c / (r + c)
            h = 2.0 * r / (r + c)
            
            if w < 0.3 or h < 0.3:
                continue
            
            # Place circles in grid cells
            x = np.zeros(N)
            y = np.zeros(N)
            r_arr = np.zeros(N)
            
            cw = w / c
            ch = h / r
            inradius = min(cw, ch) / 2.0 * 0.95  # Slightly less for safety
            
            count = 0
            for i in range(r):
                for j in range(c):
                    if count >= N:
                        break
                    x[count] = (j + 0.5) * cw
                    y[count] = (i + 0.5) * ch
                    r_arr[count] = inradius
                    count += 1
                if count >= N:
                    break
            
            configs.append((x, y, r_arr, w, h))
        
        # Some random-ish starts
        rng = np.random.default_rng(123)
        for _ in range(5):
            w = rng.uniform(0.8, 1.2)
            h = 2.0 - w
            x = rng.uniform(0.05, w - 0.05, N)
            y = rng.uniform(0.05, h - 0.05, N)
            r_arr = np.full(N, 0.05)
            configs.append((x, y, r_arr, w, h))
        
        return configs
    
    configs = get_initial_configs()
    
    for idx, (x0, y0, r0, w, h) in enumerate(configs):
        if time.time() > deadline - 5:
            break
        
        # Force-directed relaxation to resolve any overlaps
        x, y, r = _force_relax(x0, y0, r0, w, h, deadline - 10)
        
        # Greedy expansion
        x, y, r = _greedy_expand(x, y, r, w, h, deadline - 5)
        
        # Check validity
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
        
        # Local search if time permits
        if time.time() < deadline - 15:
            x_ls, y_ls, r_ls = _local_search(x, y, r, w, h, deadline - 10, n_iter=1000)
            if _is_valid(x_ls, y_ls, r_ls, w, h):
                val = float(np.sum(r_ls))
                if val > best_value:
                    best_value = val
                    best_result = (x_ls.copy(), y_ls.copy(), r_ls.copy(), w, h)
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r, w, h = best_result
    
    # Final polish: one more round of greedy expansion
    if time.time() < deadline - 3:
        x, y, r = _greedy_expand(x, y, r, w, h, deadline - 2)
        if not _is_valid(x, y, r, w, h):
            # Revert to previous valid state
            x, y, r, w, h = best_result
    
    # Construct result array
    result = np.zeros((N, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r
    
    return result

# EVOLVE-BLOCK-END
