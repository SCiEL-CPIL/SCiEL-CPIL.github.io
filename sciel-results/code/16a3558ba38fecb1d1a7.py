import numpy as np

N = 21

def construct():
    """
    Pack N circles in a rectangle of perimeter 4 to maximize sum of radii.
    Uses simulated annealing with multiple restarts.
    """
    rng = np.random.default_rng(42)
    
    def compute_penalty(x, y, r, w):
        """Compute penalty for invalid configuration (overlap + out of bounds)."""
        h = 2.0 - w
        penalty = 0.0
        
        # Out of bounds penalty
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i]) ** 2
            if x[i] > w - r[i]:
                penalty += (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                penalty += (r[i] - y[i]) ** 2
            if y[i] > h - r[i]:
                penalty += (y[i] - (h - r[i])) ** 2
        
        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist2 = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist2 < min_dist * min_dist:
                    dist = np.sqrt(dist2)
                    overlap = min_dist - dist
                    penalty += overlap * overlap
        
        return penalty
    
    def is_valid(x, y, r, w, tol=1e-9):
        """Check if configuration is valid."""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return False
        
        for i in range(N):
            if r[i] <= 0:
                return False
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist2 = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist2 < min_dist * min_dist - tol:
                    return False
        
        return True
    
    def objective(x, y, r, w):
        """Objective: maximize sum(r) subject to validity. Returns -sum(r) + large penalty if invalid."""
        h = 2.0 - w
        if w <= 0.01 or h <= 0.01:
            return 1e10
        
        penalty = compute_penalty(x, y, r, w)
        if penalty > 1e-6:
            return 1e10 + penalty
        
        return -np.sum(r)
    
    def anneal(x0, y0, r0, w0, n_iter=50000, t0=0.1, t_end=1e-6, alpha=0.9997):
        """Simulated annealing."""
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        w = w0
        
        best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w
        best_obj = objective(x, y, r, w)
        
        T = t0
        for it in range(n_iter):
            T *= alpha
            if T < t_end:
                T = t_end
            
            # Random perturbation
            x_new = x.copy()
            y_new = y.copy()
            r_new = r.copy()
            w_new = w
            
            # Perturb positions
            idx = rng.integers(N)
            dx = rng.normal(0, 0.05)
            dy = rng.normal(0, 0.05)
            x_new[idx] += dx
            y_new[idx] += dy
            
            # Perturb radius
            idx_r = rng.integers(N)
            dr = rng.normal(0, 0.02)
            r_new[idx_r] = max(0.01, r_new[idx_r] + dr)
            
            # Perturb width occasionally
            if rng.random() < 0.1:
                dw = rng.normal(0, 0.05)
                w_new = np.clip(w + dw, 0.1, 1.9)
            
            obj_new = objective(x_new, y_new, r_new, w_new)
            obj_cur = objective(x, y, r, w)
            
            # Accept if better or with probability exp(-delta/T)
            if obj_new < obj_cur or rng.random() < np.exp(-(obj_new - obj_cur) / T):
                x, y, r, w = x_new, y_new, r_new, w_new
                if obj_new < best_obj:
                    best_obj = obj_new
                    best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w
        
        return best_x, best_y, best_r, best_w, best_obj
    
    # Generate multiple initial configurations
    best_solution = None
    best_value = -1e10
    
    # Configuration 1: Grid packing
    for attempt in range(3):
        # Try different aspect ratios
        if attempt == 0:
            w0 = 1.0
        elif attempt == 1:
            w0 = 1.2
        else:
            w0 = 0.8
        
        h0 = 2.0 - w0
        
        # Grid initialization
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        # Place circles in a grid pattern
        cols = int(np.sqrt(N * w0 / h0))
        rows = int(np.ceil(N / cols))
        
        cell_w = w0 / cols
        cell_h = h0 / rows
        
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                x0[count] = (i + 0.5) * cell_w
                y0[count] = (j + 0.5) * cell_h
                r0[count] = min(cell_w, cell_h) * 0.4
                count += 1
            if count >= N:
                break
        
        # Run annealing
        x, y, r, w, obj = anneal(x0, y0, r0, w0, n_iter=30000)
        
        if is_valid(x, y, r, w):
            value = np.sum(r)
            if value > best_value:
                best_value = value
                best_solution = (x, y, r, w)
    
    # Configuration 2: Hexagonal packing
    w0 = 1.0
    h0 = 2.0 - w0
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    r0 = np.zeros(N)
    
    # Hexagonal grid
    r_init = 0.15
    dx = r_init * 2 * 0.866
    dy = r_init * 2
    
    count = 0
    for row in range(10):
        for col in range(10):
            if count >= N:
                break
            x = col * dx + (row % 2) * dx / 2 + r_init
            y = row * dy + r_init
            if x < w0 and y < h0:
                x0[count] = x
                y0[count] = y
                r0[count] = r_init
                count += 1
        if count >= N:
            break
    
    if count >= N:
        x, y, r, w, obj = anneal(x0[:N], y0[:N], r0[:N], w0, n_iter=30000)
        
        if is_valid(x, y, r, w):
            value = np.sum(r)
            if value > best_value:
                best_value = value
                best_solution = (x, y, r, w)
    
    # Configuration 3: Random initialization
    for attempt in range(2):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        
        x0 = rng.uniform(0.1, w0 - 0.1, N)
        y0 = rng.uniform(0.1, h0 - 0.1, N)
        r0 = np.full(N, 0.1)
        
        x, y, r, w, obj = anneal(x0, y0, r0, w0, n_iter=30000)
        
        if is_valid(x, y, r, w):
            value = np.sum(r)
            if value > best_value:
                best_value = value
                best_solution = (x, y, r, w)
    
    if best_solution is not None:
        x, y, r, w = best_solution
        # Final validation
        if is_valid(x, y, r, w, tol=1e-8):
            return np.column_stack((x, y, r))
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
