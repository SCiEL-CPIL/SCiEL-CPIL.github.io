"""Circle packing in perimeter-4 rectangle: multi-start local search with force-directed and gradient polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    rng = np.random.RandomState(42)

    def get_rect_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def evaluate(state):
        """
        Returns (score, penalty). Score = sum_r - penalty.
        Higher score is better. Valid solutions have penalty=0.
        """
        t = state[0]
        w, h = get_rect_dims(t)
        
        penalty = 0.0
        
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = state[1 + 3*i]
            coords[i, 1] = state[1 + 3*i+1]
            coords[i, 2] = state[1 + 3*i+2]
        
        radii = coords[:, 2]
        if np.any(radii < 1e-5):
            radii = np.maximum(radii, 1e-5)
            penalty += 0.1 * np.sum(radii[radii < 1e-5])
        
        sum_r = np.sum(radii)
        
        # Boundary violations
        x = coords[:, 0]
        y = coords[:, 1]
        
        viol_left = np.maximum(0, radii - x)
        viol_right = np.maximum(0, x + radii - w)
        viol_bottom = np.maximum(0, radii - y)
        viol_top = np.maximum(0, y + radii - h)
        
        penalty += 500.0 * (np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top))
        
        # Overlap violations - vectorized using squared distances
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = radii[idx_i] + radii[idx_j]
        
        # Only compute sqrt where needed (where dist_sq < min_dist^2)
        overlap_mask = dist_sq < min_dist * min_dist
        if np.any(overlap_mask):
            dist = np.sqrt(dist_sq[overlap_mask])
            overlaps = min_dist[overlap_mask] - dist
            penalty += 1000.0 * np.sum(overlaps)
        
        score = sum_r - penalty
        return score, penalty

    def check_valid(state, eps=1e-7):
        """
        Check if state is valid: all circles inside rectangle, no overlaps.
        """
        t = state[0]
        w, h = get_rect_dims(t)
        
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = state[1 + 3*i]
            coords[i, 1] = state[1 + 3*i+1]
            coords[i, 2] = state[1 + 3*i+2]
        
        radii = coords[:, 2]
        if np.any(radii < 1e-6):
            return False
        
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Boundary check
        if np.any(x - radii < -eps) or np.any(x + radii > w + eps) or \
           np.any(y - radii < -eps) or np.any(y + radii > h + eps):
            return False
        
        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j] - eps
                if dist_sq < min_dist * min_dist:
                    return False
        return True

    def make_initial_hexagonal(t_init=0.5, jitter=0.0):
        """
        Create an initial state using a hexagonal lattice pattern.
        This provides a much better starting point than square grids.
        """
        w, h = get_rect_dims(t_init)
        
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        # For hexagonal packing, we want rows offset by half a diameter.
        # Let's try to fit as many circles as possible in a hexagonal pattern.
        # We will place circles row by row.
        
        # Estimate radius. In a hexagonal packing of N circles in a rectangle,
        # the radius is roughly related to area.
        # Area = N * pi * r^2. Rectangle area = w*h.
        # Packing density ~ 0.9069.
        # N * pi * r^2 ~ 0.9069 * w * h
        # r ~ sqrt(0.9069 * w * h / (N * pi))
        
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        
        # Hexagonal spacing:
        # Horizontal distance between centers in same row: d = 2r
        # Vertical distance between rows: dy = sqrt(3) * r
        # Odd rows shifted by r horizontally.
        
        # Let's start with a guess for r and adjust if needed.
        # We will place circles and check if they fit.
        
        # The width available for centers is [r, w-r].
        # The height available for centers is [r, h-r].
        
        # Let's try to fit circles.
        # If r is too big, they won't fit. If too small, we have space.
        
        # We will just place them in a grid-like hex pattern and then let the optimizer adjust.
        # To ensure a good start, we pick r such that the hex lattice fits.
        
        # Number of circles per row depends on row index.
        # Row 0: x = r + i * 2r
        # Row 1: x = r + r + i * 2r = r(2+i)
        
        # Let's just generate positions based on r_current and see how many fit.
        # We'll iterate to find a good r.
        
        # Simple approach: start with r_est, place circles, if any go out of bounds, shrink r.
        
        positions = []
        
        # Try to fill the rectangle
        # Max rows = floor((h - 2r) / (sqrt(3)*r)) + 1
        # Max cols in row 0 = floor((w - 2r) / (2r)) + 1
        
        # We'll just generate a dense hex grid and take the first N that fit.
        # If they don't fit, we shrink r.
        
        for attempt in range(20):
            positions = []
            y = r_current if 'r_current' in dir() else r_est
            row_idx = 0
            while y <= h - r_current + 1e-9:
                # Determine x start
                if row_idx % 2 == 0:
                    x_start = r_current
                else:
                    x_start = r_current + r_current # offset by r
                
                x = x_start
                while x <= w - r_current + 1e-9:
                    positions.append((x, y, r_current))
                    x += 2.0 * r_current
                y += np.sqrt(3) * r_current
                row_idx += 1
            
            if len(positions) >= N:
                # We have enough positions. Take the first N.
                # But we might want to center them or pick the best N.
                # Taking the first N (top-left) is fine for initialization.
                break
            else:
                # Shrink radius
                r_current = r_est if 'r_current' not in dir() else r_current * 0.95
                r_current = max(r_current, 0.01)
        
        # If we still don't have enough, just use a smaller r
        if len(positions) < N:
            # Fallback to square grid
            side = 5
            idx = 0
            positions = []
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.95
                    positions.append((gx, gy, r))
                    idx += 1
        
        # Fill state
        for i in range(N):
            if i < len(positions):
                x, y, r = positions[i]
                # Add jitter
                if jitter > 0:
                    x += rng.uniform(-jitter, jitter)
                    y += rng.uniform(-jitter, jitter)
                    r *= (1 + rng.uniform(-0.05, 0.05))
                state[1 + 3*i] = x
                state[1 + 3*i+1] = y
                state[1 + 3*i+2] = r
            else:
                # Fill remaining with small circles
                x = rng.uniform(0.05, w - 0.05)
                y = rng.uniform(0.05, h - 0.05)
                r = 0.02
                state[1 + 3*i] = x
                state[1 + 3*i+1] = y
                state[1 + 3*i+2] = r
        
        return state

    def force_directed_relax(state, n_steps=200, step_size=0.01, deadline=None):
        """
        Force-directed relaxation: push overlapping circles apart and scale to fit perimeter.
        Returns updated state.
        """
        state = state.copy()
        
        for step in range(n_steps):
            if deadline is not None and time.time() > deadline:
                break
            
            t = state[0]
            w, h = get_rect_dims(t)
            
            coords = np.zeros((N, 3))
            for i in range(N):
                coords[i, 0] = state[1 + 3*i]
                coords[i, 1] = state[1 + 3*i+1]
                coords[i, 2] = state[1 + 3*i+2]
            
            xs = coords[:, 0]
            ys = coords[:, 1]
            rs = coords[:, 2]
            
            # Compute forces from overlaps
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    d = np.sqrt(dx**2 + dy**2)
                    req = rs[i] + rs[j]
                    if d < req and d > 1e-12:
                        # Push apart
                        overlap = req - d
                        fx = overlap * dx / d
                        fy = overlap * dy / d
                        forces_x[i] -= fx
                        forces_y[i] -= fy
                        forces_x[j] += fx
                        forces_y[j] += fy
                    elif d <= 1e-12:
                        # Random push if exactly overlapping
                        angle = np.random.uniform(0, 2 * np.pi)
                        fx = req * np.cos(angle)
                        fy = req * np.sin(angle)
                        forces_x[i] -= fx
                        forces_y[i] -= fy
                        forces_x[j] += fx
                        forces_y[j] += fy
            
            # Apply forces
            xs += step_size * forces_x
            ys += step_size * forces_y
            
            # Scale to fit perimeter constraint
            min_x = np.min(xs - rs)
            max_x = np.max(xs + rs)
            min_y = np.min(ys - rs)
            max_y = np.max(ys + rs)
            w_actual = max_x - min_x
            h_actual = max_y - min_y
            perim = 2 * (w_actual + h_actual)
            
            if perim > 4:
                scale = 2.0 / perim
                # Scale positions relative to center
                cx = (min_x + max_x) / 2
                cy = (min_y + max_y) / 2
                xs = cx + (xs - cx) * scale
                ys = cy + (ys - cy) * scale
                rs = rs * scale
            
            # Update state
            for i in range(N):
                state[1 + 3*i] = xs[i]
                state[1 + 3*i+1] = ys[i]
                state[1 + 3*i+2] = rs[i]
        
        return state

    def optimize_from_init(state0, deadline, rng, max_lam_stages=4):
        """
        Run optimization from an initial configuration.
        Uses penalty method with increasing lambda, then force-directed polish.
        """
        best_state = state0.copy()
        best_score, best_penalty = evaluate(state0)
        
        # Multi-stage penalty optimization
        lam_stages = [1e3, 1e4, 1e5, 1e6]
        
        for lam in lam_stages:
            if time.time() > deadline:
                break
            
            def obj(p):
                # Convert to penalty form
                t = p[0]
                w, h = get_rect_dims(t)
                
                coords = np.zeros((N, 3))
                for i in range(N):
                    coords[i, 0] = p[1 + 3*i]
                    coords[i, 1] = p[1 + 3*i+1]
                    coords[i, 2] = p[1 + 3*i+2]
                
                radii = coords[:, 2]
                if np.any(radii < 1e-5):
                    radii = np.maximum(radii, 1e-5)
                
                sum_r = np.sum(radii)
                
                # Boundary violations
                x = coords[:, 0]
                y = coords[:, 1]
                
                viol_left = np.maximum(0, radii - x)
                viol_right = np.maximum(0, x + radii - w)
                viol_bottom = np.maximum(0, radii - y)
                viol_top = np.maximum(0, y + radii - h)
                
                penalty = lam * 500.0 * (np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top))
                
                # Overlap violations
                idx_i, idx_j = np.triu_indices(N, k=1)
                dx = x[idx_i] - x[idx_j]
                dy = y[idx_i] - y[idx_j]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[idx_i] + radii[idx_j]
                
                overlap_mask = dist_sq < min_dist * min_dist
                if np.any(overlap_mask):
                    dist = np.sqrt(dist_sq[overlap_mask])
                    overlaps = min_dist[overlap_mask] - dist
                    penalty += lam * 1000.0 * np.sum(overlaps)
                
                return -sum_r + penalty
            
            try:
                res = minimize(obj, state0, method='L-BFGS-B', 
                              options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10})
                state0 = res.x
                score, penalty = evaluate(state0)
                
                if penalty < 1e-5:
                    if score > best_score:
                        best_score = score
                        best_state = state0.copy()
            except Exception:
                continue
        
        # Force-directed polish
        if time.time() < deadline - 5:
            polished = force_directed_relax(best_state, n_steps=300, step_size=0.005, deadline=deadline)
            score, penalty = evaluate(polished)
            if penalty < 1e-5 and score > best_score:
                best_score = score
                best_state = polished.copy()
        
        # Final high-precision polish
        if time.time() < deadline - 10:
            def obj_final(p):
                t = p[0]
                w, h = get_rect_dims(t)
                
                coords = np.zeros((N, 3))
                for i in range(N):
                    coords[i, 0] = p[1 + 3*i]
                    coords[i, 1] = p[1 + 3*i+1]
                    coords[i, 2] = p[1 + 3*i+2]
                
                radii = coords[:, 2]
                if np.any(radii < 1e-5):
                    radii = np.maximum(radii, 1e-5)
                
                sum_r = np.sum(radii)
                
                # Boundary violations
                x = coords[:, 0]
                y = coords[:, 1]
                
                viol_left = np.maximum(0, radii - x)
                viol_right = np.maximum(0, x + radii - w)
                viol_bottom = np.maximum(0, radii - y)
                viol_top = np.maximum(0, y + radii - h)
                
                penalty = 1e7 * 500.0 * (np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top))
                
                # Overlap violations
                idx_i, idx_j = np.triu_indices(N, k=1)
                dx = x[idx_i] - x[idx_j]
                dy = y[idx_i] - y[idx_j]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[idx_i] + radii[idx_j]
                
                overlap_mask = dist_sq < min_dist * min_dist
                if np.any(overlap_mask):
                    dist = np.sqrt(dist_sq[overlap_mask])
                    overlaps = min_dist[overlap_mask] - dist
                    penalty += 1e7 * 1000.0 * np.sum(overlaps)
                
                return -sum_r + penalty
            
            try:
                res = minimize(obj_final, best_state, method='L-BFGS-B',
                              options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
                state0 = res.x
                score, penalty = evaluate(state0)
                if penalty < 1e-5 and score > best_score:
                    best_score = score
                    best_state = state0.copy()
            except Exception:
                pass
        
        return best_state, best_score

    def make_grid_init(t_init, nx, ny, rng):
        """Create a grid-based initial configuration."""
        w, h = get_rect_dims(t_init)
        
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        count = 0
        for i in range(nx):
            for j in range(ny):
                if count >= N:
                    break
                cx = (i + 0.5) * w / nx
                cy = (j + 0.5) * h / ny
                r = min(w / nx, h / ny) * 0.45
                # Small random perturbation
                cx += rng.uniform(-0.01, 0.01)
                cy += rng.uniform(-0.01, 0.01)
                r *= (1 + rng.uniform(-0.05, 0.05))
                state[1 + 3*count] = cx
                state[1 + 3*count+1] = cy
                state[1 + 3*count+2] = r
                count += 1
        
        # If we have fewer than N, fill with small circles
        while count < N:
            cx = rng.uniform(0, w)
            cy = rng.uniform(0, h)
            r = 0.02
            state[1 + 3*count] = cx
            state[1 + 3*count+1] = cy
            state[1 + 3*count+2] = r
            count += 1
        
        return state

    def make_random_init(t_init, rng):
        """Create a random initial configuration."""
        w, h = get_rect_dims(t_init)
        
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        for i in range(N):
            cx = rng.uniform(0.05, w - 0.05)
            cy = rng.uniform(0.05, h - 0.05)
            r = rng.uniform(0.03, min(w, h) * 0.2)
            state[1 + 3*i] = cx
            state[1 + 3*i+1] = cy
            state[1 + 3*i+2] = r
        
        return state

    # Main search
    best_state = None
    best_score = 0.0
    
    # Try multiple aspect ratios and initialization strategies
    aspect_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Grid configurations
    grid_configs = [(4, 5), (5, 4), (3, 7), (7, 3), (6, 4), (4, 6), (5, 5)]
    
    attempts = 0
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Try grid init
        for nx, ny in grid_configs[:4]:
            if time.time() > deadline:
                break
            attempts += 1
            init_state = make_grid_init(ar, nx, ny, rng)
            state, s = optimize_from_init(init_state, deadline, rng)
            if s > best_score:
                best_score = s
                best_state = state.copy()
        
        # Try hexagonal init
        if time.time() < deadline - 10:
            for nx, ny in [(4, 5), (5, 4), (6, 4)]:
                if time.time() > deadline:
                    break
                attempts += 1
                init_state = make_initial_hexagonal(ar, jitter=0.02)
                state, s = optimize_from_init(init_state, deadline, rng)
                if s > best_score:
                    best_score = s
                    best_state = state.copy()
        
        # Try random init
        if time.time() < deadline - 10:
            attempts += 1
            init_state = make_random_init(ar, rng)
            state, s = optimize_from_init(init_state, deadline, rng)
            if s > best_score:
                best_score = s
                best_state = state.copy()
    
    if best_state is not None:
        # Convert state to circles array
        circles = np.zeros((N, 3))
        for i in range(N):
            circles[i, 0] = best_state[1 + 3*i]
            circles[i, 1] = best_state[1 + 3*i+1]
            circles[i, 2] = best_state[1 + 3*i+2]
        return circles
    
    # Fallback: simple grid
    w, h = 1.0, 1.0
    nx, ny = 5, 5
    cells = []
    for i in range(nx):
        for j in range(ny):
            if len(cells) >= N:
                break
            cells.append([(i + 0.5) * w / nx, (j + 0.5) * h / ny, min(w/nx, h/ny) * 0.4])
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
