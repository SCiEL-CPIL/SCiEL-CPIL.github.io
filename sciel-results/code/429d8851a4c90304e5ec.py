"""Circle packing in perimeter-4 rectangle: Physics-based relaxation with repulsive forces."""
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


def _physics_step(x, y, r, W, H, dt, k_repel, k_attract):
    """
    Perform one physics-based relaxation step.
    Forces:
    - Repulsive force between overlapping circles (pushes apart)
    - Attractive force from walls (pulls circles toward center of available space)
    - Gentle expansion force on radii
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    fx = np.zeros(N)
    fy = np.zeros(N)
    
    # Repulsive forces between circles
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-10:
                # Random small separation
                angle = np.random.uniform(0, 2 * np.pi)
                dx = np.cos(angle) * 1e-6
                dy = np.sin(angle) * 1e-6
                d = 1e-6
            
            min_dist = r[i] + r[j]
            if d < min_dist:
                # Overlapping: strong repulsive force
                overlap = min_dist - d
                force_mag = k_repel * overlap
                fx[i] += force_mag * dx / d
                fy[i] += force_mag * dy / d
                fx[j] -= force_mag * dx / d
                fy[j] -= force_mag * dy / d
    
    # Wall constraints: force circles back into bounds
    for i in range(N):
        # Left wall
        if x[i] < r[i]:
            fx[i] += k_repel * (r[i] - x[i])
        # Right wall
        if x[i] > W - r[i]:
            fx[i] -= k_repel * (x[i] - (W - r[i]))
        # Bottom wall
        if y[i] < r[i]:
            fy[i] += k_repel * (r[i] - y[i])
        # Top wall
        if y[i] > H - r[i]:
            fy[i] -= k_repel * (y[i] - (H - r[i]))
    
    # Update positions
    x += fx * dt
    y += fy * dt
    
    # Clamp positions to valid range
    for i in range(N):
        x[i] = np.clip(x[i], r[i], W - r[i])
        y[i] = np.clip(y[i], r[i], H - r[i])
    
    # Try to expand radii
    for i in range(N):
        # Max radius from walls
        r_wall = min(x[i], W - x[i], y[i], H - y[i])
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
        # Gradually increase radius toward maximum
        if r_max > r[i]:
            r[i] += (r_max - r[i]) * 0.1
    
    return x, y, r


def _physics_relax(x0, y0, r0, W, H, time_budget, n_steps=5000):
    """
    Run physics-based relaxation to find a good packing.
    """
    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    r = r0.copy().astype(float)
    
    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    dt = 0.01
    k_repel = 50.0
    k_attract = 1.0
    
    for step in range(n_steps):
        if time.time() > time_budget:
            break
        
        # Adaptive step size: decrease as we converge
        dt_current = dt * (1.0 - 0.5 * step / n_steps)
        
        x, y, r = _physics_step(x, y, r, W, H, dt_current, k_repel, k_attract)
        
        # Check validity
        if _is_valid(x, y, r, W, H):
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    return best_x, best_y, best_r, best_val


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
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_value = 0.0
    best_result = None
    best_w = 1.0

    rng = np.random.default_rng(42)
    
    # Define a set of diverse initial aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15, 0.75, 1.25, 0.7, 1.3, 0.65]
    
    # Time budget per aspect ratio
    time_per_ar = (deadline - start_time - 10) / len(aspect_ratios)
    
    for ar_idx, w0 in enumerate(aspect_ratios):
        if time.time() > deadline - 10:
            break
            
        h0 = 2.0 - w0
        if h0 < 0.3 or w0 < 0.3:
            continue
        
        # Initialize positions in a grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0
            y[i] = (row + 0.5) / side * h0
            # Initial radius slightly smaller than cell size to ensure validity
            r[i] = min(w0 / side, h0 / side) * 0.3
            
        # Ensure initial state is valid
        if not _is_valid(x, y, r, w0, h0):
            # Shrink if invalid
            scale = 0.5
            r *= scale
            if not _is_valid(x, y, r, w0, h0):
                continue

        # Run physics-based relaxation
        step_deadline = start_time + (ar_idx + 1) * time_per_ar
        phys_x, phys_y, phys_r, phys_val = _physics_relax(
            x, y, r, w0, h0, step_deadline, n_steps=3000
        )
        
        if phys_val > best_value:
            best_value = phys_val
            best_result = np.column_stack((phys_x, phys_y, phys_r))
            best_w = w0
        
        # Also try greedy expansion from the physics result
        if time.time() < step_deadline - 1:
            gx, gy, gr = _greedy_expand(phys_x, phys_y, phys_r, w0, h0, step_deadline - 1)
            if _is_valid(gx, gy, gr, w0, h0):
                gval = np.sum(gr)
                if gval > best_value:
                    best_value = gval
                    best_result = np.column_stack((gx, gy, gr))
                    best_w = w0

    # Final polish
    if best_result is not None and time.time() < deadline - 5:
        x = best_result[:, 0].copy()
        y = best_result[:, 1].copy()
        r = best_result[:, 2].copy()
        w = best_w
        h = 2.0 - w
        
        # Try physics relaxation again from the best result
        phys_x, phys_y, phys_r, phys_val = _physics_relax(
            x, y, r, w, h, deadline - 1, n_steps=2000
        )
        if phys_val > best_value:
            best_value = phys_val
            best_result = np.column_stack((phys_x, phys_y, phys_r))
        
        # Try greedy expansion
        gx, gy, gr = _greedy_expand(x, y, r, w, h, deadline - 1)
        if _is_valid(gx, gy, gr, w, h):
            gval = float(np.sum(gr))
            if gval > best_value:
                best_value = gval
                best_result = np.column_stack((gx, gy, gr))

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety shrink
    x = best_result[:, 0]
    y = best_result[:, 1]
    r = best_result[:, 2]
    w = best_w
    h = 2.0 - w
    
    for _ in range(5):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
