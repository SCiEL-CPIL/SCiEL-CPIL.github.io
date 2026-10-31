"""Circle packing in perimeter-4 rectangle: Simulated Annealing + Local Search."""
import numpy as np
import time

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
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


def _compute_max_radius(cx, cy, r, W, H, i):
    """Compute maximum radius for circle i given current positions of others."""
    r_max = min(cx[i], W - cx[i], cy[i], H - cy[i])
    for j in range(N):
        if i == j:
            continue
        dx = cx[i] - cx[j]
        dy = cy[i] - cy[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _local_expand(cx, cy, r, W, H, max_iter=1000):
    """Iteratively expand each circle to its maximum possible radius."""
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(N):
            new_r = _compute_max_radius(cx, cy, r, W, H, i)
            if new_r > r[i] + 1e-10:
                r[i] = new_r
                improved = True
    return cx, cy, r


def _simulated_annealing(cx, cy, r, W, H, start_temp=0.05, end_temp=0.001,
                         steps_per_temp=50, n_temps=100, rng=None):
    """Simulated annealing to find better circle arrangements."""
    if rng is None:
        rng = np.random.default_rng()
    
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    current_value = np.sum(r)
    best_value = current_value
    best_cx = cx.copy()
    best_cy = cy.copy()
    best_r = r.copy()
    
    for t in range(n_temps):
        temp = start_temp * (end_temp / start_temp) ** (t / max(1, n_temps - 1))
        
        for _ in range(steps_per_temp):
            # Pick a random circle to move
            i = rng.integers(N)
            
            # Random perturbation
            dx = rng.uniform(-0.1, 0.1)
            dy = rng.uniform(-0.1, 0.1)
            
            new_cx = cx[i] + dx
            new_cy = cy[i] + dy
            
            # Clamp to walls
            new_cx = np.clip(new_cx, r[i], W - r[i])
            new_cy = np.clip(new_cy, r[i], H - r[i])
            
            # Check if move is valid
            valid = True
            for j in range(N):
                if i == j:
                    continue
                ddx = new_cx - cx[j]
                ddy = new_cy - cy[j]
                if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-8:
                    valid = False
                    break
            
            if not valid:
                # Try to accept with probability exp(-delta/temp)
                # Compute how much overlap there is
                min_dist = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    ddx = new_cx - cx[j]
                    ddy = new_cy - cy[j]
                    d = np.sqrt(ddx * ddx + ddy * ddy)
                    overlap = (r[i] + r[j]) - d
                    if overlap > 0:
                        min_dist = min(min_dist, overlap)
                
                if min_dist < 0.5 and rng.random() < np.exp(-min_dist / temp):
                    cx[i] = new_cx
                    cy[i] = new_cy
            else:
                cx[i] = new_cx
                cy[i] = new_cy
            
            # Try to expand circle i
            new_r_i = _compute_max_radius(cx, cy, r, W, H, i)
            if new_r_i > r[i]:
                r[i] = new_r_i
            
            # Update best
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_cx = cx.copy()
                best_cy = cy.copy()
                best_r = r.copy()
    
    return best_cx, best_cy, best_r


def _hill_climbing(cx, cy, r, W, H, max_iter=5000, step_size=0.01):
    """Hill climbing: try small moves and keep if they improve the sum of radii."""
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    rng = np.random.default_rng(42)
    
    current_value = np.sum(r)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try 8 directions
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                for step in [step_size, step_size * 0.5, step_size * 0.25]:
                    dx = np.cos(angle) * step
                    dy = np.sin(angle) * step
                    new_cx = cx[i] + dx
                    new_cy = cy[i] + dy
                    
                    # Clamp
                    new_cx = np.clip(new_cx, r[i], W - r[i])
                    new_cy = np.clip(new_cy, r[i], H - r[i])
                    
                    # Check validity
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        ddx = new_cx - cx[j]
                        ddy = new_cy - cy[j]
                        if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-8:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Try to expand
                    old_r_i = r[i]
                    new_r_i = _compute_max_radius(cx, cy, r, W, H, i)
                    
                    # Temporarily move and check
                    cx[i] = new_cx
                    cy[i] = new_cy
                    r[i] = new_r_i
                    
                    val = np.sum(r)
                    if val > current_value + 1e-10:
                        current_value = val
                        improved = True
                    else:
                        # Revert
                        cx[i] = new_cx - dx
                        cy[i] = new_cy - dy
                        r[i] = old_r_i
                    
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return cx, cy, r


def _optimize_aspect(cx, cy, r, W, H, max_iter=100):
    """Try different aspect ratios and re-pack."""
    best_value = np.sum(r)
    best_cx, best_cy, best_r, best_W = cx.copy(), cy.copy(), r.copy(), W
    
    for w_ratio in [W / 2.0 + delta for delta in [-0.1, -0.05, 0.05, 0.1]]:
        w_ratio = np.clip(w_ratio, 0.4, 1.6)
        new_W = w_ratio
        new_H = 2.0 - new_W
        
        if new_H < 0.3:
            continue
        
        # Scale positions
        scale_x = new_W / W
        scale_y = new_H / H
        new_cx = cx * scale_x
        new_cy = cy * scale_y
        
        # Scale radii to fit
        max_r = min(new_W, new_H) / 2.0
        new_r = np.minimum(r, max_r)
        
        # Check if valid
        if _is_valid(new_cx, new_cy, new_r, new_W, new_H):
            # Local expand
            new_cx, new_cy, new_r = _local_expand(new_cx, new_cy, new_r, new_W, new_H, max_iter=200)
            val = np.sum(new_r)
            if val > best_value:
                best_value = val
                best_cx, best_cy, best_r, best_W = new_cx, new_cy, new_r, new_W
    
    return best_cx, best_cy, best_r, best_W


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_cx, best_cy, best_r, best_W = None, None, None, None
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Try multiple initial configurations with different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.55, 1.82, 0.5, 2.0]
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, side)
            cx[i] = (col + 0.5) / side * w0
            cy[i] = (row + 0.5) / side * h0
            r[i] = min(w0 / side, h0 / side) * 0.4
        
        # Local expand
        cx, cy, r = _local_expand(cx, cy, r, w0, h0, max_iter=200)
        val = np.sum(r)
        
        if val > best_value:
            best_value = val
            best_cx, best_cy, best_r, best_W = cx.copy(), cy.copy(), r.copy(), w0
    
    # Phase 2: Simulated annealing on the best configuration
    if best_cx is not None and time.time() < deadline - 40:
        h0 = 2.0 - best_W
        cx_sa, cy_sa, r_sa = _simulated_annealing(
            best_cx, best_cy, best_r, best_W, h0,
            start_temp=0.03, end_temp=0.001,
            steps_per_temp=30, n_temps=80,
            rng=rng
        )
        
        # Local expand after SA
        cx_sa, cy_sa, r_sa = _local_expand(cx_sa, cy_sa, r_sa, best_W, h0, max_iter=500)
        val = np.sum(r_sa)
        
        if val > best_value:
            best_value = val
            best_cx, best_cy, best_r, best_W = cx_sa, cy_sa, r_sa, best_W
    
    # Phase 3: Hill climbing
    if best_cx is not None and time.time() < deadline - 25:
        h0 = 2.0 - best_W
        cx_hc, cy_hc, r_hc = _hill_climbing(
            best_cx, best_cy, best_r, best_W, h0,
            max_iter=3000, step_size=0.008
        )
        
        # Local expand
        cx_hc, cy_hc, r_hc = _local_expand(cx_hc, cy_hc, r_hc, best_W, h0, max_iter=500)
        val = np.sum(r_hc)
        
        if val > best_value:
            best_value = val
            best_cx, best_cy, best_r, best_W = cx_hc, cy_hc, r_hc, best_W
    
    # Phase 4: Aspect ratio optimization
    if best_cx is not None and time.time() < deadline - 15:
        h0 = 2.0 - best_W
        cx_opt, cy_opt, r_opt, W_opt = _optimize_aspect(
            best_cx, best_cy, best_r, best_W, h0,
            max_iter=200
        )
        
        if np.sum(r_opt) > best_value:
            best_value = np.sum(r_opt)
            best_cx, best_cy, best_r, best_W = cx_opt, cy_opt, r_opt, W_opt
    
    # Phase 5: Final local expansion
    if best_cx is not None and time.time() < deadline - 5:
        h0 = 2.0 - best_W
        best_cx, best_cy, best_r = _local_expand(best_cx, best_cy, best_r, best_W, h0, max_iter=1000)
        best_value = np.sum(best_r)
    
    # Final validation and safety shrink
    if best_cx is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    h0 = 2.0 - best_W
    if not _is_valid(best_cx, best_cy, best_r, best_W, h0):
        # Shrink radii until valid
        for scale in np.linspace(1.0, 0.9, 20):
            r_trial = best_r * scale
            if _is_valid(best_cx, best_cy, r_trial, best_W, h0):
                best_r = r_trial
                best_value = np.sum(best_r)
                break
    
    return np.column_stack((best_cx, best_cy, best_r))
