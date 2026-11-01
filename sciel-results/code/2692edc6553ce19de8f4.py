"""Circle packing in perimeter-4 rectangle: Force-Directed Relaxation with Iterative Growth."""
import numpy as np
import time

N = 21


def _force_relax(x, y, r, W, H, time_budget, dt=0.01, max_iter=2000):
    """
    Relax positions to resolve overlaps using force-directed method.
    Assumes r is fixed for this relaxation step.
    Returns updated x, y.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    
    # Precompute indices for pairs
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
            
    for _ in range(max_iter):
        if time.time() > time_budget:
            break
            
        # Check if we are "settled" (no significant forces)
        max_force = 0.0
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # 1. Wall forces
        for i in range(N):
            ri = r[i]
            # Left wall
            if x[i] < ri:
                f = (ri - x[i])
                fx[i] += f
                max_force = max(max_force, f)
            # Right wall
            if x[i] > W - ri:
                f = (x[i] - (W - ri))
                fx[i] -= f
                max_force = max(max_force, f)
            # Bottom wall
            if y[i] < ri:
                f = (ri - y[i])
                fy[i] += f
                max_force = max(max_force, f)
            # Top wall
            if y[i] > H - ri:
                f = (y[i] - (H - ri))
                fy[i] -= f
                max_force = max(max_force, f)
                
        # 2. Circle-circle repulsion
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d_min = r[i] + r[j]
            
            if d_sq < d_min * d_min:
                d = np.sqrt(d_sq)
                if d < 1e-6:
                    # Perfectly overlapping, push apart randomly
                    angle = np.random.uniform(0, 2*np.pi)
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                    d = 1e-6
                else:
                    # Normalized direction
                    dx /= d
                    dy /= d
                    
                # Overlap amount
                overlap = d_min - d
                # Force proportional to overlap (or overlap^2 for smoother)
                # Using linear force for stability
                f = overlap * 0.5 # Split force between i and j
                
                fx[i] += dx * f
                fy[i] += dy * f
                fx[j] -= dx * f
                fy[j] -= dy * f
                
                max_force = max(max_force, f)
                
        # If max force is very small, we are settled
        if max_force < 1e-5:
            break
            
        # Update positions
        x += dt * fx
        y += dt * fy
        
        # Project back into valid box [r_i, W-r_i]
        for i in range(N):
            ri = r[i]
            if x[i] < ri: x[i] = ri
            if x[i] > W - ri: x[i] = W - ri
            if y[i] < ri: y[i] = ri
            if y[i] > H - ri: y[i] = H - ri
            
    return x, y


def _try_grow_radii(x, y, r, W, H, time_budget, growth_step=0.005, max_grow_iters=1000):
    """
    Iteratively try to increase all radii.
    If growth causes overlap, relax positions.
    If relaxation fails to resolve, revert growth.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    # Check initial validity
    def _valid(cx, cy, cr, cW, cH):
        if np.any(cx - cr < -1e-9) or np.any(cy - cr < -1e-9):
            return False
        if np.any(cx + cr > cW + 1e-9) or np.any(cy + cr > cH + 1e-9):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                if dx*dx + dy*dy < (cr[i] + cr[j])**2 - 1e-9:
                    return False
        return True

    if not _valid(x, y, r, W, H):
        return x, y, r

    for _ in range(max_grow_iters):
        if time.time() > time_budget:
            break
        
        # Try to grow all radii
        r_new = r + growth_step
        
        # Check if new radii fit in walls
        valid_walls = True
        for i in range(N):
            if r_new[i] > x[i] or r_new[i] > W - x[i] or r_new[i] > y[i] or r_new[i] > H - y[i]:
                valid_walls = False
                break
        
        if not valid_walls:
            # Can't grow further due to walls
            # Maybe we can grow some circles that are not wall-limited?
            # For simplicity, stop growing if any circle is wall-limited.
            # Or, we can try to move circles to center to allow growth.
            # Let's try one more relaxation step to center them, then retry?
            # No, let's just stop. The force relaxation should have pushed them to optimal spots.
            break
            
        # Check if new radii cause overlaps
        # We don't need to check all pairs if we just relaxed, but growth might cause new overlaps.
        # Let's just attempt the growth and let the relaxation handle it.
        r = r_new
        
        # Relax positions to resolve overlaps
        x, y = _force_relax(x, y, r, W, H, time_budget - 1.0)
        
        # Check if the state is valid after relaxation
        if not _valid(x, y, r, W, H):
            # Relaxation failed to resolve overlaps, revert growth
            r = r - growth_step
            # Try to relax a bit more with the smaller radius?
            x, y = _force_relax(x, y, r, W, H, time_budget - 1.0)
            if not _valid(x, y, r, W, H):
                # If still invalid, we are stuck. Break.
                break
            else:
                # If valid, we can try smaller growth steps?
                # For now, stop growing.
                break
        else:
            # Valid, continue growing
            continue
            
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_value = 0.0
    best_result = None
    best_w = 1.0
    
    rng = np.random.default_rng(42)
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.2 or w0 < 0.2:
            continue
            
        # Initialize positions in a grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, side)
            # Add some jitter to break symmetry
            x[i] = (col + 0.5 + rng.normal(0, 0.05)) / side * w0
            y[i] = (row + 0.5 + rng.normal(0, 0.05)) / side * h0
            # Initial radius
            r[i] = min(w0 / side, h0 / side) * 0.4
            
        # Ensure initial state is valid
        # Shrink if needed
        max_r = min(np.min(x), np.min(w0 - x), np.min(y), np.min(h0 - y))
        for i in range(N):
            r[i] = min(r[i], max_r)
            
        # Run force-directed growth
        x, y, r = _try_grow_radii(x, y, r, w0, h0, deadline - 5.0)
        
        # Final validity check
        def _final_valid(cx, cy, cr, cW, cH):
            if np.any(cx - cr < -1e-9) or np.any(cy - cr < -1e-9):
                return False
            if np.any(cx + cr > cW + 1e-9) or np.any(cy + cr > cH + 1e-9):
                return False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    if dx*dx + dy*dy < (cr[i] + cr[j])**2 - 1e-9:
                        return False
            return True

        if _final_valid(x, y, r, w0, h0):
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_result = np.column_stack((x, y, r))
                best_w = w0

    # Fallback if no valid result found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
