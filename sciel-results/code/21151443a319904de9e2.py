"""Circle packing in perimeter-4 rectangle: Physics-based Soft-Body Simulation."""
import numpy as np
import time
import math

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


def _physics_packing(W, H, time_budget, seed=42):
    """
    Use a soft-body physics simulation to pack circles.
    We start with small circles and let them grow while repelling each other.
    """
    rng = np.random.default_rng(seed)
    
    # Initial positions: random within the rectangle
    x = rng.uniform(0.1, W - 0.1, N)
    y = rng.uniform(0.1, H - 0.1, N)
    r = np.full(N, 0.01)
    
    # Physics parameters
    dt = 0.005
    repulsion_k = 100.0
    wall_k = 1000.0
    damping = 0.9
    
    # We will simulate for a fixed number of steps, increasing radius slightly each step
    # or just let them repel and settle. 
    # A better strategy: Start with small radii, simulate to reach equilibrium, then increase radii.
    
    # Let's use a "growth" approach.
    # Step 1: Find a valid non-overlapping configuration for small circles.
    # Step 2: Gradually increase radii, re-simulating to resolve overlaps.
    
    max_iter = 2000
    current_time = time.time()
    
    # Initialize velocities
    vx = np.zeros(N)
    vy = np.zeros(N)
    
    # We want to maximize sum(r).
    # Strategy: 
    # 1. Start with small r.
    # 2. Run simulation to remove overlaps.
    # 3. Increase r by a small delta.
    # 4. Repeat until time runs out or no more growth is possible.
    
    growth_rate = 0.0005
    stagnation_count = 0
    max_stagnation = 100
    
    while time.time() < time_budget and stagnation_count < max_stagnation:
        # Increase radii
        r += growth_rate
        
        # Run simulation steps to resolve overlaps
        sim_steps = 50
        for step in range(sim_steps):
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    dist = math.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        # Repulsive force proportional to overlap
                        overlap = min_dist - dist
                        force = repulsion_k * overlap
                        fx[i] += (dx / dist) * force
                        fy[i] += (dy / dist) * force
                        fx[j] -= (dx / dist) * force
                        fy[j] -= (dy / dist) * force
            
            # Wall repulsion
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += wall_k * (r[i] - x[i])
                if x[i] > W - r[i]:
                    fx[i] -= wall_k * (x[i] - (W - r[i]))
                if y[i] < r[i]:
                    fy[i] += wall_k * (r[i] - y[i])
                if y[i] > H - r[i]:
                    fy[i] -= wall_k * (y[i] - (H - r[i]))
            
            # Update velocities and positions
            vx = (vx + fx * dt) * damping
            vy = (vy + fy * dt) * damping
            x += vx * dt
            y += vy * dt
            
            # Hard clamp to prevent escaping rectangle (safety)
            x = np.clip(x, 0, W)
            y = np.clip(y, 0, H)
            
            # If we are still overlapping significantly, we might need to adjust
            # but the forces should handle it.
        
        # Check if we made progress (reduced overlap)
        # For now, we just assume growth is good if we didn't explode.
        # To be more robust, we could check if the configuration is "stable" enough.
        
        # If radii are too large, they might not fit. 
        # We can detect this if the simulation doesn't converge (high residual force).
        # Simple check: if max radius > min(W,H)/2 * 0.9, we are likely stuck.
        
        stagnation_count += 1
        if stagnation_count % 10 == 0:
            # Check for validity or "validity-like" metric
            # If we have huge overlaps, maybe reduce growth rate or stop
            pass
    
    return x, y, r


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 50:
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

    # Try different aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.65, 0.85, 1.15]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        
        if W < 0.3 or H < 0.3:
            continue
            
        # Run physics simulation
        # Give it a chunk of time
        chunk_time = min(8.0, deadline - time.time())
        if chunk_time < 2.0:
            continue
            
        try:
            x, y, r = _physics_packing(W, H, time.time() + chunk_time, seed=int(ar*100))
            
            # Polish with greedy expansion
            x, y, r = _greedy_expand(x, y, r, W, H, time.time() + 1.0)
            
            # Validate
            if _is_valid(x, y, r, W, H):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    # Store result
                    res = np.empty(3*N+1)
                    res[0::3][:N] = x
                    res[1::3][:N] = y
                    res[2::3][:N] = r
                    res[-1] = W
                    best_result = res
        except Exception:
            continue

    if best_result is None:
        # Fallback to a simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final Polish
    if time.time() < deadline - 2:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w
        
        x, y, r = _greedy_expand(x, y, r, w, h, deadline - 1)
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = np.empty(3*N+1)
                best_result[0::3][:N] = x
                best_result[1::3][:N] = y
                best_result[2::3][:N] = r
                best_result[-1] = w

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
