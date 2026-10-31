"""Circle packing in perimeter-4 rectangle: improved force-directed pressure simulation."""
import numpy as np
import time

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        rs = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < rs[iu] ** 2 - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def simulate(w0, h0, time_budget, seed=0):
        """Run force-directed pressure simulation for a given rectangle."""
        rng = np.random.default_rng(seed)
        
        # Grid-based initialization to reduce initial chaos
        # Approximate number of circles per row/col
        aspect = w0 / h0
        if aspect >= 1.0:
            cols = int(np.sqrt(N * aspect))
            rows = int(np.ceil(N / cols))
        else:
            rows = int(np.sqrt(N / aspect))
            cols = int(np.ceil(N / rows))
        
        x = np.empty(N)
        y = np.empty(N)
        r = np.empty(N)
        
        # Fill grid
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                # Center of cell with jitter
                cx = (j + 0.5) * (w0 / cols)
                cy = (i + 0.5) * (h0 / rows)
                # Add small jitter to break symmetry
                x[idx] = cx + rng.uniform(-0.02, 0.02)
                y[idx] = cy + rng.uniform(-0.02, 0.02)
                # Initial radius estimate: min distance to wall/neighbor
                # Simple estimate: 0.5 * min(cell_width, cell_height) * 0.8
                cell_w = w0 / cols
                cell_h = h0 / rows
                r[idx] = 0.4 * min(cell_w, cell_h)
                idx += 1
        
        # Handle remaining if grid didn't fill exactly (should be rare with ceil)
        while idx < N:
            x[idx] = rng.uniform(0.1, w0 - 0.1)
            y[idx] = rng.uniform(0.1, h0 - 0.1)
            r[idx] = 0.05
            idx += 1

        # Clip to valid range
        x = np.clip(x, r, w0 - r)
        y = np.clip(y, r, h0 - r)

        vx = np.zeros(N)
        vy = np.zeros(N)
        dt = 0.005
        damping = 0.9
        iu = np.triu_indices(N, k=1)

        def forces(x, y, r, w, h, temp=1.0):
            fx = np.zeros(N)
            fy = np.zeros(N)
            # Wall repulsion (stronger to keep circles inside)
            k_wall = 100.0 * temp
            fx -= np.maximum(r - x, 0.0) * k_wall
            fx += np.maximum(x + r - w, 0.0) * k_wall
            fy -= np.maximum(r - y, 0.0) * k_wall
            fy += np.maximum(y + r - h, 0.0) * k_wall
            
            # Center attraction (weak, to keep packing compact)
            k_center = 0.2 * temp
            fx += (w * 0.5 - x) * k_center
            fy += (h * 0.5 - y) * k_center
            
            # Pairwise repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d2 = dx * dx + dy * dy
            rs = r[:, None] + r[None, :]
            
            # Only calculate for upper triangle to save time
            dp = np.sqrt(np.maximum(d2[iu], 1e-12))
            overlap = np.maximum(rs[iu] - dp, 0.0)
            
            # Force proportional to overlap, scaled by temperature
            k_pair = 30.0 * temp
            fp = overlap * k_pair
            
            ux = dx[iu] / dp
            uy = dy[iu] / dp
            
            for k in range(len(iu[0])):
                a = iu[0][k]; b = iu[1][k]
                fx[a] += ux[k] * fp[k]; fx[b] -= ux[k] * fp[k]
                fy[a] += uy[k] * fp[k]; fy[b] -= uy[k] * fp[k]
            return fx, fy

        def step(x, y, r, vx, vy, dt, damping, temp):
            fx, fy = forces(x, y, r, w0, h0, temp)
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            x = x + vx * dt
            y = y + vy * dt
            x = np.clip(x, r, w0 - r)
            y = np.clip(y, r, h0 - r)
            return x, y, vx, vy

        # Phase A: Settle positions with high temperature
        # We want to resolve overlaps first.
        # Start with a base radius and inflate.
        
        # Strategy: 
        # 1. Run simulation with current r, high temp, to remove overlaps.
        # 2. Inflate r slightly.
        # 3. Repeat.
        
        # Initial relaxation
        for it in range(200):
            x, y, vx, vy = step(x, y, r, vx, vy, dt, damping, 1.0)
            if time.time() > start_time + time_budget:
                break

        # Inflation loop
        # We will try to increase the sum of radii.
        # We can do this by scaling all radii up and re-optimizing.
        
        base_r = r.copy()
        max_sum = np.sum(base_r)
        
        # Try multiple inflation steps
        for infl_step in range(15):
            # Increase target scale
            scale = 1.0 + 0.05 * (infl_step + 1)
            target_r = base_r * scale
            
            # Animate r from current to target
            # Actually, just set r to target and let forces push them apart
            r = target_r.copy()
            
            # Run simulation to resolve overlaps
            # Use lower temperature as we go to settle more precisely
            temp = 1.0 - 0.03 * infl_step
            if temp < 0.1: temp = 0.1
            
            for it in range(150):
                x, y, vx, vy = step(x, y, r, vx, vy, dt, damping, temp)
                if time.time() > start_time + time_budget:
                    break
            
            # Check if valid
            valid, val = check_validity(x, y, r, w0, h0)
            if valid:
                if val > max_sum:
                    max_sum = val
                    base_r = r.copy()
            else:
                # If invalid, we might have overshot. 
                # We can try to shrink back a bit or just keep the last valid state.
                # For simplicity, we revert to last valid base_r if we want to be safe,
                # but usually force-directed settles into a valid state if we start valid.
                # If it becomes invalid, it means circles are overlapping.
                # We should continue simulation to fix it.
                pass
        
        return x, y, r, w0, h0

    def expand_uniform(x, y, r, w, h, time_budget):
        """Binary-search uniform radius scale that stays valid."""
        lo = 1.0
        hi = 1.20
        # find a feasible hi
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if check_validity(x, y, r * mid, w, h)[0]:
                lo = mid
            else:
                hi = mid
            if time.time() > start_time + time_budget:
                break
        return r * lo

    def relocate_polish(x, y, r, w, h, time_budget):
        """Nudge each circle to its max-radius position, then re-expand."""
        x = x.copy(); y = y.copy(); r = r.copy()
        iu = np.triu_indices(N, k=1)
        
        # Simple local search: try to increase radius of each circle
        # by moving it slightly in random directions or axis-aligned directions
        for rnd in range(10):
            if time.time() > start_time + time_budget:
                break
            improved = False
            for i in range(N):
                # Calculate max possible radius at current position
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                rmax = r_wall
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j])
                    rmax = min(rmax, d - r[j])
                rmax = max(0.0, rmax)
                
                if rmax > r[i] + 1e-6:
                    r[i] = rmax
                    improved = True
                
                # Try small moves to open more space for this circle
                # Only if we can increase its radius
                if r[i] < rmax - 1e-5:
                    # Try 8 directions
                    for dx, dy in [(0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01),
                                   (0.007, 0.007), (-0.007, 0.007), (0.007, -0.007), (-0.007, -0.007)]:
                        nx = x[i] + dx
                        ny = y[i] + dy
                        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                            continue
                        ok = True
                        for j in range(N):
                            if i == j:
                                continue
                            d2 = (nx - x[j]) ** 2 + (ny - y[j]) ** 2
                            if d2 < (r[i] + r[j]) ** 2 - 1e-9:
                                ok = False
                                break
                        if ok:
                            # Check if moving here allows larger radius
                            new_rmax = min(nx, w - nx, ny, h - ny)
                            for j in range(N):
                                if i == j:
                                    continue
                                d = np.hypot(nx - x[j], ny - y[j])
                                new_rmax = min(new_rmax, d - r[j])
                            new_rmax = max(0.0, new_rmax)
                            if new_rmax > r[i] + 1e-6:
                                x[i] = nx
                                y[i] = ny
                                r[i] = new_rmax
                                improved = True
                                break
            if not improved:
                break
        return x, y, r

    # Sweep a range of aspect ratios
    # Perimeter is 4, so 2(w+h) = 4 => w+h=2.
    # w ranges from 0.3 to 1.7 (symmetric)
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.66, 1.75, 0.6, 1.9, 0.55]
    
    # Shuffle to try diverse ones first? Or stick to order.
    # Let's iterate through them.
    
    for k, w0 in enumerate(aspect_ratios):
        h0 = 2.0 - w0
        if h0 < 0.3 or w0 < 0.3:
            continue
        remaining = deadline - (time.time() - start_time)
        if remaining < 4.0:
            break
        budget = min(5.0, remaining - 2.0)
        try:
            x, y, r, w, h = simulate(w0, h0, budget, seed=100 + k)
            # Uniform expansion
            r = expand_uniform(x, y, r, w, h, deadline - (time.time() - start_time) - 3)
            # Relocate polish
            x, y, r = relocate_polish(x, y, r, w, h, deadline - (time.time() - start_time) - 3)
            # Final expansion
            r = expand_uniform(x, y, r, w, h, deadline - (time.time() - start_time) - 1)
            valid, val = check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
        except Exception:
            continue

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    # Final safety shrink
    for _ in range(10):
        rt = r * (1 - 1e-9)
        if check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break

    return np.column_stack((x, y, r))
