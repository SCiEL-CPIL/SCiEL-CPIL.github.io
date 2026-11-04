import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    # Safety margin: return at least 5 seconds before the 150s hard limit
    deadline = start_time + 145.0
    
    best_total = 0.0
    best_result = None

    def _check_validity(xy, r, tol=1e-9):
        """
        Strictly check if the packing is valid.
        Returns (is_valid, total_radius).
        """
        n = len(xy)
        # 1. Boundary check
        for i in range(n):
            x, y, ri = xy[i]
            if x - ri < -tol or 1.0 - x - ri < -tol or y - ri < -tol or 1.0 - y - ri < -tol:
                return False, 0.0
        
        # 2. Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    return False, 0.0
        return True, np.sum(r)

    def _get_max_radius(x, y, others_xy, others_r):
        """
        Calculate the maximum radius a circle at (x,y) can have 
        given a list of other fixed circles.
        """
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if max_r <= 0:
            return 0.0
        for px, py, pr in others_xy:
            dx = x - px
            dy = y - py
            dist = np.sqrt(dx * dx + dy * dy)
            r_lim = dist - pr
            if r_lim < max_r:
                max_r = r_lim
                if max_r <= 0:
                    return 0.0
        return max_r

    def _jittered_expansion(xy, r, others_idx, time_limit):
        """
        A different optimization method: sequential hard-constrained expansion.
        We iterate over circles in a specific order. For each circle, we try to 
        increase its radius by jittering its position. We only accept a change if
        the circle remains strictly non-overlapping with all others.
        """
        n = len(xy)
        start = time.time()
        
        # Use a random order for expansion to break symmetry
        order = np.random.permutation(n)
        
        # We iterate until we run out of time or no improvements are made
        while time.time() < time_limit:
            improved = False
            for i in order:
                if time.time() > time_limit:
                    break
                
                others = np.array([j for j in range(n) if j != i])
                others_xy = xy[others]
                others_r = r[others]
                
                current_r = r[i]
                best_local_r = current_r
                best_local_x = xy[i, 0]
                best_local_y = xy[i, 1]
                
                # Try a few random jittered positions around the current one
                # The jitter magnitude scales with the current radius
                jitter_scale = max(0.01, current_r * 0.5)
                
                for _ in range(20):
                    # Random jitter
                    jx = np.random.uniform(-jitter_scale, jitter_scale)
                    jy = np.random.uniform(-jitter_scale, jitter_scale)
                    new_x = np.clip(xy[i, 0] + jx, 0.0, 1.0)
                    new_y = np.clip(xy[i, 1] + jy, 0.0, 1.0)
                    
                    # Calculate max radius at this new position
                    max_r = _get_max_radius(new_x, new_y, others_xy, others_r)
                    
                    if max_r > best_local_r + 1e-9:
                        best_local_r = max_r
                        best_local_x = new_x
                        best_local_y = new_y
                        
                # If we found a better spot, update
                if best_local_r > current_r + 1e-9:
                    xy[i, 0] = best_local_x
                    xy[i, 1] = best_local_y
                    r[i] = best_local_r
                    improved = True
            
            if not improved:
                break

        return xy, r

    def _nelder_mead_polish(xy, r, time_limit):
        """
        Final polish using Nelder-Mead on the negative sum of radii, 
        with a huge penalty for violations to guide it towards feasibility 
        and then maximum radius.
        """
        n = len(xy)
        params = np.zeros(3 * n)
        params[0::3] = xy[:, 0]
        params[1::3] = xy[:, 1]
        params[2::3] = r
        
        def objective(p):
            x = p[0::3]
            y = p[1::3]
            rad = np.maximum(p[2::3], 1e-6) # Ensure positive radii for calculation
            
            # Wall penalty
            pen = 0.0
            for i in range(n):
                pen += max(0, rad[i] - x[i])**2
                pen += max(0, rad[i] - (1-x[i]))**2
                pen += max(0, rad[i] - y[i])**2
                pen += max(0, rad[i] - (1-y[i]))**2
            
            # Overlap penalty
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_d = rad[i] + rad[j]
                    if dist < min_d:
                        pen += (min_d - dist)**2
            
            # We want to maximize sum(rad), so minimize -sum(rad) + penalty
            return -np.sum(rad) + 1000.0 * pen

        try:
            # NM is not great with bounds, but the penalty should keep it in check
            # We pass the current best as the start
            res = minimize(objective, params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            opt_x = res.x[0::3]
            opt_y = res.x[1::3]
            opt_r = np.maximum(res.x[2::3], 0.0)
            
            # Clip to box to ensure boundary validity
            for i in range(n):
                opt_r[i] = min(opt_r[i], opt_x[i], 1-opt_x[i], opt_y[i], 1-opt_y[i])
                opt_x[i] = np.clip(opt_x[i], opt_r[i], 1-opt_r[i])
                opt_y[i] = np.clip(opt_y[i], opt_r[i], 1-opt_r[i])
            
            return np.column_stack([opt_x, opt_y]), opt_r
        except Exception:
            return xy, r

    def _generate_initial_configs():
        """
        Generate diverse starting points using different geometric strategies.
        """
        configs = []
        
        # 1. Hexagonal-like packing
        # Approximate hex packing for 26 circles
        # Rows of 5, 5, 5, 5, 4, 2? Or just a grid that gets optimized.
        # Let's use a 5x5 grid + 1, or 6x5 grid.
        # 6x5 = 30, take 26.
        side = 6
        rows = 5
        # Center of cell
        dx = 1.0 / side
        dy = 1.0 / rows
        r_init = 0.5 * min(dx, dy) * 0.95
        positions = []
        for i in range(rows):
            for j in range(side):
                if len(positions) < N:
                    # Offset odd rows for hex-like structure
                    offset = 0.5 * dx if i % 2 == 1 else 0.0
                    x = (j + 0.5) * dx + offset
                    y = (i + 0.5) * dy
                    # Clip x to [0,1]
                    x = np.clip(x, r_init, 1-r_init)
                    positions.append([x, y, r_init])
        configs.append(np.array(positions))
        
        # 2. Random uniform
        positions = []
        for _ in range(N):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = 0.05
            positions.append([x, y, r])
        configs.append(np.array(positions))
        
        # 3. Grid 5x5 + 1
        side = 5
        positions = []
        dx = 1.0 / side
        dy = 1.0 / side
        r_init = 0.5 * dx * 0.95
        for i in range(side):
            for j in range(side):
                if len(positions) < N:
                    x = (j + 0.5) * dx
                    y = (i + 0.5) * dy
                    positions.append([x, y, r_init])
        # Add one more in the center or somewhere
        if len(positions) < N:
            positions.append([0.5, 0.5, 0.05])
        configs.append(np.array(positions[:N]))
        
        return configs

    # Main Loop
    inits = _generate_initial_configs()
    num_restarts = 10 # We will run as many as time allows
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Pick an initial config, cycling through them
        init_pos = inits[restart % len(inits)].copy()
        
        # Add some random noise to the initial positions for diversity
        noise_scale = 0.05
        init_pos[:, 0] += np.random.uniform(-noise_scale, noise_scale, N)
        init_pos[:, 1] += np.random.uniform(-noise_scale, noise_scale, N)
        
        xy = init_pos[:, :2]
        r = init_pos[:, 2].copy()
        
        # Clip initial radii to be valid within the box
        for i in range(N):
            r[i] = min(r[i], xy[i,0], 1-xy[i,0], xy[i,1], 1-xy[i,1])
            
        # Phase 1: Jittered Expansion (Hard Constraints)
        # Allocate ~60% of remaining time to this
        remaining = deadline - time.time()
        time_limit_phase1 = time.time() + remaining * 0.6
        xy, r = _jittered_expansion(xy, r, None, time_limit_phase1)
        
        # Check validity and score
        valid, total = _check_validity(xy, r)
        if valid and total > best_total:
            best_total = total
            best_result = np.column_stack([xy, r]).copy()
        
        # Phase 2: Nelder-Mead Polish
        if time.time() < deadline - 2.0:
            time_limit_phase2 = deadline - 2.0
            # NM doesn't have a built-in time limit, so we just run it with maxiter
            # and hope it finishes. To be safe, we can run it in a way that we check time?
            # No, minimize is a C call. We must rely on maxiter.
            # Let's set a reasonable maxiter that takes < 10s usually.
            
            # Actually, let's wrap it to be safe? No, standard approach is to trust maxiter.
            # 1000 iterations of NM on 78 variables might be slow.
            # Let's reduce maxiter to 200 for safety, or use a smaller subset?
            # NM is local. It should be fast.
            
            new_xy, new_r = _nelder_mead_polish(xy, r, time_limit_phase2)
            
            # Validate NM result
            valid2, total2 = _check_validity(new_xy, new_r)
            if valid2 and total2 > best_total:
                best_total = total2
                best_result = np.column_stack([new_xy, new_r]).copy()

    # Fallback if no valid result was found
    if best_result is None:
        # Simple grid fallback
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    return best_result
