"""Circle packing in perimeter-4 rectangle: Enhanced SLSQP + Local Search Polish."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 21

def _soft_max(v, k=50.0):
    """Smooth approximation of max(v, 0) using log-sum-exp trick."""
    # For a scalar v: soft_max(v) ≈ (1/k)*log(1+exp(k*v))
    # This is smooth and approximates max(v,0) well
    if v > 50.0 / k:
        return v
    return (1.0 / k) * np.log1p(np.exp(k * v))

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

def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j: 
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)

def _tighten_positions(x, y, r, w, h, time_budget):
    """
    Use local search to minimize overlaps or maximize minimum distance.
    Here we use a simple force-directed relaxation with adaptive step size.
    """
    x = x.copy()
    y = y.copy()
    n = N
    dt = 0.005
    min_dt = 0.0001
    max_iter = int(min(2000, time_budget * 100))
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50
        
        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        
        if np.max(overlap) < 1e-9:
            break
            
        f_pairs = overlap * 20.0
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs
        
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fx[i_idx] += ux[idx] * f_pairs[idx]
            fy[i_idx] += uy[idx] * f_pairs[idx]
            fx[j_idx] -= ux[idx] * f_pairs[idx]
            fy[j_idx] -= uy[idx] * f_pairs[idx]
            
        # Adaptive step
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 0.5:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.05:
            dt = min(0.01, dt * 1.2)
            
        x += dt * fx
        y += dt * fy
        
        # Clamp
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
    return x, y

def _local_search_positions(x, y, r, w, h, time_budget, step_sizes=None):
    """
    8-direction local search to improve positions for better packing.
    Tries small moves in 8 directions and keeps the best valid improvement.
    """
    if step_sizes is None:
        step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    x = x.copy()
    y = y.copy()
    n = N
    directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
    
    for step in step_sizes:
        if time.time() > time_budget:
            break
        
        improved = False
        for _ in range(20):
            if time.time() > time_budget:
                break
            
            for i in range(n):
                for dx_dir, dy_dir in directions:
                    new_x = x[i] + dx_dir * step
                    new_y = y[i] + dy_dir * step
                    
                    # Check bounds
                    if new_x < r[i] or new_x > w - r[i]:
                        continue
                    if new_y < r[i] or new_y > h - r[i]:
                        continue
                    
                    # Check pairwise distances
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        d2 = dx*dx + dy*dy
                        r_sum = r[i] + r[j]
                        if d2 < r_sum*r_sum - 1e-10:
                            valid = False
                            break
                    
                    if valid:
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
        
        if not improved:
            continue
    
    return x, y

def _expand_and_relax(x, y, r, w, h, time_budget):
    """
    Iteratively expand each circle to its max possible radius, then tighten positions.
    Includes local search for better positioning.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = N
    max_rounds = 50
    
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        
        # Step 1: Expand each circle
        for i in range(n):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-7:
                r[i] = r_max
                improved = True
        
        # Step 2: Tighten positions to resolve overlaps
        x, y = _tighten_positions(x, y, r, w, h, time_budget - 0.5)
        
        # Step 3: Local search to improve positions
        if time.time() < time_budget - 1:
            x, y = _local_search_positions(x, y, r, w, h, time_budget - 1)
        
        if not improved:
            # Try one more pass of expansion after tightening
            for i in range(n):
                r_max = _max_radius_for_circle(x, y, r, w, h, i)
                if r_max > r[i] + 1e-7:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
                
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_value = 0.0
    best_result = None
    
    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0
        
        val = float(np.sum(r))
        return True, val

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        
        # Grid-based
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
            
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
            
        return guesses

    guesses = get_initial_guesses()
    
    # Phase 1: SLSQP Optimization
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break
            
        # Quick relaxation
        x0 = z0[0::3][:N].copy()
        y0 = z0[1::3][:N].copy()
        r0 = z0[2::3][:N].copy()
        w0 = z0[-1]
        h0 = 2.0 - w0
        
        # Force relax to resolve initial overlaps
        x_rel, y_rel = _tighten_positions(x0, y0, r0, w0, h0, time.time() + 0.5)
        
        z_relaxed = np.zeros(3 * N + 1)
        z_relaxed[0::3][:N] = x_rel
        z_relaxed[1::3][:N] = y_rel
        z_relaxed[2::3][:N] = r0
        z_relaxed[-1] = w0
        
        # SLSQP with penalty (using smooth soft-max for better convergence)
        penalty_weight = 1000.0
        soft_k = 50.0
        
        def objective(z):
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w = z[-1]
            h = 2.0 - w
            
            if w < 0.2 or h < 0.2 or np.any(r < 0):
                return 1e6
                
            obj = -np.sum(r)
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]
            
            iu = np.triu_indices(N, k=1)
            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            # Use soft-max for smooth penalty
            overlap_soft = np.array([_soft_max(r_sum_pairs[i] - d_pairs[i], soft_k) for i in range(len(d_pairs))])
            overlap_penalty = penalty_weight * np.sum(overlap_soft * overlap_soft)
            
            # Smooth boundary penalties
            left_viol = np.array([_soft_max(r[i] - x[i], soft_k) for i in range(N)])
            right_viol = np.array([_soft_max(x[i] + r[i] - w, soft_k) for i in range(N)])
            bottom_viol = np.array([_soft_max(r[i] - y[i], soft_k) for i in range(N)])
            top_viol = np.array([_soft_max(y[i] + r[i] - h, soft_k) for i in range(N)])
            
            boundary_penalty = penalty_weight * (
                np.sum(left_viol * left_viol) +
                np.sum(right_viol * right_viol) +
                np.sum(bottom_viol * bottom_viol) +
                np.sum(top_viol * top_viol)
            )
            
            obj += overlap_penalty + boundary_penalty
            
            if w < 0.5:
                obj += penalty_weight * (0.5 - w) ** 2
            elif w > 1.5:
                obj += penalty_weight * (w - 1.5) ** 2
                
            return obj

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 20)))
        
        try:
            res = minimize(objective, z_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
            
            z_opt = res.x
            valid, val = check_validity(z_opt)
            
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try repair
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()
                    
        except Exception:
            continue
            
    # Phase 2: Polish
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w
        
        # Run expand and relax
        time_budget = deadline - 5
        x, y, r = _expand_and_relax(x, y, r, w, h, time_budget)
        
        # Check validity and update
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z_new = np.zeros(3 * N + 1)
                z_new[0::3][:N] = x
                z_new[1::3][:N] = y
                z_new[2::3][:N] = r
                z_new[-1] = w
                best_result = z_new
        
        # Phase 3: Second expansion pass after SLSQP refinement
        if best_result is not None and time.time() < deadline - 3:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            
            # Do a final tight expansion
            for i in range(N):
                r_max = _max_radius_for_circle(x, y, r, w, h, i)
                if r_max > r[i] + 1e-8:
                    r[i] = r_max
            
            # Final local search
            if time.time() < deadline - 1:
                x, y = _local_search_positions(x, y, r, w, h, deadline - 1)
            
            # Check validity and update
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    z_new = np.zeros(3 * N + 1)
                    z_new[0::3][:N] = x
                    z_new[1::3][:N] = y
                    z_new[2::3][:N] = r
                    z_new[-1] = w
                    best_result = z_new

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w
    
    # Safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
