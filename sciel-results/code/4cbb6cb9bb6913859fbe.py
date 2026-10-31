"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-8):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Check dist >= sum of radii
    if np.any(np.sqrt(dist_sq[mask]) < min_dist[mask] - tol):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(300):
        valid = True
        
        # Boundary
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        
        # Overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Shrink both proportionally
                    k = max(0.0, (dist - 1e-9) / sum_r)
                    if k < 1.0:
                        radii[i] = max(radii[i] * k * 0.99999, 1e-6)
                        radii[j] = max(radii[j] * k * 0.99999, 1e-6)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _objective(x, penalty_weight=1000.0):
    """
    Objective: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist - np.sqrt(dist_sq + 1e-12))
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_grad(x, penalty_weight=1000.0):
    """Gradient of the objective function."""
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    
    grad = np.zeros_like(x)
    
    # Gradient of -sum(radii)
    grad[2*N:] = -1.0
    
    # Boundary penalty gradient
    for i in range(N):
        r = radii[i]
        x_i, y_i = pos[i]
        
        # Left wall
        if r - x_i > 0:
            grad[2*i] += 2.0 * penalty_weight * (r - x_i)  # d/dx
            grad[2*N+i] += 2.0 * penalty_weight * (r - x_i)  # d/dr
        
        # Right wall
        if x_i - (1.0 - r) > 0:
            grad[2*i] -= 2.0 * penalty_weight * (x_i - (1.0 - r))
            grad[2*N+i] += 2.0 * penalty_weight * (x_i - (1.0 - r))
        
        # Bottom wall
        if r - y_i > 0:
            grad[2*i+1] += 2.0 * penalty_weight * (r - y_i)
            grad[2*N+i] += 2.0 * penalty_weight * (r - y_i)
        
        # Top wall
        if y_i - (1.0 - r) > 0:
            grad[2*i+1] -= 2.0 * penalty_weight * (y_i - (1.0 - r))
            grad[2*N+i] += 2.0 * penalty_weight * (y_i - (1.0 - r))
    
    # Overlap penalty gradient
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            min_d = radii[i] + radii[j]
            
            if min_d - dist > 0:
                viol = min_d - dist
                coef = 2.0 * penalty_weight * viol / dist
                
                # d/dx_i
                grad[2*i] += coef * dx
                grad[2*i+1] += coef * dy
                # d/dx_j
                grad[2*j] -= coef * dx
                grad[2*j+1] -= coef * dy
                # d/dr_i and d/dr_j
                grad[2*N+i] += 2.0 * penalty_weight * viol
                grad[2*N+j] += 2.0 * penalty_weight * viol
    
    return grad

def _initial_grid():
    """Grid-based initialization with corner circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with 4 corner circles
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.12
    
    # Add edge-midpoint circles
    edges = [(0.5, 0.06), (0.5, 0.94), (0.06, 0.5), (0.94, 0.5)]
    for i, (ex, ey) in enumerate(edges):
        idx = 4 + i
        pos[idx] = [ex, ey]
        radii[idx] = 0.06
    
    # Fill remaining with smaller circles in a grid pattern
    remaining = N - 8
    rng = np.random.RandomState(42)
    for k in range(remaining):
        idx = 8 + k
        # Try random positions
        best_pos = None
        best_r = 0.0
        for _ in range(2000):
            x = rng.uniform(0.03, 0.97)
            y = rng.uniform(0.03, 0.97)
            max_r = min(x, 1-x, y, 1-y)
            for j in range(idx):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            if max_r > best_r and max_r > 0.015:
                best_r = max_r
                best_pos = (x, y)
        
        if best_pos is not None and best_r > 0.01:
            pos[idx] = best_pos
            radii[idx] = best_r * 0.95
        else:
            # Fallback
            pos[idx] = [0.5 + 0.2*(k % 2), 0.5 + 0.2*((k//2) % 2)]
            radii[idx] = 0.03
    
    return pos, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.105
    dx = r * np.sqrt(3) * 1.02
    dy = r * 1.5 * 1.02
    
    k = 0
    for row in range(5):
        offset = dx / 2 if row % 2 == 1 else 0
        y = row * dy + r + 0.02
        
        for col in range(5):
            if k >= N:
                break
            x = col * dx + r + offset + 0.01
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r * 0.98
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        placed = False
        for _ in range(500):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                placed = True
                break
        if not placed:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.02
            k += 1
            
    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.13, 0.13), (0.87, 0.13), (0.13, 0.87), (0.87, 0.87)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02)]
        radii[i] = 0.13
    
    for i in range(4, N):
        # Bias toward edges and corners
        if rng.random() < 0.5:
            edge = rng.randint(4)
            t = rng.uniform(0.15, 0.85)
            if edge == 0:
                x, y = t, rng.uniform(0.04, 0.12)
            elif edge == 1:
                x, y = t, rng.uniform(0.88, 0.96)
            elif edge == 2:
                x, y = rng.uniform(0.04, 0.12), t
            else:
                x, y = rng.uniform(0.88, 0.96), t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        
        if max_r < 0.012:
            # Try more random positions
            for attempt in range(300):
                x2 = rng.uniform(0.03, 0.97)
                y2 = rng.uniform(0.03, 0.97)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.008) * 0.96
        
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.08, 0.92, (N, 2))
    radii = rng.uniform(0.03, 0.10, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=5000, t0=0.04, t_end=0.0005, seed=0, deadline=None):
    """
    Simulated annealing: perturb positions and radii, accept based on energy.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Current energy (penalty-based)
    x_curr = np.concatenate([pos.flatten(), radii])
    current_energy = _objective(x_curr, 500.0)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        # Perturb: move one or two circles
        n_perturb = 1 if rng.random() < 0.7 else 2
        for _ in range(n_perturb):
            i = rng.randint(n)
            scale = 0.015 * (1.0 + temp * 5.0)
            dx = rng.normal(0, scale)
            dy = rng.normal(0, scale)
            dr = rng.normal(0, scale * 0.5)
            
            new_x = pos[i, 0] + dx
            new_y = pos[i, 1] + dy
            new_r = radii[i] + dr
            
            # Clamp
            new_x = np.clip(new_x, 0.005, 0.995)
            new_y = np.clip(new_y, 0.005, 0.995)
            new_r = np.clip(new_r, 0.005, 0.45)
            
            old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
            pos[i, 0] = new_x
            pos[i, 1] = new_y
            radii[i] = new_r
        
        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 500.0)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            # Reject - restore (simplified: just recompute from best)
            # For efficiency, we accept the rejection implicitly by not updating
            pass
        
        # Periodically check validity and update best
        if step % 50 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing with L-BFGS-B using analytic gradients."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))
        
    # L-BFGS-B with analytic gradient
    if time.time() < deadline - 3:
        try:
            result = minimize(_objective, x, jac=_objective_grad, method='L-BFGS-B', 
                              bounds=bounds, args=(2000.0,),
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20})
            if result.fun < _objective(x, 2000.0) - 1e-12:
                x = result.x
        except Exception:
            pass
    
    # Second pass with higher penalty
    if time.time() < deadline - 2:
        try:
            result = minimize(_objective, x, jac=_objective_grad, method='L-BFGS-B', 
                              bounds=bounds, args=(10000.0,),
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20})
            if result.fun < _objective(x, 10000.0) - 1e-12:
                x = result.x
        except Exception:
            pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    return pos_out, radii_out

def _local_optimize(pos, radii, deadline):
    """
    Local optimization: try to increase each radius and move circles.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(12345)
    
    # Try to increase each radius
    for i in range(n):
        if time.time() > deadline:
            break
            
        old_r = radii[i]
        for delta_r in [0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002]:
            radii[i] = old_r + delta_r
            
            if _is_valid(pos, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            else:
                radii[i] = old_r
                break
        
    # Try moving circles to free up space
    for _ in range(500):
        if time.time() > deadline:
            break
            
        i = rng.randint(n)
        dx = rng.normal(0, 0.008)
        dy = rng.normal(0, 0.008)
        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        
        if 0.01 <= new_x <= 0.99 and 0.01 <= new_y <= 0.99:
            old_x, old_y = pos[i, 0], pos[i, 1]
            pos[i, 0] = new_x
            pos[i, 1] = new_y
            
            if _is_valid(pos, radii):
                # Try to increase radius
                old_r = radii[i]
                for delta_r in [0.005, 0.003, 0.001, 0.0005]:
                    radii[i] = old_r + delta_r
                    if _is_valid(pos, radii):
                        score = np.sum(radii)
                        if score > best_score:
                            best_score = score
                            best_pos = pos.copy()
                            best_radii = radii.copy()
                    else:
                        radii[i] = old_r
                        break
            else:
                pos[i, 0] = old_x
                pos[i, 1] = old_y
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_grid, rad_grid = _initial_grid()
    inits.append((pos_grid, rad_grid))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    pos_cb2, rad_cb2 = _initial_corner_biased(seed=13)
    inits.append((pos_cb2, rad_cb2))
    
    for i in range(2):
        pos_r, rad_r = _initial_random(200 + i * 50)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_grid.copy()
    best_radii = rad_grid.copy()
    best_score = np.sum(best_radii)
    
    # Initial post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if _is_valid(best_pos, best_radii):
        best_score = np.sum(best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Post-process initial
        pos_pp, radii_pp = _post_process(pos_init, rad_init)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        current_pos = pos_pp.copy()
        current_radii = radii_pp.copy()
        
        # Stage 1: L-BFGS-B optimization with increasing penalties
        for p_w in [200.0, 1000.0, 5000.0]:
            if time.time() > deadline - 10:
                break
            x = np.concatenate([current_pos.flatten(), current_radii])
            bounds = []
            for i in range(N):
                bounds.append((0.005, 0.995))
                bounds.append((0.005, 0.995))
            for i in range(N):
                bounds.append((0.005, 0.5))
            try:
                result = minimize(_objective, x, jac=_objective_grad, method='L-BFGS-B', 
                                  bounds=bounds, args=(p_w,),
                                  options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 25})
                if result.fun < _objective(x, p_w) - 1e-10:
                    current_pos = result.x[:2*N].reshape(N, 2)
                    current_radii = np.maximum(result.x[2*N:], 1e-4)
                    
                    pos_out, radii_out = _post_process(current_pos, current_radii)
                    if _is_valid(pos_out, radii_out):
                        score = np.sum(radii_out)
                        if score > best_score:
                            best_score = score
                            best_pos = pos_out.copy()
                            best_radii = radii_out.copy()
            except Exception:
                pass
        
        # Stage 2: Simulated annealing
        if time.time() < deadline - 12:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_pos, current_radii,
                    n_steps=3000, t0=0.03, t_end=0.0003,
                    seed=idx * 100 + 42, deadline=deadline - 8
                )
                pos_out, radii_out = _post_process(pos_sa, rad_sa)
                if _is_valid(pos_out, radii_out):
                    score = np.sum(radii_out)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_out.copy()
                        best_radii = radii_out.copy()
            except Exception:
                pass
        
        # Stage 3: Local optimization
        if time.time() < deadline - 6:
            try:
                pos_lo, rad_lo = _local_optimize(
                    current_pos, current_radii, deadline=deadline - 4
                )
                pos_out, radii_out = _post_process(pos_lo, rad_lo)
                if _is_valid(pos_out, radii_out):
                    score = np.sum(radii_out)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_out.copy()
                        best_radii = radii_out.copy()
            except Exception:
                pass
    
    # Final polish on the best solution found
    if time.time() < deadline - 3:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_out, radii_out = _post_process(pos_pol, rad_pol)
            if _is_valid(pos_out, radii_out):
                score = np.sum(radii_out)
                if score > best_score:
                    best_score = score
                    best_pos = pos_out.copy()
                    best_radii = radii_out.copy()
        except Exception:
            pass
    
    # One more local optimization pass on the final best
    if time.time() < deadline - 2:
        try:
            pos_lo, rad_lo = _local_optimize(best_pos, best_radii, deadline=deadline - 1)
            pos_out, radii_out = _post_process(pos_lo, rad_lo)
            if _is_valid(pos_out, radii_out):
                score = np.sum(radii_out)
                if score > best_score:
                    best_score = score
                    best_pos = pos_out.copy()
                    best_radii = radii_out.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback: valid configuration
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        pos[k] = [0.5, 0.015]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
