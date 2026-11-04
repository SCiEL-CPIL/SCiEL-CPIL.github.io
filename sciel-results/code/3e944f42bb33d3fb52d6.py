"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
_DEADLINE = 145.0  # 5s safety margin from 150s

# EVOLVE-BLOCK-START

def _remaining(deadline):
    return max(0.0, deadline - time.time())

def _compute_radii(pos, n, max_iter=500, tol=1e-13):
    """Compute maximum valid radius for each circle given positions."""
    radii = np.zeros(n)
    for i in range(n):
        x, y = pos[i]
        radii[i] = min(x, y, 1.0 - x, 1.0 - y)
    
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            x, y = pos[i]
            r_bound = min(x, y, 1.0 - x, 1.0 - y)
            r_max = r_bound
            for j in range(n):
                if i == j:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_candidate = dist - radii[j]
                if r_candidate < r_max:
                    r_max = r_candidate
            r_new = max(0.0, r_max)
            if abs(r_new - radii[i]) > tol:
                radii[i] = r_new
                changed = True
        if not changed:
            break
    return radii

def _grow_radii(pos, radii, n, max_iter=300):
    """Grow all radii uniformly while maintaining feasibility."""
    radii = radii.copy()
    for _ in range(max_iter):
        delta_max = np.inf
        for i in range(n):
            x, y = pos[i]
            delta_max = min(delta_max, x - radii[i], y - radii[i],
                           1.0 - x - radii[i], 1.0 - y - radii[i])
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                delta_max = min(delta_max, (dist - radii[i] - radii[j]) / 2.0)
        
        if delta_max <= 1e-13:
            break
        delta = delta_max * 0.95
        if delta < 1e-12:
            break
        radii += delta
    return radii

def _is_valid(pos, radii, n, tol=1e-9):
    """Check if packing is valid."""
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _project_feasible(pos, radii, n):
    """Project to feasibility by reducing radii as needed."""
    pos = pos.copy()
    radii = radii.copy()
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
    for _ in range(200):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                md = radii[i] + radii[j]
                if d < md - 1e-10:
                    excess = md - d
                    reduction = excess / 2.0 + 1e-7
                    radii[i] = max(0.0, radii[i] - reduction)
                    radii[j] = max(0.0, radii[j] - reduction)
                    changed = True
        if not changed:
            break
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
    return pos, radii

def _score(pos, radii, n):
    """Return sum of radii if valid, else -inf."""
    if _is_valid(pos, radii, n):
        return np.sum(radii)
    return -np.inf

def _soft_objective(params, n, beta):
    """Smooth soft objective: minimize -sum(r) + beta*(overlap^2 + boundary^2)."""
    pos = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Pairwise overlaps
    overlap_sum = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy + 1e-15)
            excess = radii[i] + radii[j] - d
            if excess > 0:
                overlap_sum += excess * excess
    
    # Boundary violations
    bound_sum = 0.0
    for i in range(n):
        bound_sum += max(0, radii[i] - pos[i, 0])**2
        bound_sum += max(0, radii[i] - pos[i, 1])**2
        bound_sum += max(0, radii[i] - (1 - pos[i, 0]))**2
        bound_sum += max(0, radii[i] - (1 - pos[i, 1]))**2
    
    return -np.sum(radii) + beta * (overlap_sum + bound_sum)

def _soft_grad(params, n, beta):
    """Gradient of soft objective."""
    pos = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    grad = np.zeros_like(params)
    
    for i in range(n):
        # Gradient w.r.t. radius i
        grad[2*n + i] = -1.0
        for j in range(n):
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy + 1e-15)
            excess = radii[i] + radii[j] - d
            if excess > 0:
                grad[2*n + i] += 2 * beta * excess
                # Position gradient for i
                grad[2*i] += -2 * beta * excess * dx / d
                grad[2*i + 1] += -2 * beta * excess * dy / d
        # Boundary gradients
        for k, (val, idx) in enumerate([(pos[i,0], 2*i), (pos[i,1], 2*i+1),
                                          (1-pos[i,0], 2*i), (1-pos[i,1], 2*i+1)]):
            v = radii[i] - val
            if v > 0:
                grad[2*n + i] += 2 * beta * v
                sign = 1.0 if k < 2 else -1.0
                grad[idx] += 2 * beta * v * sign
    
    return grad

def _gradient_optimize(pos, radii, n, deadline, max_beta=500.0):
    """L-BFGS-B optimization with smooth soft objective."""
    if _remaining(deadline) < 3:
        return pos, radii
    
    params = np.zeros(3*n)
    params[:2*n] = pos.flatten()
    params[2*n:] = radii
    
    bounds = [(0, 1)] * 2*n + [(0, 0.5)] * n
    
    best_params = params.copy()
    best_score = _score(pos, radii, n)
    
    for beta in [20.0, 100.0, 300.0, max_beta]:
        if _remaining(deadline) < 5:
            break
        try:
            result = minimize(
                _soft_objective, _soft_grad, best_params,
                args=(n, beta), method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_pos = result.x[:2*n].reshape(n, 2)
            opt_radii = np.maximum(0, result.x[2*n:])
            opt_pos, opt_radii = _project_feasible(opt_pos, opt_radii, n)
            sc = _score(opt_pos, opt_radii, n)
            if sc > best_score:
                best_score = sc
                best_params = np.concatenate([opt_pos.flatten(), opt_radii])
        except Exception:
            continue
    
    bp = best_params[:2*n].reshape(n, 2)
    br = best_params[2*n:]
    br = _compute_radii(bp, n)
    return bp, br

def _langevin_dynamics(pos, radii, n, deadline, steps=5000, T0=0.01, T1=0.0001):
    """Langevin dynamics (hard-disk) to escape local minima."""
    if _remaining(deadline) < 10:
        return pos, radii
    
    pos = pos.copy()
    rng = np.random.RandomState(12345)
    step_base = 0.003
    
    for step in range(steps):
        if _remaining(deadline) < 5:
            break
        T = T0 * (T1/T0) ** (step / max(1, steps-1))
        
        for i in range(n):
            # Compute forces from overlaps
            fx, fy = 0.0, 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy + 1e-15)
                overlap = radii[i] + radii[j] - d
                if overlap > 0:
                    f = 50.0 * overlap
                    fx += f * dx / d
                    fy += f * dy / d
            
            # Boundary forces
            if pos[i, 0] < radii[i]:
                fx += 50.0 * (radii[i] - pos[i, 0])
            if pos[i, 1] < radii[i]:
                fy += 50.0 * (radii[i] - pos[i, 1])
            if pos[i, 0] > 1 - radii[i]:
                fx -= 50.0 * (pos[i, 0] - (1 - radii[i]))
            if pos[i, 1] > 1 - radii[i]:
                fy -= 50.0 * (pos[i, 1] - (1 - radii[i]))
            
            # Langevin update
            noise = T * rng.normal(size=2)
            pos[i, 0] += step_base * (fx + noise[0])
            pos[i, 1] += step_base * (fy + noise[1])
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)
        
        # Recompute radii periodically
        if step % 100 == 0:
            radii = _compute_radii(pos, n)
            grown = _grow_radii(pos, radii, n)
            if np.sum(grown) > np.sum(radii):
                radii = grown
    
    radii = _compute_radii(pos, n)
    radii = _grow_radii(pos, radii, n)
    return pos, radii

def _discrete_best_fit(pos, radii, n, deadline, step_sizes=None):
    """Discrete best-fit local search."""
    if step_sizes is None:
        step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    for step_size in step_sizes:
        if _remaining(deadline) < 3:
            break
        improved = True
        while improved:
            if _remaining(deadline) < 3:
                break
            improved = False
            for i in range(n):
                orig_x, orig_y = best_pos[i]
                best_local_sum = best_sum
                best_local_pos = best_pos.copy()
                best_local_radii = best_radii.copy()
                
                candidates = []
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = orig_x + dx, orig_y + dy
                        if 0.001 <= nx <= 0.999 and 0.001 <= ny <= 0.999:
                            candidates.append((nx, ny))
                
                diag = step_size * 0.7071
                for dx in [-diag, diag]:
                    for dy in [-diag, diag]:
                        nx, ny = orig_x + dx, orig_y + dy
                        if 0.001 <= nx <= 0.999 and 0.001 <= ny <= 0.999:
                            candidates.append((nx, ny))
                
                for cx, cy in candidates:
                    trial_pos = best_local_pos.copy()
                    trial_pos[i] = [cx, cy]
                    trial_radii = _compute_radii(trial_pos, n)
                    trial_sum = np.sum(trial_radii)
                    if trial_sum > best_local_sum + 1e-9:
                        best_local_sum = trial_sum
                        best_local_pos = trial_pos
                        best_local_radii = trial_radii
                
                if best_local_sum > best_sum + 1e-9:
                    best_pos = best_local_pos
                    best_radii = best_local_radii
                    best_sum = best_local_sum
                    improved = True
    
    grown = _grow_radii(best_pos, best_radii, n)
    if np.sum(grown) > best_sum:
        best_radii = grown
    return best_pos, best_radii

def _initial_hexagonal(n):
    pos = np.zeros((n, 2))
    idx = 0
    row = 0
    while idx < n:
        offset = 0.05 if row % 2 == 1 else 0.0
        for col in range(6):
            if idx >= n:
                break
            x = (col + 0.5) / 6.0 + offset
            y = (row + 0.5) / 6.0
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pos[idx] = [x, y]
            idx += 1
        row += 1
    return pos

def _initial_grid(n):
    pos = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(6):
            if idx >= n:
                break
            pos[idx] = [(i + 0.5)/5.0, (j + 0.5)/6.0]
            idx += 1
    return pos

def _initial_structured(n):
    pos = np.zeros((n, 2))
    idx = 0
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for c in corners:
        if idx < n:
            pos[idx] = c; idx += 1
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for e in edges:
        if idx < n:
            pos[idx] = e; idx += 1
    for i in range(3):
        for j in range(3):
            if idx >= n:
                break
            pos[idx] = [0.25 + i*0.25, 0.25 + j*0.25]; idx += 1
    rng = np.random.RandomState(42)
    while idx < n:
        pos[idx] = rng.uniform(0.15, 0.85, size=2); idx += 1
    return pos

def _initial_random(n, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.zeros((n, 2))
    valid = []
    attempts = 0
    while len(valid) < n and attempts < 2000:
        p = rng.uniform(0.1, 0.9, size=2)
        ok = True
        for vp in valid:
            if np.linalg.norm(p - vp) < 0.12:
                ok = False; break
        if ok:
            valid.append(p)
        attempts += 1
    while len(valid) < n:
        valid.append(rng.uniform(0.1, 0.9, size=2))
    pos[:] = valid
    return pos

def _repulsion_init(n, seed=0, iters=400):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    for it in range(iters):
        forces = np.zeros_like(pos)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                dx = pos[i,0]-pos[j,0]; dy = pos[i,1]-pos[j,1]
                d = np.sqrt(dx*dx+dy*dy)+1e-10
                f = 1.0/(d*d)
                forces[i,0] += f*dx/d; forces[i,1] += f*dy/d
            forces[i,0] -= 0.05*(pos[i,0]-0.5)
            forces[i,1] -= 0.05*(pos[i,1]-0.5)
        step = 0.05*(1.0-it/iters)
        pos += step*forces
        pos = np.clip(pos, 0.05, 0.95)
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    best_result = None
    best_score = -np.inf
    
    def try_update(pos, radii):
        nonlocal best_result, best_score
        pos, radii = _project_feasible(pos, radii, N)
        sc = _score(pos, radii, N)
        if sc > best_score:
            best_score = sc
            best_result = np.column_stack([pos, radii])
    
    # Generate initial configurations
    configs = []
    configs.append(_initial_hexagonal(N))
    configs.append(_initial_grid(N))
    configs.append(_initial_structured(N))
    for s in range(6):
        configs.append(_initial_random(N, seed=s))
    for s in range(3):
        configs.append(_repulsion_init(N, seed=s))
    
    for pos in configs:
        if _remaining(deadline) < 15:
            break
        radii = _compute_radii(pos, N)
        radii = _grow_radii(pos, radii, N)
        
        # Langevin dynamics phase
        if _remaining(deadline) > 25:
            lp, lr = _langevin_dynamics(pos, radii, N, deadline, steps=3000)
            try_update(lp, lr)
            pos, radii = lp, lr
        
        # Gradient optimization
        if _remaining(deadline) > 12:
            gp, gr = _gradient_optimize(pos, radii, N, deadline)
            try_update(gp, gr)
            pos, radii = gp, gr
        
        # Discrete refinement
        if _remaining(deadline) > 8:
            dp, dr = _discrete_best_fit(pos, radii, N, deadline)
            try_update(dp, dr)
    
    # Final polish on best result
    if best_result is not None and _remaining(deadline) > 8:
        pp = best_result[:, :2].copy()
        pr = best_result[:, 2].copy()
        gp, gr = _gradient_optimize(pp, pr, N, deadline)
        dp, dr = _discrete_best_fit(gp, gr, N, deadline, step_sizes=[0.002, 0.001, 0.0005, 0.0002, 0.0001])
        grown = _grow_radii(dp, dr, N)
        try_update(dp, grown)
    
    if best_result is not None and _remaining(deadline) > 5:
        pp = best_result[:, :2].copy()
        pr = best_result[:, 2].copy()
        dp, dr = _discrete_best_fit(pp, pr, N, deadline, step_sizes=[0.0005, 0.0002, 0.0001])
        grown = _grow_radii(dp, dr, N)
        try_update(dp, grown)
    
    if best_result is None:
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(6):
                if idx >= N: break
                result[idx] = [(i+0.5)/5.0, (j+0.5)/6.0, 0.07]
                idx += 1
        if idx < N:
            result[idx] = [0.5, 0.5, 0.05]
        return result
    
    return best_result

# EVOLVE-BLOCK-END
