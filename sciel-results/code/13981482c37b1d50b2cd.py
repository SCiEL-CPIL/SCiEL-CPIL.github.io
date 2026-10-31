"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE_FRACTION = 0.85  # safety margin

def _elapsed(start):
    return time.time() - start

def _is_feasible(pts, rads, tol=1e-9):
    """Check non-overlap and boundary feasibility."""
    n = len(rads)
    # boundary
    for i in range(n):
        x, y = pts[i]
        r = rads[i]
        if r < -tol:
            return False
        if x - r < -tol or y - r < -tol or (1 - x) - r < -tol or (1 - y) - r < -tol:
            return False
    # non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < rads[i] + rads[j] - tol:
                return False
    return True

def _clip_rads(pts, rads):
    """Clip radii to be feasible w.r.t. boundaries, and non-negative."""
    rads = np.maximum(rads, 0.0)
    for i in range(len(rads)):
        x, y = pts[i]
        max_r = min(x, y, 1 - x, 1 - y)
        if rads[i] > max_r:
            rads[i] = max(max_r, 0.0)
    return rads

def _penalty_objective(z, pts_init=None):
    """z = [x0,y0,r0, x1,y1,r1, ...]. Minimize negative sum of radii + penalties."""
    n = N
    pts = z[0::3].reshape(n, 2)
    rads = z[1::3].reshape(n)
    
    # Clip for evaluation
    rads_c = _clip_rads(pts, rads.copy())
    
    score = np.sum(rads_c)
    penalty = 0.0
    
    # Boundary penalty (soft)
    for i in range(n):
        x, y = pts[i]
        r = rads_c[i]
        b = min(x, y, 1 - x, 1 - y)
        if r > b:
            penalty += (r - b) ** 2
    
    # Non-overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            req = rads_c[i] + rads_c[j]
            if d < req:
                penalty += (req - d) ** 2
    
    return -score + penalty

def _random_initial():
    """Random feasible-ish initial configuration."""
    pts = np.random.rand(N, 2)
    rads = np.full(N, 0.5 / np.sqrt(N) * 0.7)
    rads = _clip_rads(pts, rads)
    return pts, rads

def _grid_initial():
    """Grid initial: 5x5 grid, take 25, plus one more."""
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side
            cells.append((x, y, r))
    # 25 cells; add a 26th at a corner-ish spot with small radius
    # Actually, let's do a 6x5 grid and take 26
    pts = []
    rads = []
    # 5 columns, 5 rows = 25
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            pts.append([x, y])
            rads.append(0.5 / 5.0)
    # 26th: place at center of a gap? Just put it at (0.5, 0.5) with small r, then let optimizer fix
    pts.append([0.5, 0.5])
    rads.append(0.05)
    pts = np.array(pts)
    rads = np.array(rads)
    rads = _clip_rads(pts, rads)
    return pts, rads

def _local_search(pts, rads, start_time, max_time):
    """Simulated annealing / hill climbing on z = [x,y,r]*N."""
    z = np.concatenate([pts.ravel(), rads.ravel()])
    best_z = z.copy()
    best_val = _penalty_objective(z)
    # We want to MINIMIZE the penalty objective; score = -obj
    current_val = best_val
    
    temp = 0.05
    cooling = 0.995
    step_scale = 0.05
    n_steps = 0
    
    while True:
        elapsed = _elapsed(start_time)
        if elapsed > max_time:
            break
        
        # Propose a move: perturb a random coordinate
        idx = np.random.randint(0, len(z))
        old_val = z[idx]
        z[idx] += np.random.randn() * step_scale
        
        # Keep positions in [0,1]
        if idx % 3 == 0 or idx % 3 == 1:
            if z[idx] < 0: z[idx] = 0.0
            if z[idx] > 1: z[idx] = 1.0
        # Radii non-negative
        if idx % 3 == 2:
            if z[idx] < 0: z[idx] = 0.0
            if z[idx] > 0.5: z[idx] = 0.5
        
        new_val = _penalty_objective(z)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_z = z.copy()
        else:
            # Accept with probability
            delta = new_val - current_val
            if temp > 1e-6:
                if np.random.rand() < np.exp(-delta / temp):
                    current_val = new_val
                else:
                    z[idx] = old_val
            else:
                z[idx] = old_val
        
        temp *= cooling
        step_scale *= 0.999
        n_steps += 1
        
        # Periodic restart if stuck
        if n_steps % 5000 == 0 and temp < 0.01:
            # Reheat slightly
            temp = 0.02
            step_scale = 0.03
    
    return best_z

def _slsqp_polish(z, start_time, max_time):
    """Final high-precision polish using SLSQP on the exact objective with constraints."""
    n = N
    def obj(z):
        pts = z[0::3].reshape(n, 2)
        rads = z[1::3].reshape(n)
        rads_c = _clip_rads(pts, rads.copy())
        return -np.sum(rads_c)
    
    # Constraints: non-overlap and boundary, as inequality g(z) >= 0
    cons = []
    
    # Boundary: x_i - r_i >= 0, y_i - r_i >= 0, (1-x_i) - r_i >= 0, (1-y_i) - r_i >= 0
    for i in range(n):
        x_idx = 3 * i
        y_idx = 3 * i + 1
        r_idx = 3 * i + 2
        cons.append({'type': 'ineq', 'fun': lambda z, a=x_idx, b=r_idx: z[a] - z[b]})
        cons.append({'type': 'ineq', 'fun': lambda z, a=y_idx, b=r_idx: z[a] - z[b]})
        cons.append({'type': 'ineq', 'fun': lambda z, a=x_idx, b=r_idx: 1 - z[a] - z[b]})
        cons.append({'type': 'ineq', 'fun': lambda z, a=y_idx, b=r_idx: 1 - z[a] - z[b]})
    
    # Non-overlap: dist_ij - r_i - r_j >= 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = 3*i, 3*i+1, 3*i+2
            xj, yj, rj = 3*j, 3*j+1, 3*j+2
            def nonoverlap(z, a=xi, b=yi, c=ri, d=xj, e=yj, f=rj):
                dx = z[a] - z[d]
                dy = z[b] - z[e]
                d2 = dx*dx + dy*dy
                d = np.sqrt(max(d2, 1e-12))
                return d - z[c] - z[f]
            cons.append({'type': 'ineq', 'fun': nonoverlap})
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-10, 'disp': False})
        if res.success and res.fun < _penalty_objective(z):
            return res.x
    except Exception:
        pass
    return z

def construct():
    start_time = time.time()
    max_time = 150.0 * DEADLINE_FRACTION  # 127.5s
    
    best_z = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = []
    
    # Grid initialization
    pts, rads = _grid_initial()
    inits.append(np.concatenate([pts.ravel(), rads.ravel()]))
    
    # Random initializations
    for _ in range(3):
        pts, rads = _random_initial()
        inits.append(np.concatenate([pts.ravel(), rads.ravel()]))
    
    for z_init in inits:
        elapsed = _elapsed(start_time)
        if elapsed > max_time - 30:
            break
        
        # Local search
        z_ls = _local_search(z_init.copy(), None, start_time, max_time - 10)
        score = -_penalty_objective(z_ls)
        if score > best_score:
            best_score = score
            best_z = z_ls.copy()
        
        elapsed = _elapsed(start_time)
        if elapsed > max_time - 5:
            break
    
    if best_z is None:
        # Fallback: return grid
        pts, rads = _grid_initial()
        return np.column_stack([pts, rads])
    
    # Final SLSQP polish if time permits
    elapsed = _elapsed(start_time)
    if elapsed < max_time - 2:
        z_polished = _slsqp_polish(best_z.copy(), start_time, max_time)
        score_polished = -_penalty_objective(z_polished)
        if score_polished > best_score:
            best_z = z_polished
            best_score = score_polished
    
    # Extract and clip final result
    pts = best_z[0::3].reshape(N, 2)
    rads = best_z[1::3].reshape(N)
    rads = _clip_rads(pts, rads)
    
    # Final feasibility check; if infeasible, reduce radii slightly
    if not _is_feasible(pts, rads, tol=1e-8):
        rads *= 0.999
        rads = _clip_rads(pts, rads)
    
    return np.column_stack([pts, rads])
