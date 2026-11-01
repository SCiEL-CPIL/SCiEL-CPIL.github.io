"""
Packing 50 equal circles in the unit square [-0.5, 0.5]^2.
Combined approach:
- Fast vectorized scoring via cKDTree
- Hexagonal lattice initialization
- Simulated Annealing for global exploration (escaping local minima)
- Nelder-Mead for final gradient-free polish
"""

import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50

def _score(pts):
    """Return the achievable common radius for a centre array (N,2)."""
    if not np.all(np.isfinite(pts)):
        return -np.inf
    # Boundary term: distance from each centre to the nearest side.
    d_bound = np.min(0.5 - np.abs(pts), axis=1)
    r = float(np.min(d_bound))
    # Pairwise term: half the minimum centre distance.
    if N > 1:
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        dmin = float(np.min(d[:, 1]))
        r = min(r, dmin / 2.0)
    return r if r > 0 else 0.0

def _valid(pts):
    return (pts.shape == (N, 2)
            and np.all(np.isfinite(pts))
            and np.all(pts >= -0.5 - 1e-9)
            and np.all(pts <= 0.5 + 1e-9))

def _neg_radius(params):
    """Objective: minimize negative radius (i.e., maximize radius)."""
    points = params.reshape(-1, 2)
    if np.any(np.abs(points) > 0.5 + 1e-8):
        return 1e6
    r = _score(points)
    if r <= 0:
        return 1e6
    return -r

def _hex_initial():
    """Hexagonal (triangular) lattice, rows staggered, clipped to square."""
    # Optimal spacing for 50 points in unit square is roughly s ~ 0.15
    s = 0.152
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    xs = np.arange(-0.5, 0.5 + 1e-9, dx)
    pts = []
    row = 0
    y = -0.5
    while y <= 0.5 + 1e-9:
        offset = (row % 2) * (dx / 2.0)
        for x in xs:
            xx = x + offset
            if -0.5 <= xx <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([xx, y])
        y += dy
        row += 1
    pts = np.array(pts, dtype=float)
    if len(pts) < N:
        # Pad with grid if needed
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = np.vstack([pts, grid[len(pts):]])
    else:
        # Keep N points closest to the centre to maximise boundary margin
        d2 = np.sum(pts * pts, axis=1)
        order = np.argsort(d2)
        pts = pts[order[:N]]
    return pts.copy()

def _grid_initial():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _force_directed(pts, rng, time_budget, lr_init=0.01, lr_min=1e-6, decay=0.9995):
    """
    Force-directed repulsion: each circle pushes others away (inverse-square)
    and is pushed away from walls. Gradient descent on the potential.
    This is a different optimization family than SA - it uses deterministic
    forces rather than random perturbations.
    """
    best = pts.copy()
    best_s = _score(best)
    curr = pts.copy()
    curr_s = _score(curr)
    
    lr = lr_init
    t0 = time.time()
    patience = 0
    patience_limit = 500  # Stop if no improvement for this many steps
    
    while time.time() - t0 < time_budget:
        if lr < lr_min:
            break
        
        # Compute forces
        # 1. Pairwise repulsion: push apart if too close
        # 2. Wall repulsion: push away from walls if too close
        
        r_target = curr_s * 2.0 + 0.01  # Target separation slightly above current min distance
        
        # Pairwise forces using cKDTree for efficiency
        tree = cKDTree(curr)
        idx = tree.query_ball_point(curr, r_target * 2.0)
        
        forces = np.zeros_like(curr)
        for i in range(N):
            for j in idx[i]:
                if i == j:
                    continue
                d_vec = curr[i] - curr[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    # Random tiny push for coincident points
                    forces[i] += rng.normal(0, 0.001, size=2)
                    continue
                if d < r_target:
                    # Repulsive force proportional to (r_target - d)
                    f_mag = (r_target - d) / d
                    forces[i] += f_mag * d_vec
        
        # Wall forces: push away from each wall
        margin = 0.01
        for i in range(N):
            # Left wall
            if curr[i, 0] < -0.5 + margin:
                forces[i, 0] += (0.5 - margin - curr[i, 0]) / margin
            # Right wall
            if curr[i, 0] > 0.5 - margin:
                forces[i, 0] -= (curr[i, 0] - (0.5 - margin)) / margin
            # Bottom wall
            if curr[i, 1] < -0.5 + margin:
                forces[i, 1] += (0.5 - margin - curr[i, 1]) / margin
            # Top wall
            if curr[i, 1] > 0.5 - margin:
                forces[i, 1] -= (curr[i, 1] - (0.5 - margin)) / margin
        
        # Gradient descent step
        curr = curr + lr * forces
        curr = np.clip(curr, -0.4999999, 0.4999999)
        
        # Score
        curr_s = _score(curr)
        
        if curr_s > best_s:
            best = curr.copy()
            best_s = curr_s
            patience = 0
        else:
            patience += 1
            if patience > patience_limit:
                break
        
        # Decay learning rate
        lr *= decay
        
    return best, best_s

def _simulated_annealing(pts, rng, time_budget):
    """
    Simulated Annealing to maximize radius.
    Allows accepting worse moves to escape local minima.
    """
    best = pts.copy()
    best_s = _score(best)
    curr = pts.copy()
    curr_s = _score(curr)
    
    # SA Parameters
    T = 0.005  # Initial temperature (scale of radius improvement)
    T_min = 1e-6
    cooling = 0.999  # Cooling rate per step
    
    t0 = time.time()
    step_size = 0.02
    
    while time.time() - t0 < time_budget:
        if T < T_min:
            break
            
        # Perturb a random subset of points
        n_move = rng.integers(1, max(2, N // 5) + 1)
        idx = rng.choice(N, size=n_move, replace=False)
        
        cand = curr.copy()
        cand[idx] += rng.normal(0.0, step_size, size=(n_move, 2))
        cand = np.clip(cand, -0.4999999, 0.4999999)
        
        cand_s = _score(cand)
        
        if cand_s > curr_s:
            # Always accept improvement
            curr = cand
            curr_s = cand_s
            if curr_s > best_s:
                best = curr.copy()
                best_s = curr_s
        else:
            # Accept worse move with probability exp(-(curr_s - cand_s) / T)
            delta = curr_s - cand_s
            if delta < 1e-12:
                curr = cand
                curr_s = cand_s
            else:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    curr = cand
                    curr_s = cand_s
        
        T *= cooling
    
    return best, best_s

def _polish(pts, deadline):
    """Gradient-free polish using Nelder-Mead."""
    best_points = pts.copy()
    best_r = _score(best_points)
    x0 = best_points.flatten()
    try:
        # Use Nelder-Mead for final refinement
        result = minimize(
            _neg_radius, x0, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < _neg_radius(x0):
            new_points = result.x.reshape(-1, 2)
            new_points = np.clip(new_points, -0.49999999, 0.49999999)
            r_new = _score(new_points)
            if r_new > best_r:
                best_points = new_points
                best_r = r_new
    except Exception:
        pass
    return best_points, best_r

def construct():
    start = time.time()
    deadline = start + 115.0  # Leave 5s safety margin for 120s limit

    rng = np.random.default_rng(42)

    # Initial candidates
    candidates = [_hex_initial(), _grid_initial()]
    best_pts = candidates[0].copy()
    best_s = _score(best_pts)
    for c in candidates[1:]:
        s = _score(c)
        if s > best_s:
            best_s = s
            best_pts = c.copy()

    # Multi-restart with force-directed + SA hybrid
    # Force-directed is the primary engine; SA for global escapes
    
    restart_configs = [
        {'init': 'hex', 'method': 'force', 'time_frac': 0.35},
        {'init': 'hex_perturb', 'method': 'force', 'time_frac': 0.25},
        {'init': 'grid_perturb', 'method': 'sa', 'time_frac': 0.2},
        {'init': 'best', 'method': 'force', 'time_frac': 0.15},
    ]
    
    time_left = deadline - start - 10.0  # Reserve 10s for polish
    
    for cfg in restart_configs:
        if time.time() > deadline - 15.0:  # Stop if too close to polish time
            break
            
        t_alloc = time_left * cfg['time_frac']
        if t_alloc < 2.0:
            continue
            
        # Prepare initial points
        if cfg['init'] == 'hex':
            cur = _hex_initial()
        elif cfg['init'] == 'hex_perturb':
            cur = _hex_initial()
            cur += rng.normal(0.0, 0.02, size=(N, 2))
            cur = np.clip(cur, -0.4999999, 0.4999999)
        elif cfg['init'] == 'grid_perturb':
            cur = _grid_initial()
            cur += rng.normal(0.0, 0.03, size=(N, 2))
            cur = np.clip(cur, -0.4999999, 0.4999999)
        else:  # 'best'
            cur = best_pts.copy()
            
        if not _valid(cur):
            cur = best_pts.copy()
            
        # Run optimization
        if cfg['method'] == 'force':
            cur, s = _force_directed(cur, rng, t_alloc)
        else:  # 'sa'
            cur, s = _simulated_annealing(cur, rng, t_alloc)
        
        if _valid(cur) and s > best_s:
            best_s = s
            best_pts = cur.copy()
            
        time_left = deadline - start - time.time() - 10.0  # Update remaining time

    # Final polish with gradient-free optimization
    if time.time() < deadline - 2.0:
        best_pts, best_s = _polish(best_pts, deadline)

    # Final safety
    best_pts = np.clip(best_pts, -0.499999999999, 0.499999999999)
    if not _valid(best_pts):
        best_pts = _grid_initial()

    return best_pts

# EVOLVE-BLOCK-START
# construct() is the entry point
# EVOLVE-BLOCK-END
