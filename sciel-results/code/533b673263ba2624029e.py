import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import KDTree

N = 200
HALF = 0.5
TIME_LIMIT = 145.0

def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # inside square
    m = np.max(np.abs(c), axis=1)
    if np.any(m > HALF + 1e-12):
        return 0.0
    # pairwise min distance
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return min(min_pair / 2.0, HALF - np.max(m))

def _hex_points(s, ox, oy):
    """Hexagonal lattice points with spacing s, offset (ox, oy)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    cols = int(np.ceil(1.0 / dx)) + 3
    rows = int(np.ceil(1.0 / dy)) + 3
    xs = np.arange(-2, cols) * dx + ox
    ys = np.arange(-2, rows) * dy + oy
    # offset every other column
    pts = []
    for j, x in enumerate(xs):
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i, y in enumerate(ys):
            pts.append((x + 0.0, y + yoff))
    arr = np.array(pts)
    # keep inside square
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    return arr[keep]

def _greedy_select(cands, k, r_min):
    """Greedy farthest-point selection of k points maximizing min distance."""
    n = cands.shape[0]
    if n <= k:
        return cands.copy()
    selected = []
    idx = int(np.random.randint(n))
    selected.append(idx)
    min_dist = np.full(n, np.inf)
    for _ in range(k - 1):
        last = cands[selected[-1]]
        d = np.linalg.norm(cands - last, axis=1)
        min_dist = np.minimum(min_dist, d)
        j = int(np.argmax(min_dist))
        if min_dist[j] < r_min - 1e-9:
            break
        selected.append(j)
    return cands[selected]

def _soft_min_obj_and_grad(x_flat, beta):
    """
    Objective: maximize soft-min of distances and margins.
    x_flat: shape (2N,)
    Returns: (objective_value, gradient) for minimization (negative of max).
    """
    x = x_flat.reshape(N, 2)
    
    # Compute pairwise distances
    diff = x[:, None, :] - x[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Margins
    margins = HALF - np.max(np.abs(x), axis=1)
    
    # Mask for pairs close enough to matter
    mask = dists < 1.0
    
    d_flat = dists[mask]
    m_flat = margins
    
    # Compute weights for soft-min
    exp_d = np.exp(-beta * d_flat)
    exp_m = np.exp(-beta * m_flat)
    S = np.sum(exp_d) + np.sum(exp_m)
    
    if S == 0:
        return 0.0, np.zeros_like(x_flat)
        
    w_d = exp_d / S
    w_m = exp_m / S
    
    # Extract indices for non-zero weights
    i_idx, j_idx = np.where(mask)
    valid = i_idx != j_idx
    i_idx = i_idx[valid]
    j_idx = j_idx[valid]
    w_d_valid = w_d
    
    # Compute gradient
    grad_x = np.zeros_like(x)
    
    # Pairwise contributions
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = w_d_valid[k]
        if w < 1e-10: continue
        d_ij = dists[i, j]
        if d_ij < 1e-10: continue
        v = (x[i] - x[j]) / d_ij
        grad_x[i] += w * v
        grad_x[j] -= w * v
        
    # Margin contributions
    for i in range(N):
        w = w_m[i]
        if w < 1e-10: continue
        ax = np.abs(x[i, 0])
        ay = np.abs(x[i, 1])
        if ax > ay:
            grad_x[i, 0] -= w * np.sign(x[i, 0])
        else:
            grad_x[i, 1] -= w * np.sign(x[i, 1])
            
    # Objective value for minimization
    obj_val = (1.0 / beta) * np.log(S)
    
    # Gradient for minimization
    grad_flat = -grad_x.flatten()
    
    return obj_val, grad_flat

def _gradient_polish(pos, time_budget=60.0):
    """
    Local search using gradient ascent on soft-min objective.
    """
    t0 = time.time()
    pos = pos.copy().astype(float)
    
    # Start with a small beta and increase it (annealing)
    beta = 10.0
    
    bounds = [(-HALF, HALF)] * (2 * N)
    
    best_r = _score(pos)
    best_pos = pos.copy()
    
    while time.time() - t0 < time_budget:
        x_flat = pos.flatten()
        
        # Define objective and gradient for this beta
        def obj_grad(x):
            val, grad = _soft_min_obj_and_grad(x, beta)
            return val, grad
            
        # Use L-BFGS-B
        try:
            res = minimize(
                obj_grad, 
                x_flat, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds,
                options={'maxiter': 50, 'gtol': 1e-5}
            )
            pos = res.x.reshape(N, 2)
        except Exception:
            break
            
        r = _score(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
            
        # Increase beta to sharpen the minimum
        beta *= 1.5
        if beta > 1000:
            break
            
    return best_pos, best_r

def _force_directed_optimize(pts, r_target, max_iters=3000, learning_rate=0.002, cooling=0.999):
    """
    Force-directed optimization to maximize the minimum distance.
    """
    pts = pts.copy()
    current_lr = learning_rate
    
    for it in range(max_iters):
        if time.time() > TIME_LIMIT - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Build KDTree for neighbor queries
        tree = KDTree(pts)
        
        # For each point, find neighbors within some radius
        cutoff = max(2 * r_target, 0.05)
        
        for i in range(N):
            # Wall forces: push points away from boundaries
            # Left wall
            if pts[i, 0] < -0.5 + r_target:
                forces[i, 0] += 5.0 * (r_target - (pts[i, 0] + 0.5))
            # Right wall
            if pts[i, 0] > 0.5 - r_target:
                forces[i, 0] -= 5.0 * (pts[i, 0] - (0.5 - r_target))
            # Bottom wall
            if pts[i, 1] < -0.5 + r_target:
                forces[i, 1] += 5.0 * (r_target - (pts[i, 1] + 0.5))
            # Top wall
            if pts[i, 1] > 0.5 - r_target:
                forces[i, 1] -= 5.0 * (pts[i, 1] - (0.5 - r_target))
            
            # Repulsive forces from neighbors
            neighbors = tree.query_ball_point(pts[i], cutoff)
            for j in neighbors:
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random nudge for coincident points
                    forces[i] += np.random.randn(2) * 0.01
                    continue
                if dist < 2 * r_target:
                    # Repulsive force: stronger when closer
                    overlap = 2 * r_target - dist
                    force_mag = 10.0 * overlap / dist
                    forces[i] += force_mag * diff
        
        # Update positions
        pts += current_lr * forces
        
        # Project back to square (hard constraint)
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Cool down
        current_lr *= cooling
        
        # Early stopping if forces are very small
        if np.max(np.abs(forces)) < 1e-6:
            break
    
    return pts

def construct():
    t0 = time.time()
    DEADLINE = TIME_LIMIT - 5.0
    
    rng = np.random.default_rng(2024)
    
    best_c = None
    best_v = 0.0
    
    # Target spacing estimate
    s0 = 1.0 / np.sqrt(N * np.sqrt(3.0) / 2.0)
    
    # Phase 1: Generate diverse initial configurations
    n_trials = 20
    for trial in range(n_trials):
        if time.time() > DEADLINE - 30.0:  # Leave time for polish
            break
        
        # Vary spacing and offset
        s = s0 * (0.98 + 0.04 * rng.random())
        ox = rng.uniform(-0.5, 0.5)
        oy = rng.uniform(-0.5, 0.5)
        cands = _hex_points(s, ox, oy)
        
        if cands.shape[0] < N:
            continue
        
        r_min = 0.5 * s * 0.9
        sel = _greedy_select(cands, N, r_min)
        if sel.shape[0] < N:
            continue
        
        v = _score(sel)
        if v > best_v:
            best_v = v
            best_c = sel.copy()
            
    # Phase 2: Force-directed optimization
    if best_c is not None:
        remaining = DEADLINE - time.time()
        if remaining > 15.0:
            r_target = best_v * 1.01
            fd_result = _force_directed_optimize(best_c, r_target, max_iters=2000, learning_rate=0.002)
            fd_v = _score(fd_result)
            if fd_v > best_v:
                best_v = fd_v
                best_c = fd_result.copy()
                
    # Phase 3: Gradient Polish
    if best_c is not None:
        remaining = DEADLINE - time.time()
        if remaining > 10.0:
            polished, pv = _gradient_polish(best_c, time_budget=min(remaining - 5.0, 60.0))
            if pv > best_v:
                best_c = polished
                best_v = pv
    else:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_v = _score(best_c)
        
    # Final validity clamp
    best_c = np.clip(best_c, -HALF, HALF)
    
    return best_c
