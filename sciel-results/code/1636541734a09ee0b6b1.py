"""
Packing 50 equal circles in the unit square [-0.5, 0.5]^2.
Structural change: Direct force-based / repulsion optimization with
simulated annealing on a continuous energy landscape, plus targeted
pairwise refinement. This avoids the local minima trap of pure
geometric perturbation by using a smooth objective that guides
optimization toward the global packing.
"""

import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50

def _score(pts):
    """Return the achievable common radius for a centre array (N,2)."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    d_bound = np.min(0.5 - np.abs(pts), axis=1)
    r = float(np.min(d_bound))
    if r < 0:
        return 0.0
    if N > 1:
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        dmin = float(np.min(d[:, 1]))
        r = min(r, dmin / 2.0)
    return max(r, 0.0)

def _valid(pts):
    return (pts.shape == (N, 2)
            and np.all(np.isfinite(pts))
            and np.all(pts >= -0.5 - 1e-9)
            and np.all(pts <= 0.5 + 1e-9))

def _energy(pts, target_r):
    """
    Energy function: penalizes violations of the desired radius.
    Lower energy = better packing at target_r.
    """
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 1e10
    
    # Boundary violations
    d_bound = 0.5 - np.abs(pts)
    bound_viol = np.maximum(0, target_r - d_bound.min(axis=1))
    
    # Pairwise violations
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    pair_dists = d[:, 1]
    pair_viol = np.maximum(0, 2 * target_r - pair_dists)
    
    return float(np.sum(bound_viol**2) + np.sum(pair_viol**2))

def _hex_initial():
    """Hexagonal (triangular) lattice initialization."""
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
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = np.vstack([pts, grid[len(pts):]])
    else:
        d2 = np.sum(pts * pts, axis=1)
        order = np.argsort(d2)
        pts = pts[order[:N]]
    return pts.copy()

def _grid_initial():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _force_based_refine(pts, target_r, rng, time_budget, lr=0.005, decay=0.999):
    """
    Force-based optimization: move points away from violations.
    This is a gradient-free method that uses the structure of the
    constraint violations to guide movement.
    """
    curr = pts.copy()
    best = pts.copy()
    best_r = _score(best)
    curr_r = _score(curr)
    
    t0 = time.time()
    step = lr
    
    while time.time() - t0 < time_budget:
        if step < 1e-8:
            break
            
        # Compute forces
        tree = cKDTree(curr)
        d, idx = tree.query(curr, k=2)
        pair_dists = d[:, 1]
        pair_idxs = idx[:, 1]
        
        forces = np.zeros_like(curr)
        
        # Pairwise repulsion for violated pairs
        for i in range(N):
            j = pair_idxs[i]
            dist = pair_dists[i]
            if dist < 2 * target_r:
                # Push apart
                diff = curr[i] - curr[j]
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    force = diff / norm * (2 * target_r - dist) * 0.5
                    forces[i] += force
                    forces[j] -= force
        
        # Boundary repulsion
        for i in range(N):
            for k in range(2):
                if curr[i, k] + target_r > 0.5:
                    forces[i, k] -= (curr[i, k] + target_r - 0.5)
                if curr[i, k] - target_r < -0.5:
                    forces[i, k] += (target_r - 0.5 - curr[i, k])
        
        # Add some randomness to escape local minima
        noise = rng.normal(0, step * 0.5, size=(N, 2))
        
        # Update positions
        curr = curr + step * forces + noise
        curr = np.clip(curr, -0.4999999, 0.4999999)
        
        curr_r = _score(curr)
        if curr_r > best_r:
            best = curr.copy()
            best_r = curr_r
        
        # Anneal the step size
        step *= decay
    
    return best, best_r

def _pairwise_refine(pts, rng, time_budget):
    """
    Targeted refinement: identify the tightest pairs and boundaries,
    and specifically optimize those interactions.
    """
    curr = pts.copy()
    best = pts.copy()
    best_r = _score(best)
    
    t0 = time.time()
    base_r = best_r
    temp = 0.002
    
    while time.time() - t0 < time_budget:
        if temp < 1e-7:
            break
        
        # Find the most constrained pairs and boundaries
        tree = cKDTree(curr)
        d, idx = tree.query(curr, k=2)
        pair_dists = d[:, 1]
        pair_idxs = idx[:, 1]
        
        d_bound = np.min(0.5 - np.abs(curr), axis=1)
        
        # Identify critical constraints
        min_pair_dist = np.min(pair_dists)
        min_bound = np.min(d_bound)
        
        # Select points involved in the tightest constraints
        critical_pairs = np.where(pair_dists < min_pair_dist + 0.005)[0]
        critical_bounds = np.where(d_bound < min_bound + 0.005)[0]
        
        critical_idx = np.unique(np.concatenate([critical_pairs, critical_bounds]))
        if len(critical_idx) < 2:
            critical_idx = np.random.choice(N, size=3, replace=False)
        
        # Perturb only critical points
        n_move = min(len(critical_idx), rng.integers(2, 6))
        move_idx = critical_idx[:n_move]
        
        cand = curr.copy()
        cand[move_idx] += rng.normal(0, temp, size=(n_move, 2))
        cand = np.clip(cand, -0.4999999, 0.4999999)
        
        cand_r = _score(cand)
        
        if cand_r >= curr_r - temp * 0.1:
            # Accept with probability
            if cand_r >= curr_r or rng.random() < np.exp((cand_r - curr_r) / temp):
                curr = cand
                curr_r = cand_r
                if curr_r > best_r:
                    best = curr.copy()
                    best_r = curr_r
        
        temp *= 0.9995
    
    return best, best_r

def _polish(pts, deadline):
    """Gradient-free polish using Nelder-Mead."""
    best_points = pts.copy()
    best_r = _score(best_points)
    x0 = best_points.flatten()
    
    def _neg_radius(params):
        points = params.reshape(-1, 2)
        if np.any(np.abs(points) > 0.5 + 1e-8):
            return 1e6
        r = _score(points)
        if r <= 0:
            return 1e6
        return -r
    
    try:
        result = minimize(
            _neg_radius, x0, method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
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
    deadline = start + 145.0  # Leave 5s safety margin for 150s limit

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

    # Phase 1: Force-based refinement from multiple initializations
    time_left = deadline - start - 20.0
    n_phases = 3
    
    for phase in range(n_phases):
        if time.time() > deadline - 25.0:
            break
        
        t_alloc = time_left / (n_phases - phase)
        if t_alloc < 3.0:
            continue
        
        # Different initialization for each phase
        if phase == 0:
            cur = _hex_initial()
        elif phase == 1:
            cur = _hex_initial() + rng.normal(0.0, 0.03, size=(N, 2))
            cur = np.clip(cur, -0.4999999, 0.4999999)
        else:
            cur = _grid_initial() + rng.normal(0.0, 0.04, size=(N, 2))
            cur = np.clip(cur, -0.4999999, 0.4999999)
        
        if not _valid(cur):
            cur = best_pts.copy()
        
        # Force-based refinement
        target_r = best_s * 1.01  # Try to improve slightly
        cur, s = _force_based_refine(cur, target_r, rng, t_alloc * 0.6)
        
        # Pairwise refinement
        cur, s = _pairwise_refine(cur, rng, t_alloc * 0.4)
        
        if _valid(cur) and s > best_s:
            best_s = s
            best_pts = cur.copy()
        
        time_left = deadline - start - time.time() - 20.0

    # Phase 2: Final polish
    if time.time() < deadline - 5.0:
        best_pts, best_s = _polish(best_pts, deadline)

    # Final safety
    best_pts = np.clip(best_pts, -0.499999999999, 0.499999999999)
    if not _valid(best_pts):
        best_pts = _grid_initial()

    return best_pts

# EVOLVE-BLOCK-START
# construct() is the entry point
# EVOLVE-BLOCK-END
