"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Global deadline guard: evaluator kills at 150s; keep a safety margin.
_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 10s safety margin before the 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _compute_radius(pos):
    """Vectorized packing radius: min(min_pair/2, min_boundary)."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Boundary constraint: distance from each centre to the nearest side.
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0.0:
        return 0.0
    # Pairwise distances via squared distances (cheaper, no sqrt on the full matrix).
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = math.sqrt(float(np.min(dist_sq)))
    return min(min_pair * 0.5, min_boundary)


def _generate_hex_grid(spacing):
    """Generate a hexagonal grid of points inside [-0.5, 0.5]^2."""
    dy = spacing * math.sqrt(3.0) / 2.0
    max_coord = 0.5
    points = []
    row = 0
    y = -max_coord
    while y <= max_coord:
        x_offset = 0.0 if (row % 2 == 0) else spacing / 2.0
        x = -max_coord + x_offset
        while x <= max_coord:
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                points.append((x, y))
            x += spacing
        y += dy
        row += 1
    return np.array(points, dtype=float) if points else np.zeros((0, 2))


def _greedy_select(candidates, n_select):
    """
    Greedy best-fit: repeatedly pick the candidate that maximises the minimum
    distance to (already-selected points, boundary). Returns (selected, score).
    """
    m = len(candidates)
    if m == 0:
        return np.zeros((0, 2)), 0.0
    if m <= n_select:
        sel = candidates.copy()
        return sel, _compute_radius(sel)

    # Per-candidate boundary distance (distance to nearest side).
    boundary = np.min(0.5 - np.abs(candidates), axis=1)
    # score[i] = min(boundary_i, min distance to selected so far).
    score = boundary.copy()
    sel_idx = []

    for _ in range(n_select):
        if _time_left() < 5.0:
            break
        best = int(np.argmax(score))
        sel_idx.append(best)
        p = candidates[best]
        # Distance from every candidate to the newly selected point.
        diff = candidates - p
        d = np.sqrt(np.sum(diff * diff, axis=1))
        d[best] = np.inf
        # New score is the tighter of old score and distance to new point.
        score = np.minimum(score, d)

    sel = candidates[sel_idx]
    return sel, _compute_radius(sel)


def _slsqp_objective(x, r):
    """
    Objective for SLSQP: minimize negative radius (i.e., maximize r).
    x is a flat array of length 2*N containing positions.
    r is the current radius estimate.
    """
    return -r


def _slsqp_constraints(pos):
    """
    Generate constraint functions for SLSQP.
    Each constraint is of the form: g(x) >= 0
    """
    n = len(pos)
    constraints = []
    
    # Boundary constraints: each point must be at least r from each side
    # x_i >= -0.5 + r, x_i <= 0.5 - r, y_i >= -0.5 + r, y_i <= 0.5 - r
    # But since r is fixed during optimization, we use:
    # 0.5 - |x_i| >= r and 0.5 - |y_i| >= r
    # This is non-differentiable, so we use:
    # x_i + r <= 0.5 and -x_i + r <= 0.5
    # i.e., 0.5 - x_i - r >= 0 and 0.5 + x_i - r >= 0
    
    # Pairwise constraints: |p_i - p_j| >= 2r
    # This is non-differentiable, so we use squared distance:
    # |p_i - p_j|^2 >= (2r)^2
    # i.e., |p_i - p_j|^2 - 4r^2 >= 0
    
    # We'll generate constraints in batches to avoid too many constraints
    # For now, let's just return a placeholder and handle it differently
    
    return constraints


def _run_slsqp(pos_init, r_init, time_budget):
    """
    Run SLSQP optimization to maximize the radius.
    """
    n = len(pos_init)
    pos_init = np.array(pos_init, dtype=float).copy()
    
    # Flatten positions
    x0 = pos_init.flatten()
    
    # Objective: minimize -r (we'll use a surrogate)
    # Actually, let's use a different approach:
    # Maximize the minimum of all pairwise distances/2 and boundary distances
    # This is equivalent to maximizing r
    
    # We'll use a smooth approximation: minimize the sum of squared "violations"
    # where violation_i = max(0, 2r - d_i) for pairs, and max(0, r - b_i) for boundaries
    
    def objective(x):
        pos = x.reshape(-1, 2)
        r = _compute_radius(pos)
        return -r
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    # We'll use SLSQP with a reasonable max iterations
    result = None
    try:
        start_time = time.time()
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        elapsed = time.time() - start_time
        if result.success and result.fun < -r_init:
            best_pos = result.x.reshape(-1, 2)
            best_r = -result.fun
            return best_pos, best_r, elapsed
    except Exception:
        pass
    
    return pos_init, r_init, 0.0


def _local_refine(pos, start_budget):
    """
    Iterative local refinement using SLSQP.
    """
    pos = np.array(pos, dtype=float, copy=True)
    cur_r = _compute_radius(pos)
    
    # Run SLSQP optimization
    if _time_left() > 15.0:
        refined, r, elapsed = _run_slsqp(pos, cur_r, _time_left() - 15.0)
        if r > cur_r:
            pos = refined
            cur_r = r
    
    return pos, cur_r


def construct():
    best_pos = None
    best_r = 0.0

    # 1) Greedy selection from a dense hexagonal lattice.
    #    Optimal radius ~0.0366 -> spacing ~2r ~0.073; use finer for resolution.
    spacings = [0.04, 0.03, 0.025]
    for sp in spacings:
        if _time_left() < 15.0:
            break
        try:
            candidates = _generate_hex_grid(sp)
            if len(candidates) < N:
                continue
            sel, r = _greedy_select(candidates, N)
            if len(sel) >= N and r > best_r:
                best_r = r
                best_pos = sel.copy()
        except Exception:
            continue

    # Fallback: coarse uniform grid if greedy produced nothing usable.
    if best_pos is None or best_r < 0.02:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        r = _compute_radius(grid)
        if r > best_r:
            best_r = r
            best_pos = grid.copy()

    if best_pos is None:
        # Absolute safe fallback.
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_r = _compute_radius(best_pos)

    # 2) Local refinement to polish the greedy solution.
    if _time_left() > 10.0:
        refined, r = _local_refine(best_pos, _time_left())
        if r > best_r:
            best_r = r
            best_pos = refined

    # Final safety: guarantee validity (finite, inside square, correct shape).
    best_pos = np.asarray(best_pos, dtype=float)
    if best_pos.shape != (N, 2):
        # Repair shape by padding/truncating with a valid grid.
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_pos = grid
    best_pos = np.clip(best_pos, -0.49999, 0.49999)
    if not np.all(np.isfinite(best_pos)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]

    return best_pos
# EVOLVE-BLOCK-END
