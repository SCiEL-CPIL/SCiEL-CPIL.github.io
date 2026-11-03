"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# Global deadline guard: evaluator kills at 150s; keep a safety margin.
_T0 = time.time()
_DEADLINE = _T0 + 138.0  # 12s safety margin before the 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _compute_radius(pos):
    """Vectorized packing radius: min(min_pair/2, min_boundary)."""
    n = len(pos)
    if n == 0:
        return 0.0
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0.0:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = math.sqrt(float(np.min(dist_sq)))
    return min(min_pair * 0.5, min_boundary)


def _generate_hex_grid(spacing, n_target):
    """Generate a hexagonal lattice and select n_target points via greedy."""
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
    if not points:
        return np.zeros((0, 2))
    candidates = np.array(points, dtype=float)
    m = len(candidates)
    if m <= n_target:
        return candidates
    # Greedy selection
    boundary = np.min(0.5 - np.abs(candidates), axis=1)
    score = boundary.copy()
    sel_idx = []
    for _ in range(n_target):
        if _time_left() < 10.0:
            break
        best = int(np.argmax(score))
        sel_idx.append(best)
        p = candidates[best]
        diff = candidates - p
        d = np.sqrt(np.sum(diff * diff, axis=1))
        d[best] = np.inf
        score = np.minimum(score, d)
    return candidates[sel_idx]


def _vectorized_force_spread(pos, iterations=500, strength=0.001):
    """Vectorized force-directed: repulsion for close pairs + centering."""
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    if n < 2:
        return pos
    for it in range(iterations):
        if _time_left() < 8.0:
            break
        # Full pairwise (n=200 -> 200x200 is fine)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n,n,2)
        dist_sq = np.sum(diff * diff, axis=2)  # (n,n)
        np.fill_diagonal(dist_sq, 1e30)
        dist = np.sqrt(dist_sq)
        # Repulsion: f = strength / dist^2 in direction diff/dist = strength * diff / dist^3
        # Only for pairs closer than 3*r_target to save computation conceptually
        # but vectorized it's just one pass
        inv_dist3 = np.zeros_like(dist_sq)
        mask = dist_sq > 1e-20
        inv_dist3[mask] = 1.0 / (dist_sq[mask] * np.sqrt(dist_sq[mask]))
        # Force on i from j: strength * diff[i,j] / dist^3
        forces = np.sum(strength * diff * inv_dist3[:, :, np.newaxis], axis=1)
        # Centering: pull toward origin
        center_pull = -pos * 0.0005
        pos += forces + center_pull
        pos = np.clip(pos, -0.4999, 0.4999)
    return pos


def _soft_min_objective_and_grad(p_flat, beta):
    """
    Objective: sum of exp(-c_i / beta) over all constraints c_i.
    Constraints: boundary distances (bx_i, by_i for each point) and pair distances/2.
    Minimizing this maximizes the soft-min of the constraints.
    Returns (objective, gradient) as flat arrays.
    """
    p = p_flat.reshape(-1, 2)
    n = len(p)
    
    # Boundary constraints
    bx = 0.5 - np.abs(p[:, 0])
    by = 0.5 - np.abs(p[:, 1])
    
    # Pair constraints: |p_i - p_j| / 2
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Objective
    exp_bx = np.exp(-bx / beta)
    exp_by = np.exp(-by / beta)
    exp_pair = np.exp(-dist / (2.0 * beta))
    # Only upper triangle for pairs to avoid double counting
    iu = np.triu_indices(n, k=1)
    obj = np.sum(exp_bx) + np.sum(exp_by) + np.sum(exp_pair[iu])
    
    # Gradient
    grad = np.zeros_like(p)
    
    # Boundary: d/dx_i exp(-bx_i/beta) = exp(-bx_i/beta) * sign(x_i) / beta
    grad[:, 0] += exp_bx * np.sign(p[:, 0]) / beta
    grad[:, 1] += exp_by * np.sign(p[:, 1]) / beta
    
    # Pairs: d/dp_i exp(-dist_ij/(2*beta)) = exp(-dist_ij/(2*beta)) * (-1/(2*beta)) * (p_i - p_j)/dist_ij
    # = -exp(-dist_ij/(2*beta)) * (p_i - p_j) / (2 * beta * dist_ij)
    # For each i: sum over j != i of the above
    inv_dist = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-20
    inv_dist[mask] = 1.0 / dist[mask]
    # weight[i,j] = -exp(-dist_ij/(2*beta)) / (2*beta) * inv_dist[i,j]
    weight = -exp_pair / (2.0 * beta) * inv_dist
    np.fill_diagonal(weight, 0.0)
    # grad[i] = sum_j weight[i,j] * (p_i - p_j) = sum_j weight[i,j] * diff[i,j]
    grad += np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    
    return obj, grad.ravel()


def _lbfgsb_refine(pos, beta, max_iter=2000):
    """L-BFGS-B optimization of the soft-min objective."""
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    
    # Time budget for this call
    t_start = time.time()
    time_budget = min(_time_left() - 10.0, 80.0)
    if time_budget < 5.0:
        return pos
    
    def fun(p_flat):
        obj, _ = _soft_min_objective_and_grad(p_flat, beta)
        return obj
    
    def jac(p_flat):
        _, g = _soft_min_objective_and_grad(p_flat, beta)
        return g
    
    # Bounds: each coordinate in [-0.49999, 0.49999]
    bounds = [(-0.49999, 0.49999)] * (2 * n)
    
    try:
        result = minimize(
            fun, pos.ravel(), jac=jac, method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'maxls': 50,
                'gtol': 1e-8,
                'ftol': 1e-12,
                'disp': False,
            }
        )
        pos = result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return pos


def _local_refine(pos):
    """Multi-phase refinement: force spread -> L-BFGS-B with increasing beta."""
    pos = np.array(pos, dtype=float, copy=True)
    
    # Phase 1: Vectorized force-directed to resolve overlaps
    if _time_left() > 20.0:
        pos = _vectorized_force_spread(pos, iterations=200, strength=0.0005)
    
    # Phase 2: L-BFGS-B with moderate beta
    if _time_left() > 25.0:
        pos = _lbfgsb_refine(pos, beta=50.0, max_iter=1500)
    
    # Phase 3: L-BFGS-B with higher beta for sharper approximation
    if _time_left() > 20.0:
        pos = _lbfgsb_refine(pos, beta=200.0, max_iter=1500)
    
    # Phase 4: Final polish with very high beta
    if _time_left() > 15.0:
        pos = _lbfgsb_refine(pos, beta=1000.0, max_iter=1000)
    
    # Phase 5: Tiny force-directed cleanup
    if _time_left() > 10.0:
        pos = _vectorized_force_spread(pos, iterations=100, strength=0.0002)
    
    return pos


def construct():
    best_pos = None
    best_r = 0.0

    # 1) Generate initial hex grid at target spacing (~2*r_best = 0.0732)
    # Try a few spacings around the expected optimal
    spacings = [0.0732, 0.072, 0.074, 0.070, 0.075]
    for sp in spacings:
        if _time_left() < 30.0:
            break
        try:
            grid = _generate_hex_grid(sp, N)
            if len(grid) >= N:
                r = _compute_radius(grid)
                if r > best_r:
                    best_r = r
                    best_pos = grid.copy()
        except Exception:
            continue

    # Fallback: uniform grid
    if best_pos is None or best_r < 0.02:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        r = _compute_radius(grid)
        if r > best_r:
            best_r = r
            best_pos = grid.copy()

    if best_pos is None:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_r = _compute_radius(best_pos)

    # 2) Main refinement pass
    if _time_left() > 25.0:
        refined = _local_refine(best_pos)
        r = _compute_radius(refined)
        if r > best_r:
            best_r = r
            best_pos = refined

    # 3) Second pass: perturb and refine
    if _time_left() > 30.0:
        rng = np.random.RandomState(42)
        perturbed = best_pos + rng.randn(N, 2) * 0.003
        perturbed = np.clip(perturbed, -0.4999, 0.4999)
        refined2 = _local_refine(perturbed)
        r2 = _compute_radius(refined2)
        if r2 > best_r:
            best_r = r2
            best_pos = refined2

    # 4) Third pass: different perturbation seed
    if _time_left() > 25.0:
        rng = np.random.RandomState(123)
        perturbed = best_pos + rng.randn(N, 2) * 0.005
        perturbed = np.clip(perturbed, -0.4999, 0.4999)
        refined3 = _local_refine(perturbed)
        r3 = _compute_radius(refined3)
        if r3 > best_r:
            best_r = r3
            best_pos = refined3

    # Final safety: guarantee validity.
    best_pos = np.asarray(best_pos, dtype=float)
    if best_pos.shape != (N, 2):
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
