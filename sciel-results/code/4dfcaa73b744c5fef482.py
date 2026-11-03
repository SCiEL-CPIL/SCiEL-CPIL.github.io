"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

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
    # Boundary constraint: distance from each centre to the nearest side.
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0.0:
        return 0.0
    # Pairwise distances via squared distances.
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = math.sqrt(float(np.min(dist_sq)))
    return min(min_pair * 0.5, min_boundary)


def _generate_hex_candidates(spacing):
    """Dense hexagonal lattice of candidate centres inside [-0.5,0.5]^2."""
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
    distance to (already-selected points, boundary).
    """
    m = len(candidates)
    if m == 0:
        return np.zeros((0, 2)), 0.0
    if m <= n_select:
        sel = candidates.copy()
        return sel, _compute_radius(sel)

    boundary = np.min(0.5 - np.abs(candidates), axis=1)
    score = boundary.copy()
    sel_idx = []

    for _ in range(n_select):
        if _time_left() < 8.0:
            break
        best = int(np.argmax(score))
        sel_idx.append(best)
        p = candidates[best]
        diff = candidates - p
        d = np.sqrt(np.sum(diff * diff, axis=1))
        d[best] = np.inf
        score = np.minimum(score, d)

    sel = candidates[sel_idx]
    return sel, _compute_radius(sel)


def _force_directed_spread(pos, iterations=500, k=0.002):
    """
    Force-directed relaxation: points repel each other (inverse-square) and
    are attracted toward the square center to maintain density.
    Uses scipy's cKDTree for O(n log n) neighbor queries.
    """
    from scipy.spatial import cKDTree
    
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    if n < 2:
        return pos
    
    # Target spacing estimate
    target_r = _compute_radius(pos)
    if target_r <= 0:
        target_r = 0.036
    
    # Repulsion strength scales with current radius
    rep_strength = target_r * 2.0
    
    for it in range(iterations):
        if _time_left() < 8.0:
            break
        
        # Build KD-tree for efficient neighbor queries
        tree = cKDTree(pos)
        
        # Find all pairs within 2.5 * target_r (potential conflicts)
        pairs = tree.query_pairs(r=target_r * 2.5)
        
        forces = np.zeros_like(pos)
        
        for i, j in pairs:
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                # Coincident points: push apart randomly
                d = np.random.randn(2) * 1e-8
                dist = np.linalg.norm(d)
            d_norm = d / dist
            # Inverse-square repulsion
            f = rep_strength / (dist * dist)
            forces[i] += d_norm * f
            forces[j] -= d_norm * f
        
        # Gentle attraction toward center to prevent drift
        center_pull = -pos * 0.001
        
        # Update positions with damping
        pos += (forces + center_pull) * k
        
        # Clip to square interior
        pos = np.clip(pos, -0.4999, 0.4999)
    
    return pos


def _soft_min_optimize(pos, beta=200.0, lr=0.0005, max_iter=2000):
    """
    Gradient-based soft-min optimization. Minimizes a smooth approximation
    of the negative packing radius using the soft-min of constraints.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    
    def soft_min_neg_r(p, b):
        """
        Compute -r_soft where r_soft approximates min(min_pair/2, min_boundary).
        We minimize this to maximize r.
        """
        # Boundary terms: 0.5 - |p_ik| for each coordinate
        # We want these to be >= r, so penalty if small.
        # Pair terms: |p_i - p_j|/2, want >= r.
        
        # Use log-sum-exp style soft min over all constraints.
        # Constraints: for each i, boundary_i = min(0.5 - |p_ix|, 0.5 - |p_iy|)
        # For each pair i<j, pair_ij = |p_i - p_j|/2
        
        # Compute boundary distances
        bx = 0.5 - np.abs(p[:, 0])
        by = 0.5 - np.abs(p[:, 1])
        boundary = np.minimum(bx, by)  # shape (n,)
        
        # Compute pairwise distances / 2
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        pair_dists = np.sqrt(dist_sq) / 2.0  # shape (n, n)
        
        # Flatten all constraints
        # Boundary: n values
        # Pair: n*(n-1)/2 values, but we use upper triangle
        # For efficiency, we'll work with a combined vector
        
        # Soft min: -log(sum(exp(-c/beta))) * beta
        # But we need gradient, so let's do it carefully.
        
        # Combine boundary and pair constraints into one array
        # This is memory-heavy for n=200, but manageable.
        # Actually, let's use a different approach: 
        # minimize sum of exp(-c_i / beta) for all constraints c_i
        # This is equivalent to maximizing soft-min.
        
        # Boundary terms
        exp_b = np.exp(-boundary / beta)  # shape (n,)
        
        # Pair terms - only upper triangle to avoid double counting
        iu = np.triu_indices(n, k=1)
        pair_vals = pair_dists[iu]  # shape (n*(n-1)/2,)
        exp_p = np.exp(-pair_vals / beta)
        
        # Objective: sum of all exp terms (minimize this)
        obj = np.sum(exp_b) + np.sum(exp_p)
        
        # Gradient
        # d/dp sum(exp(-c/beta)) = sum(-1/beta * exp(-c/beta) * dc/dp)
        
        grad = np.zeros_like(p)
        
        # Boundary gradient
        # c_bx = 0.5 - |x|, dc/dx = -sign(x)
        # c_by = 0.5 - |y|, dc/dy = -sign(y)
        # boundary = min(bx, by), so gradient comes from the smaller one
        
        # For each point, determine which boundary is active
        active_x = bx < by  # if bx < by, boundary = bx
        
        # d(boundary)/dx for active_x: -sign(x)
        # d(boundary)/dy for active_y: -sign(y)
        
        exp_b_x = np.exp(-bx / beta)
        exp_b_y = np.exp(-by / beta)
        
        # Gradient contribution from boundary
        # For point i, if bx_i < by_i:
        #   d(obj)/dx_i = -1/beta * exp_b_x[i] * (-sign(x_i)) = sign(x_i)/beta * exp_b_x[i]
        #   d(obj)/dy_i = 0 (since by is not the min)
        # Wait, boundary = min(bx, by). If bx < by, boundary = bx, so d(boundary)/dx = -sign(x), d(boundary)/dy = 0.
        # But if we're doing soft-min over all constraints separately, we should treat bx and by as separate constraints.
        # Let me reconsider: the actual constraint is min over all of {bx_i, by_i for all i} and {pair_ij}.
        # So we should include bx_i and by_i as separate constraints.
        
        # Let me redo: constraints are:
        # For each i: bx_i = 0.5 - |x_i|, by_i = 0.5 - |y_i|
        # For each i<j: pair_ij = |p_i - p_j|/2
        
        # Objective = sum_{i} [exp(-bx_i/beta) + exp(-by_i/beta)] + sum_{i<j} exp(-pair_ij/beta)
        
        # Gradient:
        # d/dx_i exp(-bx_i/beta) = exp(-bx_i/beta) * (-1/beta) * d(bx_i)/dx_i
        # d(bx_i)/dx_i = -sign(x_i)
        # So d/dx_i = exp(-bx_i/beta) * sign(x_i) / beta
        
        grad[:, 0] += exp_b_x * np.sign(p[:, 0]) / beta
        grad[:, 1] += exp_b_y * np.sign(p[:, 1]) / beta
        
        # Pair gradient
        # d/dp_i exp(-pair_ij/beta) = exp(-pair_ij/beta) * (-1/beta) * d(pair_ij)/dp_i
        # pair_ij = |p_i - p_j|/2
        # d(pair_ij)/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
        
        # For each pair (i,j), contribution to grad[i] and grad[j]
        # This is O(n^2) per iteration, which is fine for n=200.
        
        # Vectorized: for each i, sum over j != i of exp(-pair_ij/beta) * (p_i - p_j) / (2 * |p_i - p_j|) / beta
        # Note: pair_ij = |p_i - p_j|/2, so 2*|p_i-p_j| = 4*pair_ij... wait.
        # |p_i - p_j| = 2 * pair_ij
        # d(pair_ij)/dp_i = (p_i - p_j) / (2 * |p_i - p_j|) = (p_i - p_j) / (4 * pair_ij)
        
        # Let's just compute it directly.
        diff_full = p[:, np.newaxis, :] - p[np.newaxis, :, :]  # (n, n, 2)
        dist_full = np.sqrt(np.sum(diff_full * diff_full, axis=2))  # (n, n)
        np.fill_diagonal(dist_full, np.inf)
        
        # weight_ij = exp(-pair_ij/beta) / beta, where pair_ij = dist_ij/2
        weight = np.exp(-dist_full / (2 * beta)) / beta  # (n, n)
        
        # grad[i] = sum_{j != i} weight[i,j] * (p_i - p_j) / (2 * dist_ij) * ... 
        # Wait, d(pair_ij)/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
        # So d(obj)/dp_i = sum_j weight[i,j] * (p_i - p_j) / (2 * dist_ij)
        # But weight already has 1/beta, and we have the derivative of pair which is (p_i-p_j)/(2*dist)
        
        # Actually: d(obj)/dp_i = sum_j exp(-pair_ij/beta) * (-1/beta) * d(pair_ij)/dp_i
        # = sum_j exp(-pair_ij/beta) * (-1/beta) * (p_i - p_j) / (2 * dist_ij)
        
        # So grad[i] = -sum_j [exp(-dist_ij/(2*beta)) / beta] * (p_i - p_j) / (2 * dist_ij)
        
        # Let's compute: factor[i,j] = -exp(-dist_ij/(2*beta)) / (2 * beta * dist_ij)
        # Then grad[i] = sum_j factor[i,j] * (p_i - p_j)
        
        factor = weight / (2 * dist_full)  # (n, n, ) but need to handle inf
        factor = np.where(dist_full > 1e-12, factor, 0.0)
        factor = -factor  # negative sign from derivative
        
        # grad[i] = sum_j factor[i,j] * (p_i - p_j)
        # = sum_j factor[i,j] * diff_full[i,j]
        
        grad_pair = np.sum(factor[:, :, np.newaxis] * diff_full, axis=1)  # (n, 2)
        grad += grad_pair
        
        return obj, grad
    
    # Simple gradient descent with adaptive learning rate
    for it in range(max_iter):
        if _time_left() < 8.0:
            break
        
        obj, grad = soft_min_neg_r(pos, beta)
        
        # Normalize gradient to prevent explosion
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm * min(gnorm, 0.01)
        
        pos -= lr * grad
        pos = np.clip(pos, -0.4999, 0.4999)
        
        # Adaptive beta: increase over time for sharper approximation
        beta = min(beta * 1.01, 500.0)
    
    return pos


def _local_refine(pos):
    """
    Combined local refinement: force-directed spread followed by soft-min optimization.
    """
    pos = np.array(pos, dtype=float, copy=True)
    
    # Phase 1: Force-directed spread to resolve obvious overlaps
    if _time_left() > 15.0:
        pos = _force_directed_spread(pos, iterations=300, k=0.001)
    
    # Phase 2: Soft-min gradient optimization for precise refinement
    if _time_left() > 10.0:
        pos = _soft_min_optimize(pos, beta=100.0, lr=0.0003, max_iter=1000)
    
    # Phase 3: Another round of force-directed to clean up
    if _time_left() > 8.0:
        pos = _force_directed_spread(pos, iterations=200, k=0.0005)
    
    return pos


def construct():
    best_pos = None
    best_r = 0.0

    # 1) Greedy selection from dense hexagonal lattices at multiple spacings.
    spacings = [0.035, 0.03, 0.025, 0.02]
    for sp in spacings:
        if _time_left() < 20.0:
            break
        try:
            candidates = _generate_hex_candidates(sp)
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
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_r = _compute_radius(best_pos)

    # 2) Local refinement: force-directed + soft-min optimization.
    if _time_left() > 12.0:
        refined = _local_refine(best_pos)
        r = _compute_radius(refined)
        if r > best_r:
            best_r = r
            best_pos = refined

    # 3) Try a second refinement pass with different parameters if time allows.
    if _time_left() > 15.0:
        # Perturb slightly and refine again
        rng = np.random.RandomState(42)
        perturbed = best_pos + rng.randn(N, 2) * 0.002
        perturbed = np.clip(perturbed, -0.4999, 0.4999)
        refined2 = _local_refine(perturbed)
        r2 = _compute_radius(refined2)
        if r2 > best_r:
            best_r = r2
            best_pos = refined2

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
