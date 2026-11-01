"""Improved packer: Soft-min L-BFGS-B optimization."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300
SEED = 12345
DEADLINE = 115.0  # safety margin before the 120 s wall-clock limit


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    r_pair = 0.5 * dists[:, 1].min()
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _ring_seed(n_rings=15):
    """Generate a ring-based initialization."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    for ring in range(1, n_rings + 1):
        r = ring * 0.06
        n = int(2 * np.pi * r / 0.06)
        if n < 1:
            n = 1
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    pts = np.array(pts[:N])
    # Project into unit circle
    rad = np.linalg.norm(pts, axis=1)
    mask = rad > 0.99
    if np.any(mask):
        pts[mask] = pts[mask] / rad[mask, None] * 0.99
    return pts


def _soft_min_obj_and_grad(flat_pts, beta):
    """
    Compute the soft-min objective and gradient.
    Objective: -soft_min(distances)
    We want to maximize the minimum distance, so we minimize the negative soft-min.
    """
    pts = flat_pts.reshape(N, 2)
    
    # Compute distances to boundary
    norms = np.linalg.norm(pts, axis=1)
    d_bound = 1.0 - norms
    
    # Compute pairwise distances
    # To avoid O(N^2) memory blowup for large N, we can use KDTree, 
    # but for N=300, O(N^2) is 90k ops, which is fine.
    # However, we need gradients.
    # Let's compute all pairwise distances.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise constraint is dist/2 >= r, so effective distance is dists/2
    d_pair = dists / 2.0
    
    # Combine boundary and pairwise distances into a single array of constraints
    # We have N boundary distances and N*(N-1)/2 pairwise distances
    # Actually, for soft-min, we can just sum over all valid pairs.
    # Let's flatten the pairwise distances (upper triangle)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_pairs_flat = d_pair[i_idx, j_idx]
    
    # All constraints: d_bound (N) and d_pairs_flat (N*(N-1)/2)
    all_d = np.concatenate([d_bound, d_pairs_flat])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow, subtract max
    max_d = np.max(all_d)
    shifted = -beta * (all_d - max_d)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + beta * max_d
    obj = -log_sum_exp / beta
    
    # Gradient
    # d/dx_i obj = -1/beta * d/dx_i log(sum_j exp(-beta d_j))
    # = -1/beta * (1/sum_j exp(-beta d_j)) * sum_j exp(-beta d_j) * (-beta) * d d_j/dx_i
    # = sum_j P_j * d d_j/dx_i
    # where P_j = exp(-beta d_j) / sum_k exp(-beta d_k)
    
    probs = np.exp(shifted)
    probs /= np.sum(probs)
    
    grad_pts = np.zeros_like(pts)
    
    # Gradient from boundary terms
    # d_bound_i = 1 - ||x_i||
    # d d_bound_i / d x_i = -x_i / ||x_i||
    # Contribution to grad_obj: sum_{i} P_bound_i * (-x_i / ||x_i||)
    # Note: obj is negative soft min, so we are minimizing -soft_min.
    # Wait, let's re-derive.
    # We want to maximize r = min(d_i).
    # Let g(x) = soft_min(d_i) = -1/beta log(sum exp(-beta d_i)).
    # We want to maximize g(x), so we minimize -g(x).
    # grad(-g(x)) = - grad(g(x)).
    # grad(g(x)) = sum P_i grad(d_i).
    # So grad(-g(x)) = - sum P_i grad(d_i).
    
    # Boundary part
    P_bound = probs[:N]
    # grad(d_bound_i) = -pts[i] / norms[i]
    # We add -P_bound_i * (-pts[i] / norms[i]) = P_bound_i * pts[i] / norms[i] to grad_obj
    # Wait, if we minimize -g, the gradient is -grad(g).
    # grad(g) gets contribution from boundary: P_bound_i * (-pts[i]/norms[i]).
    # So grad(-g) gets contribution: P_bound_i * (pts[i]/norms[i]).
    
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_bound = P_bound[:, np.newaxis] * (pts / safe_norms[:, np.newaxis])
    grad_pts += grad_bound
    
    # Pairwise part
    P_pair = probs[N:]
    # d_pair_ij = ||x_i - x_j|| / 2
    # grad_{x_i} d_pair_ij = (x_i - x_j) / (2 * ||x_i - x_j||)
    # grad_{x_j} d_pair_ij = (x_j - x_i) / (2 * ||x_i - x_j||)
    
    # We have flattened pairs. We need to scatter the gradients back.
    # For each pair (i, j) with probability P_k:
    # grad_pts[i] += P_k * (x_i - x_j) / (2 * dist_ij)
    # grad_pts[j] += P_k * (x_j - x_i) / (2 * dist_ij)
    
    # Efficient way:
    # diffs = pts[i] - pts[j]
    # dists = ||diffs||
    # unit_vecs = diffs / dists
    # contrib_i = P_k * unit_vecs / 2
    # contrib_j = -P_k * unit_vecs / 2
    
    diff_vecs = pts[i_idx] - pts[j_idx]
    dist_vecs = d_pairs_flat[:, np.newaxis] # shape (M, 1)
    # Avoid division by zero
    safe_dist = np.where(dist_vecs > 1e-10, dist_vecs, 1.0)
    unit_vecs = diff_vecs / safe_dist
    
    # Gradient contribution to x_i: + P * unit_vec / 2
    # Gradient contribution to x_j: - P * unit_vec / 2
    
    # Use np.add.at for accumulation
    np.add.at(grad_pts, i_idx, P_pair[:, np.newaxis] * unit_vecs * 0.5)
    np.add.at(grad_pts, j_idx, -P_pair[:, np.newaxis] * unit_vecs * 0.5)
    
    # Final gradient for the objective function to be minimized
    grad_obj = -grad_pts.flatten()
    
    return obj, grad_obj


def _soft_min_optimize(points, t_budget, rng):
    """
    Optimize using L-BFGS-B on the soft-min objective.
    """
    start = time.time()
    pts = points.copy()
    best_r = _score(pts)
    best_pts = pts.copy()
    
    # Try a few different beta values or just one good one?
    # A high beta makes the function sharper (closer to true min) but harder to optimize.
    # A low beta is smoother but might converge to a worse local min.
    # Let's try beta=50.
    
    betas = [20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() - start > t_budget - 2:
            break
            
        x0 = best_pts.flatten()
        
        # L-BFGS-B is a good choice for this
        res = minimize(
            _soft_min_obj_and_grad,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 100,
                'ftol': 1e-8,
                'gtol': 1e-5,
                'maxls': 20,
                'maxfun': 200
            }
        )
        
        new_pts = res.x.reshape(N, 2)
        
        # Check validity and score
        if np.all(np.isfinite(new_pts)):
            new_r = _score(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                
    return best_pts, best_r


def _local_search(points, t_budget, rng):
    """Fast local search (hill climbing) using KD-tree for distance checks."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    
    cur_min_dist, _ = _min_dist_and_pair(cur)
    cur_r_bound = 1.0 - np.linalg.norm(cur, axis=1)
    cur_r = min(cur_r_bound.min(), 0.5 * cur_min_dist)
    
    if cur_r > best_r:
        best_r = cur_r
        best = cur.copy()

    step = 0.01
    tree = cKDTree(cur)
    
    while time.time() - start < t_budget:
        idx = rng.integers(N)
        ang = rng.random() * 2.0 * np.pi
        mag = step * (0.5 + rng.random())
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)
        
        old_pt = cur[idx].copy()
        new_pt = old_pt + np.array([dx, dy])
        
        new_norm = np.hypot(new_pt[0], new_pt[1])
        if new_norm > 1.0:
            continue
            
        dists, indices = tree.query(new_pt, k=2)
        nn_idx = indices[0]
        nn_dist = dists[0]
        
        if nn_idx == idx:
            nn_idx = indices[1]
            nn_dist = dists[1]

        new_r_pair = 0.5 * nn_dist
        new_r_bound = 1.0 - new_norm
        new_local_r = min(new_r_pair, new_r_bound)
        
        # Simple acceptance: accept if improves local constraint
        if new_local_r >= cur_r - 1e-9:
            cur[idx] = new_pt
            tree = cKDTree(cur)  # Rebuild tree (simpler, still fast for N=300)
            cur_min_dist, _ = _min_dist_and_pair(cur)
            cur_r_bound = 1.0 - np.linalg.norm(cur, axis=1)
            cur_r = min(cur_r_bound.min(), 0.5 * cur_min_dist)
            
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
                step *= 0.99
            else:
                cur[idx] = old_pt
                step *= 1.01
                if step > 0.1:
                    step = 0.1
                if step < 0.001:
                    step = 0.001
        else:
            step *= 1.01
            if step > 0.1:
                step = 0.1
            if step < 0.001:
                step = 0.001

    return best, best_r


def _min_dist_and_pair(points):
    """Compute minimum pairwise distance."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(dists[:, 1].min()), (0, 0)


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Multiple initializations
    init_strategies = [
        _lattice_seed(2.0 / (np.sqrt(N) + 1.0)),
        _ring_seed(),
    ]
    
    # Add a random initialization
    random_pts = rng.standard_normal((N, 2)) * 0.7
    rad = np.linalg.norm(random_pts, axis=1)
    mask = rad > 0.95
    if np.any(mask):
        random_pts[mask] = random_pts[mask] / rad[mask, None] * 0.95
    init_strategies.append(random_pts)
    
    best = init_strategies[0].copy()
    best_r = _score(best)
    
    # First pass: Soft-min optimization on each initialization
    for init_pts in init_strategies:
        if time.time() - t0 > DEADLINE - 10:
            break
        t_left = DEADLINE - (time.time() - t0)
        if t_left < 15:
            break
        t_alloc = min(t_left - 5, 30.0)  # Give more time to L-BFGS
        
        cand, r = _soft_min_optimize(init_pts, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand
    
    # Second pass: local search polish on best
    if time.time() - t0 < DEADLINE - 5:
        t_left = DEADLINE - (time.time() - t0)
        if t_left > 5:
            t_alloc = min(t_left - 2, 20.0)
            cand, r = _local_search(best, t_alloc, rng)
            if r > best_r:
                best_r = r
                best = cand
    
    # Final safety
    final_r = _score(best)
    if final_r <= 0.0:
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
