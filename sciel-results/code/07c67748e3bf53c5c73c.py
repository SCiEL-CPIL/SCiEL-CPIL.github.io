"""
Circle packing: maximize radius of 50 equal circles in unit disk.
Combined approach:
1. Direct max-min formulation using differentiable soft-min (log-sum-exp)
2. Vectorized gradient computation for speed
3. Multi-stage beta annealing with L-BFGS-B
4. Force-directed relaxation as pre-conditioner
5. Multiple initializations (hex lattice, ring, random)
6. High-precision polish passes with tight tolerances
7. Explicit deadline guard to stay within time limit
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin >= 1.0):
        return 0.0
    
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise/2 and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation with numerical stability
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat


def _force_directed_relax(pts, n_steps=300, target_r=0.12):
    """
    Force-directed relaxation to improve packing.
    Pushes overlapping circles apart and pulls them toward the center.
    """
    pts = pts.copy()
    target_d = 2.0 * target_r
    
    for step in range(n_steps):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        # Only consider pairs that are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                if d < target_d and d > 1e-10:
                    overlap = target_d - d
                    direction = (pts[i] - pts[j]) / d
                    force_mag = overlap * 0.5
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary constraint: push circles inward if they're outside
        dists_origin = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if dists_origin[i] > 1.0 - target_r:
                direction = -pts[i] / dists_origin[i]
                excess = dists_origin[i] - (1.0 - target_r)
                forces[i] += excess * direction
        
        # Update positions with adaptive learning rate
        lr = 0.01 * (1.0 - step / max(n_steps, 1)) + 0.001
        pts += lr * forces
        
        # Project to unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]
    
    return pts


def _hex_lattice_init(r_guess=0.12):
    """Generate a hexagonal lattice packing as initial guess."""
    spacing = 2.0 * r_guess
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return pts[idx]
    else:
        return _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])


def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
    """Generate points from concentric rings."""
    points = np.zeros((N, 2))
    idx = 0
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(angle_offsets) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return points


def _random_init(seed):
    """Random initialization within unit disk."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        x = rng.uniform(-1, 1, size=(N * 2, 2))
        r = np.hypot(x[:, 0], x[:, 1])
        valid = r <= 0.95
        pts.extend(x[valid].tolist())
    pts = np.array(pts[:N])
    return pts


def _soft_min_optimize(pts_init, beta_schedule, max_iter_per_stage=500):
    """
    Multi-stage soft-min optimization with beta annealing.
    """
    current_pts = pts_init.copy()
    
    for stage, beta in enumerate(beta_schedule):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        def objective(x):
            return _soft_min_obj_and_grad(x, beta)
        
        x0 = current_pts.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter_per_stage, 'ftol': 1e-14,
                                   'gtol': 1e-10, 'maxls': 50})
        current_pts = result.x.reshape(N, 2)
        
        # Project to ensure validity
        dists = np.linalg.norm(current_pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            current_pts[mask] = current_pts[mask] / dists[mask, np.newaxis]
    
    return current_pts


def _polish(pts, beta=500000, max_iter=1000):
    """
    High-precision polish pass with very high beta and tight tolerances.
    """
    if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 2:
        return pts
    
    def objective(x):
        return _soft_min_obj_and_grad(x, beta)
    
    x0 = pts.flatten()
    try:
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter, 'ftol': 1e-15,
                                   'gtol': 1e-13, 'maxls': 100})
        pts_polished = result.x.reshape(N, 2)
        
        # Project to ensure validity
        dists = np.linalg.norm(pts_polished, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts_polished[mask] = pts_polished[mask] / dists[mask, np.newaxis]
        
        return pts_polished
    except Exception:
        return pts


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Multiple initializations
    inits = [
        _hex_lattice_init(0.12),
        _hex_lattice_init(0.125),
        _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        _ring_init([1, 7, 13, 19, 10], [0.0, 0.22, 0.48, 0.74, 0.92]),
        _random_init(42),
        _random_init(123),
        _random_init(456),
    ]
    
    # Beta schedule for multi-stage optimization
    beta_schedule = [100, 500, 2000, 10000, 50000]
    
    for init_idx, init_pts in enumerate(inits):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break
        if len(init_pts) < N:
            continue
        
        # Force-directed relaxation as pre-conditioner
        relaxed_pts = _force_directed_relax(init_pts, n_steps=200, target_r=0.12)
        
        # Multi-stage soft-min optimization
        optimized_pts = _soft_min_optimize(relaxed_pts, beta_schedule, max_iter_per_stage=300)
        
        # High-precision polish
        polished_pts = _polish(optimized_pts, beta=500000, max_iter=500)
        
        # Evaluate
        score = _score(polished_pts)
        if score > best_score:
            best_score = score
            best_pts = polished_pts.copy()
        
        # Also evaluate the unpolished result
        score_unpolished = _score(optimized_pts)
        if score_unpolished > best_score:
            best_score = score_unpolished
            best_pts = optimized_pts.copy()
        
        # Check time
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 5:
            break
    
    if best_pts is None:
        best_pts = _hex_lattice_init(0.12)
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
