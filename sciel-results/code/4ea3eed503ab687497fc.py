"""Packing 200 equal circles in a unit square: L-BFGS-B with analytic gradient and force-directed pre-processing."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill


def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances using pdist (faster and more memory efficient)
    dists = pdist(pts)
    min_dist = np.min(dists)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0


def _hex_init():
    """Generate a hexagonal lattice initialization."""
    best_pts = None
    best_r = 0
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.9, s_est * 1.1, 10):
        pts_list = []
        row = 0
        while True:
            y = -0.5 + s / 2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s / 2:
                break
            offset = (row % 2) * s / 2
            x = -0.5 + s / 2 + offset
            while x <= 0.5 - s / 2:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts


def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:N].copy()


def _random_init(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _force_directed_relax(pts, steps=50, dt=0.001, damping=0.9):
    """
    Simple force-directed relaxation to remove overlaps and spread points.
    Uses repulsive forces between pairs and from walls.
    """
    pts = pts.copy()
    v = np.zeros_like(pts)
    
    for _ in range(steps):
        # Calculate forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        # Using pdist for distances, but we need vectors.
        # For N=200, O(N^2) is fine.
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    # Random kick if coincident
                    forces[i] += np.random.randn(2) * 0.1
                    forces[j] -= np.random.randn(2) * 0.1
                    continue
                dist = np.sqrt(dist_sq)
                # Force magnitude ~ 1/dist^2 to be strong at close range
                # Limit force to avoid explosion
                f_mag = min(1.0 / (dist * dist), 10.0)
                f_vec = f_mag * diff / dist
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Wall repulsion
        for i in range(N):
            # Left/Right
            if pts[i][0] < -0.5 + 0.1:
                forces[i][0] += ( -0.5 + 0.1 - pts[i][0] ) * 10.0
            elif pts[i][0] > 0.5 - 0.1:
                forces[i][0] += ( 0.5 - 0.1 - pts[i][0] ) * 10.0
            # Top/Bottom
            if pts[i][1] < -0.5 + 0.1:
                forces[i][1] += ( -0.5 + 0.1 - pts[i][1] ) * 10.0
            elif pts[i][1] > 0.5 - 0.1:
                forces[i][1] += ( 0.5 - 0.1 - pts[i][1] ) * 10.0

        # Update velocity and position
        v = v * damping + forces * dt
        pts += v * dt
        
        # Clip to square to prevent escaping
        pts = np.clip(pts, -0.5, 0.5)

    return pts


def _soft_min_obj_grad(x, alpha):
    """
    Compute the soft-min objective and its analytic gradient.
    Objective: -soft_min(distances)
    Soft_min(d) = -1/alpha * log(sum(exp(-alpha * d)))
    """
    pts = x.reshape(N, 2)
    
    # 1. Boundary distances (4 per point)
    # d_left, d_right, d_bottom, d_top
    d_left = 0.5 - pts[:, 0]
    d_right = 0.5 + pts[:, 0]
    d_bottom = 0.5 - pts[:, 1]
    d_top = 0.5 + pts[:, 1]
    
    # Stack boundary distances: shape (N, 4)
    b_dists = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
    
    # 2. Pairwise distances
    # We need the actual distances and vectors for gradients
    # pdist gives condensed vector.
    # To get gradients efficiently, we can compute pairwise differences.
    # N=200, so N^2 = 40000 vectors.
    
    # Compute pairwise distances and vectors
    # diff[i, j] = pts[i] - pts[j]
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal and upper triangle (since dists[i,j] == dists[j,i])
    # We only need to sum over i < j
    # Create a mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Get unique pairwise distances
    p_dists = dists[mask]
    p_vecs = diff[mask] # Shape (N*(N-1)/2, 2)
    
    # Combine all distances for the soft-min sum
    all_dists = np.concatenate([b_dists.flatten(), p_dists])
    
    # Compute weights w_k = exp(-alpha * d_k) / sum(exp(-alpha * d_k))
    # For numerical stability, subtract min
    m = np.min(all_dists)
    exp_terms = np.exp(-alpha * (all_dists - m))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Objective value
    soft_min_val = -(1.0 / alpha) * (m + np.log(sum_exp))
    obj_val = -soft_min_val
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # 1. Boundary gradients
    # b_dists shape (N, 4)
    # weights for boundary: weights[:4*N] reshaped to (N, 4)
    w_b = weights[:4 * N].reshape(N, 4)
    
    # Gradients for d_left, d_right, d_bottom, d_top w.r.t x, y
    # d_left = 0.5 - x  -> grad = [-1, 0]
    # d_right = 0.5 + x -> grad = [1, 0]
    # d_bottom = 0.5 - y -> grad = [0, -1]
    # d_top = 0.5 + y   -> grad = [0, 1]
    
    # Accumulate boundary contributions
    # grad_x += w_left * (-1) + w_right * (1)
    # grad_y += w_bottom * (-1) + w_top * (1)
    grad[:, 0] += -w_b[:, 0] + w_b[:, 1]
    grad[:, 1] += -w_b[:, 2] + w_b[:, 3]
    
    # 2. Pairwise gradients
    # w_p = weights[4*N:]
    w_p = weights[4 * N:]
    
    # For each pair (i, j), force on i is w * (p_i - p_j) / dist
    # force on j is -w * (p_i - p_j) / dist
    
    # p_vecs[i] is p_i - p_j for the i-th pair in the condensed list
    # We need to map these back to i and j indices to accumulate into grad.
    
    # This mapping is tricky with a flat array.
    # Alternative: Loop over pairs? N=200, ~20k pairs. Python loop is slow.
    # Let's use numpy advanced indexing if possible, or just loop if optimized.
    # Actually, we can reconstruct the full matrix of weights and vectors.
    
    # Let's create a full (N, N) matrix of weights for pairs
    W_pair = np.zeros((N, N))
    V_pair = np.zeros((N, N, 2))
    
    # Fill upper triangle
    # We need the indices corresponding to the mask
    idx_i, idx_j = np.where(mask)
    W_pair[idx_i, idx_j] = w_p
    W_pair[idx_j, idx_i] = w_p
    
    V_pair[idx_i, idx_j] = p_vecs
    V_pair[idx_j, idx_i] = -p_vecs
    
    # Normalize vectors by distance
    # dists is (N, N)
    # Avoid division by zero
    safe_dists = np.where(dists > 1e-8, dists, 1.0)
    V_pair /= safe_dists[:, :, np.newaxis]
    
    # Sum over neighbors
    # grad_i = sum_j W_pair[i, j] * V_pair[i, j]
    grad += np.sum(W_pair[:, :, np.newaxis] * V_pair, axis=1)
    
    return obj_val, grad.flatten()


def _optimize(pts, time_budget, alphas=[20.0, 50.0, 100.0]):
    """
    Run L-BFGS-B optimization on the initial points with increasing alpha.
    Uses analytic gradient.
    """
    t0 = time.time()
    x0 = pts.flatten()

    for alpha in alphas:
        if time.time() - t0 > time_budget:
            break

        def obj(x):
            val, _ = _soft_min_obj_grad(x, alpha)
            return val

        def jac(x):
            _, g = _soft_min_obj_grad(x, alpha)
            return g

        try:
            res = minimize(
                obj, x0, jac=jac, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'gtol': 1e-5, 'disp': False}
            )
            x0 = res.x
        except Exception:
            pass

    return x0.reshape(N, 2)


def construct():
    t_start = time.time()

    initializers = [
        ("hex", _hex_init),
        ("grid", _grid_init),
        ("rand0", lambda: _random_init(0)),
    ]

    best_pts = None
    best_r = 0.0

    for name, init_fn in initializers:
        if time.time() > DEADLINE - 10:
            break

        pts = init_fn()
        
        # Force-directed relaxation to improve initial guess
        # Only do this if we have time
        if time.time() < DEADLINE - 20:
            pts = _force_directed_relax(pts, steps=20)
            
        pts = np.clip(pts, -0.5, 0.5)
        
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

        # Optimize
        remaining = DEADLINE - time.time()
        if remaining < 5:
            break

        # Allocate time
        if name == "hex":
            time_budget = min(remaining * 0.45, 40.0)
        else:
            time_budget = min(remaining * 0.25, 20.0)

        opt_pts = _optimize(pts, time_budget)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)

        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    # Final polish
    remaining = DEADLINE - time.time()
    if remaining > 5 and best_pts is not None:
        time_budget = min(remaining * 0.4, 10.0)
        opt_pts = _optimize(best_pts, time_budget, alphas=[100.0])
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    if best_pts is None:
        best_pts = _hex_init()

    return best_pts
