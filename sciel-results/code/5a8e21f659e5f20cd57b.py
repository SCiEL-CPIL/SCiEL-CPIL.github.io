"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside the unit circle
    dists_to_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_to_origin >= 1.0):
        return 0.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    # Minimum distance to boundary
    min_boundary = 1.0 - np.max(dists_to_origin)
    return min(min_pairwise, min_boundary)


def _softmin(x, beta):
    """Differentiable approximation to min(x)."""
    # Softmin: -1/beta * log(sum(exp(-beta * x)))
    # To avoid overflow, we shift by the max
    m = np.max(x)
    return m - np.log(np.sum(np.exp(-beta * (x - m)))) / beta


def _objective_and_grad(x, beta):
    """
    Objective: -softmin(distances/2, boundary_distances)
    We want to maximize the minimum distance, so we minimize the negative.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Boundary distances: 1 - |p_i|
    radii = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - radii
    
    # Pairwise distances / 2
    # Vectorized computation of pairwise distances
    # diff[i, j] = pts[i] - pts[j]
    # dist[i, j] = ||diff[i, j]||
    # We only need i < j
    # To save memory and time, we can compute the full matrix or just upper triangle.
    # For N=50, full matrix is 2500x2, which is fine.
    
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Extract upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Combine all "constraints" (distances)
    # We want max(min(pair_dists, boundary_dists))
    # So we minimize -softmin(concat(pair_dists, boundary_dists))
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Softmin value
    val = _softmin(all_dists, beta)
    
    # Gradient
    # d/dx (-softmin) = - d/dx softmin
    # softmin(x) = m - (1/beta) log(sum(exp(-beta(x-m))))
    # Let S = sum(exp(-beta(x-m)))
    # d/dx_i softmin = - (1/beta) * d/dx_i log S
    # d/dx_i log S = (1/S) * sum_j (d/dx_i exp(-beta(x_j-m)))
    # d/dx_i exp(-beta(x_j-m)) = exp(-beta(x_j-m)) * (-beta) * (delta_{ij} - 0) 
    # Wait, m = max(x). If m changes, it's complex. 
    # Standard softmin gradient:
    # g_i = (1/beta) * (exp(-beta(x_i - m)) / S) * beta = exp(-beta(x_i - m)) / S
    # Actually, let's use the log-sum-exp trick derivative.
    # f(x) = -1/beta * log(sum(exp(-beta * x)))
    # f'(x_i) = -1/beta * (1/sum) * exp(-beta * x_i) * (-beta) = exp(-beta * x_i) / sum(exp(-beta * x))
    # This is a probability distribution p_i.
    # So grad softmin = p_i.
    # We want to minimize -softmin, so grad objective = -p_i.
    
    # Compute p_i for all_dists
    # p_i = exp(-beta * (all_dists_i - m)) / sum(exp(-beta * (all_dists - m)))
    exp_vals = np.exp(-beta * (all_dists - np.max(all_dists)))
    sum_exp = np.sum(exp_vals)
    p = exp_vals / sum_exp
    
    # Now we need to chain rule back to x.
    # all_dists consists of pair_dists and boundary_dists.
    # grad_x = sum_k p_k * grad_x (all_dists_k)
    
    grad = np.zeros_like(x)
    
    # Gradient for boundary parts
    # all_dists[n_pairs:] corresponds to boundary_dists
    # boundary_dists_i = 1 - ||p_i||
    # d/dp_i (1 - ||p_i||) = -p_i / ||p_i||
    
    p_boundary = p[len(pair_dists):]
    # Accumulate into grad
    # For each point i, add p_boundary[i] * (-pts[i] / radii[i])
    # Handle radii = 0
    safe_radii = np.where(radii < 1e-10, 1e-10, radii)
    boundary_grad = -p_boundary[:, np.newaxis] * (pts / safe_radii[:, np.newaxis])
    grad += boundary_grad.flatten()
    
    # Gradient for pairwise parts
    # all_dists[:n_pairs] corresponds to pair_dists
    # pair_dists_k = ||p_i - p_j|| / 2 for (i,j) in iu
    p_pairs = p[:len(pair_dists)]
    
    # iu[0] contains i indices, iu[1] contains j indices
    i_idx = iu[0]
    j_idx = iu[1]
    
    # diff_k = pts[i] - pts[j]
    # dist_k = ||diff_k||
    # d/dp_i (dist_k/2) = (pts[i] - pts[j]) / (2 * ||pts[i] - pts[j]||)
    # d/dp_j (dist_k/2) = (pts[j] - pts[i]) / (2 * ||pts[i] - pts[j]||)
    
    # We can vectorize this accumulation
    # For each pair k, weight w_k = p_pairs[k]
    # grad_i += w_k * (pts[i] - pts[j]) / dist_k
    # grad_j -= w_k * (pts[i] - pts[j]) / dist_k
    
    # dist_k is pair_dists[k] * 2
    # We have dists matrix, so dist_k = dists[i_idx, j_idx]
    d_k = dists[i_idx, j_idx]
    safe_d_k = np.where(d_k < 1e-10, 1e-10, d_k)
    
    # unit vector from j to i
    u = (pts[i_idx] - pts[j_idx]) / safe_d_k[:, np.newaxis]
    
    # weighted unit vectors
    w_u = p_pairs[:, np.newaxis] * u
    
    # Accumulate to i
    # np.add.at is safe for duplicate indices, though here indices are unique in pairs
    # But multiple pairs can share an i.
    np.add.at(grad, i_idx, w_u)
    np.add.at(grad, j_idx, -w_u)
    
    # Objective is -softmin, so we return -val, -grad
    # Wait, minimize expects to minimize f.
    # We want to maximize min_dist.
    # f = -softmin.
    # grad f = -grad softmin = -p.
    # So we return -val, -grad_accumulated?
    # Let's re-verify.
    # val = softmin(all_dists).
    # We want to minimize -val.
    # d(-val)/dx = -d(val)/dx = -p_chain.
    # My `grad` variable currently holds d(val)/dx (the chain rule result for softmin).
    # So objective gradient is -grad.
    
    return -val, -grad


def _ring_init(n_points):
    """
    Initialize points in concentric rings.
    Ring 0: 1 point at center.
    Ring 1: 6 points.
    Ring 2: 12 points.
    Ring 3: 18 points?
    Total 1+6+12+... 
    For N=50:
    1 + 6 = 7
    7 + 12 = 19
    19 + 18 = 37
    37 + 13 = 50 (last ring incomplete)
    
    Radii:
    r_0 = 0
    r_1 ~ 0.15
    r_2 ~ 0.30
    r_3 ~ 0.45
    r_4 ~ 0.60
    ...
    """
    pts = []
    remaining = n_points
    ring_idx = 0
    radius = 0.0
    spacing = 0.15 # approximate
    
    while remaining > 0:
        if ring_idx == 0:
            count = 1
            angles = [0.0]
        else:
            # Number of points in ring k is roughly proportional to circumference
            # 6 * k is a good hexagonal approximation
            count = 6 * ring_idx
            if count > remaining:
                count = remaining
            
            # Distribute angles evenly
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Offset angle for each ring to stagger
            offset = np.pi / (3 * ring_idx) if ring_idx > 0 else 0
            angles += offset
            
        for a in angles:
            x = radius * np.cos(a)
            y = radius * np.sin(a)
            pts.append([x, y])
        
        remaining -= len(angles)
        ring_idx += 1
        radius += spacing
        # Adjust spacing slightly to fit inside
        # If radius gets too large, we might need to compress
        # For now, just linear growth
        
    pts = np.array(pts)
    
    # Scale to fit within unit circle roughly
    # Max radius is approx 0.15 * (num_rings - 1)
    # For 50 points, rings: 1, 6, 12, 18, 13. Radii: 0, 0.15, 0.3, 0.45, 0.6.
    # Max dist 0.6 + 0.125 (circle radius) = 0.725 < 1.0. Good.
    
    return pts


def _random_init(n_points, seed):
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    r = np.sqrt(rng.uniform(0, 1, n_points))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _optimize(points, beta_start=10.0, beta_end=1000.0, max_iter=1000, deadline=None):
    """
    Optimize using L-BFGS-B with increasing beta (annealing the softmin).
    """
    x0 = points.flatten()
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Schedule beta
    # We'll do a few stages with increasing beta
    stages = [
        (10.0, 200),
        (50.0, 300),
        (200.0, 300),
        (1000.0, 200)
    ]
    
    current_x = x0
    
    for beta, iters in stages:
        if deadline and time.time() > deadline:
            break
            
        def obj_grad(x):
            val, grad = _objective_and_grad(x, beta)
            return val, grad
            
        try:
            res = minimize(
                obj_grad,
                current_x,
                jac=True, # Gradient provided
                method='L-BFGS-B',
                options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            current_x = res.x
            pts = current_x.reshape(-1, 2)
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            pass
            
    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_points = None
    best_score = -1.0
    
    # Try ring initialization
    try:
        pts = _ring_init(N)
        # Small random jitter to break symmetry
        rng = np.random.RandomState(42)
        pts += rng.normal(0, 0.01, pts.shape)
        
        # Optimize
        opt_pts = _optimize(pts, deadline=deadline)
        score = _compute_score(opt_pts)
        
        if score > best_score:
            best_score = score
            best_points = opt_pts.copy()
    except Exception:
        pass
    
    # If time remains, try random init with a single optimization pass
    if time.time() < deadline - 10.0:
        try:
            pts = _random_init(N, seed=1)
            opt_pts = _optimize(pts, beta_start=10.0, beta_end=100.0, max_iter=500, deadline=deadline)
            score = _compute_score(opt_pts)
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
        except Exception:
            pass
            
    # Fallback if nothing worked
    if best_points is None:
        # Simple hexagonal grid fallback
        spacing = 0.2
        pts = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x*x + y*y < 1.0:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            # Sort by distance to center
            dists = np.linalg.norm(pts, axis=1)
            order = np.argsort(dists)
            best_points = pts[order[:N]]
        else:
            # Random fallback
            best_points = _random_init(N, seed=0)
            
    return best_points
# EVOLVE-BLOCK-END
