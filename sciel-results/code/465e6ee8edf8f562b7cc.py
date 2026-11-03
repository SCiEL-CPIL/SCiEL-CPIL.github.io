import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))

def _compute_min_dist(points):
    """
    Compute the minimum distance metric r for a given configuration.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|))
    Returns (r, bottleneck_type, bottleneck_indices)
    bottleneck_type: 0 for boundary, 1 for pair
    bottleneck_indices: indices of the bottleneck constraint(s)
    """
    points = points.reshape(-1, 2)
    n = len(points)
    
    # Boundary distances
    radii = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(radii)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    min_pair = np.min(pair_dists)
    
    if min_boundary < min_pair:
        # Bottleneck is boundary
        # Find all indices where boundary distance is close to min
        close_indices = np.where(np.abs(radii - min_boundary) < 1e-10)[0]
        return min_boundary, 0, close_indices
    else:
        # Bottleneck is pair
        # Find all pairs where distance is close to min
        close_pairs = np.where(np.abs(pair_dists - min_pair) < 1e-10)[0]
        # Map back to (i, j)
        iu_arr = np.array(iu)
        close_idx_pairs = iu_arr[close_pairs]
        return min_pair, 1, close_idx_pairs

def _score_vec(points):
    """Vectorized score computation for quick evaluation."""
    points = points.reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r

def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 50):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points

def _local_optimize_lbfgs(c, time_limit=10.0):
    """Local optimization using L-BFGS-B with smooth-min objective."""
    pair_idx = _pairwise_indices(N)
    
    def neg_soft_min(x):
        c = x.reshape(N, 2)
        ci = c[pair_idx[:, 0]]
        cj = c[pair_idx[:, 1]]
        d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
        b = 1 - np.hypot(c[:, 0], c[:, 1])
        all_vals = np.concatenate([d, b])
        beta = 200.0
        return np.log(np.sum(np.exp(-beta * all_vals))) / beta
    
    t_start = time.time()
    try:
        res = minimize(
            neg_soft_min,
            c.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    return c

def _polish_gradient_descent(points, time_limit=5.0):
    """
    High-precision gradient descent polish.
    Moves the bottleneck circles to increase the minimum distance.
    """
    pts = points.copy().flatten()
    t_start = time.time()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    
    lr = 0.01  # Learning rate
    momentum = 0.9
    v = np.zeros_like(pts)
    
    while time.time() - t_start < time_limit:
        c = pts.reshape(N, 2)
        r, btype, bidx = _compute_min_dist(c)
        
        if r > best_score:
            best_score = r
            best_pts = pts.copy()
        
        # Compute gradient of r with respect to positions
        grad = np.zeros((N, 2))
        
        if btype == 0:
            # Boundary bottleneck: move circles inwards
            for i in bidx:
                norm = np.linalg.norm(c[i])
                if norm > 1e-10:
                    # Gradient of (1 - |c_i|) is -c_i / |c_i|
                    # We want to maximize r, so we move in direction of gradient
                    # grad r / grad c_i = -c_i / |c_i|
                    # To increase r, we should move c_i in direction -grad = c_i / |c_i| ? 
                    # Wait. r = 1 - |c|. dr/dc = -c/|c|.
                    # We want to increase r. So we move c in direction -dr/dc = c/|c|.
                    # This means moving OUTWARDS? No, that decreases r.
                    # Let's re-evaluate.
                    # r = 1 - |c|. If we move c towards origin, |c| decreases, r increases.
                    # Direction towards origin is -c/|c|.
                    # So we should add -c/|c| to c.
                    # Gradient of r w.r.t c is -c/|c|.
                    # We want to maximize r, so we move in direction of gradient?
                    # No, gradient points in direction of steepest increase.
                    # If grad = -c/|c|, moving in direction of grad means moving towards origin.
                    # Correct.
                    grad[i] = -c[i] / norm
        else:
            # Pair bottleneck: move circles apart
            for idx in bidx:
                i, j = idx
                d_vec = c[i] - c[j]
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 1e-10:
                    # r = |c_i - c_j| / 2
                    # dr/dc_i = (c_i - c_j) / (2 * |c_i - c_j|)
                    # dr/dc_j = -(c_i - c_j) / (2 * |c_i - c_j|)
                    # We want to maximize r.
                    # Move c_i in direction of dr/dc_i (away from j)
                    # Move c_j in direction of dr/dc_j (away from i)
                    grad[i] += d_vec / d_norm
                    grad[j] -= d_vec / d_norm
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Momentum update
        v = momentum * v + lr * grad
        pts = pts + v
        
        # Ensure validity (optional, but helps if we overshoot)
        # If we overshoot, the next iteration's gradient will correct it.
        # For now, we just rely on the gradient.
        
        # Adaptive learning rate: if score doesn't improve, reduce lr
        # (Simplified: just keep constant for now, as gradient is normalized)
    
    return best_pts.reshape(N, 2), best_score

def construct():
    t_start = time.time()
    
    # 1. Hexagonal Initialization
    points = _hexagonal_init(N)
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # 2. Coarse Optimization with L-BFGS
    if time.time() < DEADLINE - 30:
        c_opt = _local_optimize_lbfgs(points, time_limit=min(20.0, DEADLINE - (time.time() - t_start) - 30))
        r_opt = _score_vec(c_opt)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # 3. Fine Optimization with Gradient Descent Polish
    if time.time() < DEADLINE - 10:
        polished, r_polish = _polish_gradient_descent(best_points, time_limit=min(20.0, DEADLINE - (time.time() - t_start) - 10))
        if r_polish > best_score:
            best_score = r_polish
            best_points = polished.copy()
    
    # 4. Multi-Start Perturb and Polish
    # Try to escape local minima by perturbing the best solution
    n_restarts = 0
    while time.time() < DEADLINE - 15 and n_restarts < 10:
        # Perturb all points slightly
        perturbation_scale = 0.05 / (1.0 + 0.5 * n_restarts)
        x0 = best_points + np.random.randn(N, 2) * perturbation_scale
        
        # Coarse optimize the perturbed solution
        c_opt = _local_optimize_lbfgs(x0, time_limit=min(10.0, DEADLINE - (time.time() - t_start) - 15))
        r_opt = _score_vec(c_opt)
        
        # Fine polish
        if time.time() < DEADLINE - 5:
            polished, r_polish = _polish_gradient_descent(c_opt, time_limit=min(10.0, DEADLINE - (time.time() - t_start) - 5))
            r_opt = max(r_opt, r_polish)
            if r_polish > r_opt:
                c_opt = polished
        
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
        
        n_restarts += 1
    
    # 5. Final Polish
    if time.time() < DEADLINE - 2:
        polished, r_final = _polish_gradient_descent(best_points, time_limit=min(5.0, DEADLINE - (time.time() - t_start) - 2))
        if r_final > best_score:
            best_points = polished
            best_score = r_final
    
    # Ensure validity: clamp points that might be slightly outside
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale
    
    return c_final
