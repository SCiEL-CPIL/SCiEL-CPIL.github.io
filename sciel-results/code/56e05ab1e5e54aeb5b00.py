"""
Circle packing via direct max-min optimization with soft-min smoothing.
Strategy:
1. Define the objective as the minimum of all pair distances and boundary margins.
2. Since min is non-smooth, use a soft-min approximation:
   f(x) = -1/beta * log(sum(exp(-beta * x_i)))
   where x_i are the constraint values (distances/2 and margins).
   As beta increases, f(x) approaches min(x_i).
3. Optimize f(x) using L-BFGS-B (which handles gradients well).
4. Use multiple initializations: hexagonal lattice, random, and ring-based.
5. After the smooth optimization, perform a local search:
   - Identify the "bottleneck" constraint (the one closest to 0).
   - Try to move the involved circles to increase this constraint,
     while not decreasing any other constraint below the current best.
   - This is a form of pairwise local search or "kick" search.
6. Return the best configuration found.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_score(pts):
    """Compute the true packing radius (nonsmooth)."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(pts ** 2, axis=-1))
    min_margin = np.min(1.0 - radii)
    return min(min_pair, min_margin)

def _softmin_objective_and_grad(x, beta):
    """
    Objective: maximize the minimum of all constraints.
    Constraints:
      - Pair distances / 2
      - Boundary margins: 1 - ||c_i||
    
    We minimize -softmin(constraints).
    softmin(x) = -1/beta * log(sum(exp(-beta * x)))
    grad_x softmin(x) = exp(-beta * x) / sum(exp(-beta * y))
    
    Let z_i be the constraints.
    f(x) = -softmin(z) = 1/beta * log(sum(exp(-beta * z_i)))
    grad f = 1/beta * (sum(-beta * exp(-beta * z_i) * grad z_i) / sum(exp(-beta * z_i)))
           = - (sum(exp(-beta * z_i) * grad z_i) / sum(exp(-beta * z_i)))
    
    Wait, let's re-derive carefully.
    Let S = sum(exp(-beta * z_i)).
    f = (1/beta) * log(S).
    df/dx = (1/beta) * (1/S) * dS/dx
    dS/dx = sum(-beta * exp(-beta * z_i) * dz_i/dx)
    df/dx = (1/beta) * (1/S) * (-beta) * sum(exp(-beta * z_i) * dz_i/dx)
    df/dx = - (1/S) * sum(exp(-beta * z_i) * dz_i/dx)
    
    This is a weighted average of the gradients of the constraints, weighted by exp(-beta * z_i).
    The constraint with the smallest z_i (largest exp) gets the most weight.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Compute constraints
    # 1. Pairs
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # We need gradients for each constraint.
    # It's more efficient to accumulate the weighted gradient.
    
    # Weights for pairs
    # z_ij = dists_ij / 2
    # weight_ij = exp(-beta * z_ij)
    
    # To avoid overflow/underflow in exp, we can subtract the min z.
    # But since we are maximizing the min, the min z is small (hopefully).
    # If beta is large, exp(-beta * z) can underflow for large z.
    # We can use a log-sum-exp trick if needed, but for now, let's try direct.
    # To be safe, let's normalize by the min z.
    
    # Actually, let's just compute the weights and normalize them.
    # The formula is df/dx = - sum(w_i * grad_z_i) where w_i = exp(-beta*z_i) / sum(exp(-beta*z_j))
    
    # Let's collect all z values and their gradients.
    # This is memory intensive if we store all gradients.
    # Instead, we compute the weighted sum of gradients on the fly.
    
    grad = np.zeros_like(pts)
    
    # Pair constraints
    iu = np.triu_indices(n, k=1)
    d_pairs = dists[iu]
    z_pairs = d_pairs / 2.0
    
    # Compute weights
    # To prevent underflow, shift z by min_z
    min_z = np.min(z_pairs)
    # Boundary constraints
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    z_bound = 1.0 - radii
    min_z = np.min([min_z, np.min(z_bound)])
    
    # Shifted z
    z_pairs_shifted = z_pairs - min_z
    z_bound_shifted = z_bound - min_z
    
    w_pairs = np.exp(-beta * z_pairs_shifted)
    w_bound = np.exp(-beta * z_bound_shifted)
    
    total_w = np.sum(w_pairs) + np.sum(w_bound)
    
    if total_w == 0:
        return 0.0, np.zeros_like(x)
        
    w_pairs /= total_w
    w_bound /= total_w
    
    # Gradient of z_ij = dists_ij / 2
    # dz_ij/dx_i = (x_i - x_j) / (2 * dists_ij)
    # dz_ij/dx_j = (x_j - x_i) / (2 * dists_ij)
    
    # We need to accumulate:
    # grad += - sum (w_ij * dz_ij/dx_i)
    
    # Vectorized accumulation is tricky. Loop is okay for N=50.
    for k in range(len(d_pairs)):
        if w_pairs[k] > 0:
            i, j = iu[0][k], iu[1][k]
            d = d_pairs[k]
            if d > 1e-12:
                vec = (pts[i] - pts[j]) / (2.0 * d)
                grad[i] -= w_pairs[k] * vec
                grad[j] += w_pairs[k] * vec
            else:
                # If distance is 0, gradient is undefined. Add a small random kick?
                # Or just skip.
                pass

    # Boundary constraints
    # z_i = 1 - ||x_i||
    # dz_i/dx_i = -x_i / ||x_i||
    for i in range(n):
        if w_bound[i] > 0:
            r = radii[i]
            if r > 1e-12:
                grad[i] -= w_bound[i] * (-pts[i] / r)
            else:
                # Gradient is 0 at origin for norm, but constraint is 1.
                # Actually d(1-r)/dx = -x/r. At x=0, it's 0.
                pass

    # Objective value
    # f = 1/beta * log(sum(exp(-beta * z)))
    # We shifted z, so sum(exp(-beta * (z - min_z))) = exp(beta * min_z) * sum(exp(-beta * z))
    # log(sum(exp(-beta * z))) = -beta * min_z + log(sum(exp(-beta * (z - min_z))))
    # f = 1/beta * (-beta * min_z + log(total_w_unshifted))
    # Wait, total_w was normalized.
    # Let's compute the unnormalized sum for the value.
    
    # Actually, we just need the value for the optimizer.
    # value = -softmin(z) = min_z - (1/beta) * log(sum(exp(-beta * (z - min_z))))
    # This is a lower bound on the min? No, softmin is a lower bound on min.
    # softmin(z) <= min(z).
    # So -softmin(z) >= -min(z).
    # We are minimizing -softmin(z).
    
    # Let's just return the value of -softmin(z).
    # log_sum_exp = np.log(np.sum(np.exp(-beta * z_pairs_shifted)) + np.sum(np.exp(-beta * z_bound_shifted)))
    # This is log(total_w_unshifted).
    # But we normalized w.
    # Let's recompute the sum without normalization for the value.
    
    # To be safe and consistent, let's just use the normalized weights to estimate the value?
    # No, the value should be the actual softmin.
    
    # Let's compute the value properly.
    # max_z = min_z (since we subtracted min)
    # sum_exp = np.sum(np.exp(-beta * z_pairs_shifted)) + np.sum(np.exp(-beta * z_bound_shifted))
    # log_sum_exp = np.log(sum_exp)
    # value = min_z - (1.0 / beta) * log_sum_exp
    
    # But we already computed w_pairs and w_bound normalized.
    # sum_exp = total_w (before normalization)
    # We lost total_w? No, we have it.
    
    # Wait, in the code above:
    # w_pairs = exp(...)
    # w_bound = exp(...)
    # total_w = sum(w_pairs) + sum(w_bound)
    # Then we normalized.
    # So total_w is the unnormalized sum.
    
    value = min_z - (1.0 / beta) * np.log(total_w)
    
    return value, grad.flatten()

def _hex_init(spacing):
    """Hexagonal lattice initialization."""
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[np.argsort(radius, kind='stable')[:N]]
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

def _local_search(pts, max_iters=1000, time_limit=5.0):
    """
    Local search to escape local minima.
    Identify the bottleneck constraint.
    Try to move the circles involved to increase it.
    """
    t0 = time.time()
    current_score = _compute_score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    for _ in range(max_iters):
        if time.time() - t0 > time_limit:
            break
            
        # Find the bottleneck
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        min_margin = np.min(1.0 - radii)
        min_margin_idx = np.argmin(1.0 - radii)
        
        if min_pair_dist / 2.0 < min_margin:
            # Bottleneck is a pair
            i, j = min_pair_idx
            if i > j: i, j = j, i
            # Try to move i and j apart
            # Direction: from j to i
            vec = pts[i] - pts[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            else:
                vec = np.random.randn(2)
                vec /= np.linalg.norm(vec)
                
            # Try small moves
            for step in [0.01, 0.005, 0.001]:
                new_pts = pts.copy()
                new_pts[i] += step * vec
                new_pts[j] -= step * vec
                
                # Check if this improves the bottleneck without breaking others
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
                    pts = new_pts
                    break
        else:
            # Bottleneck is boundary
            i = min_margin_idx
            # Try to move i inward
            vec = -pts[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            else:
                vec = np.random.randn(2)
                vec /= np.linalg.norm(vec)
                
            for step in [0.01, 0.005, 0.001]:
                new_pts = pts.copy()
                new_pts[i] += step * vec
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
                    pts = new_pts
                    break
                    
    return best_pts, best_score

def construct():
    t_start = time.time()
    best_pts = None
    best_score = -1.0
    
    inits = [
        _hex_init(0.25),
        _hex_init(0.22),
        _random_init(42),
        _random_init(123),
        _random_init(777),
    ]
    
    # Beta schedule: start low, increase
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() - t_start > DEADLINE - 10:
            break
        if len(init_pts) < N:
            continue
            
        current_pts = init_pts.copy()
        
        for beta in betas:
            if time.time() - t_start > DEADLINE - 10:
                break
                
            time_remaining = DEADLINE - (time.time() - t_start)
            time_for_opt = min(time_remaining * 0.2, 10.0)
            if time_for_opt < 1.0:
                break
                
            # Optimize with current beta
            def obj_and_grad(x):
                return _softmin_objective_and_grad(x, beta)
                
            x0 = current_pts.flatten()
            
            try:
                res = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                               options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6})
                current_pts = res.x.reshape(-1, 2)
            except Exception:
                pass
                
            # Check score
            score = _compute_score(current_pts)
            if score > best_score:
                best_score = score
                best_pts = current_pts.copy()
                
        # Local search polish
        if time.time() - t_start < DEADLINE - 5:
            polished_pts, polished_score = _local_search(current_pts, max_iters=2000, time_limit=5.0)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
                
    if best_pts is None:
        best_pts = _hex_init(0.25)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(_compute_score(pts))
