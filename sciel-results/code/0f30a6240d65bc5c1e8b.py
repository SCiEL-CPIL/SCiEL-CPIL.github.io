"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    min_dist_to_wall = np.min(0.5 - np.abs(centers))
    if min_dist_to_wall < 0:
        return 0.0
    n = len(centers)
    min_pair_dist = np.inf
    for i in range(n):
        diffs = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if len(dists) > 0:
            d = np.min(dists)
            if d < min_pair_dist:
                min_pair_dist = d
    if not np.isfinite(min_pair_dist):
        return 0.0
    r = min(min_dist_to_wall, min_pair_dist / 2.0)
    return max(0.0, r)


def _soft_min_logsumexp(vals, beta):
    """Soft min via -logsumexp(-beta * vals) / beta."""
    if len(vals) == 0:
        return 0.0
    m = np.min(vals)
    # For numerical stability: soft_min = m - log(sum(exp(-beta*(vals - m))))/beta
    shifted = -beta * (vals - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta


def _objective_and_grad(x, beta):
    """
    Objective: maximize soft-min of (wall distances, pair distances / 2).
    Returns (-objective, gradient) for minimization.
    x is flat array of shape (2N,).
    """
    c = x.reshape(-1, 2)
    n = c.shape[0]
    
    # Wall distances: for each point, min of (0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
    wall_dists = np.zeros(n)
    for i in range(n):
        wall_dists[i] = min(0.5 - c[i, 0], c[i, 0] + 0.5, 0.5 - c[i, 1], c[i, 1] + 0.5)
    
    # Pair distances / 2
    pair_dists = np.zeros((n, n))
    for i in range(n):
        diffs = c[i+1:] - c[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        pair_dists[i, i+1:] = dists / 2.0
        pair_dists[i+1:, i] = dists / 2.0
    
    # Collect all distances (wall + pair, excluding self-pairs)
    all_dists = np.concatenate([wall_dists, pair_dists[np.triu_indices(n, k=1)]])
    
    # Soft min
    r_soft = _soft_min_logsumexp(all_dists, beta)
    
    # Gradient computation via automatic differentiation approximation
    # We'll compute gradient numerically for reliability
    grad = np.zeros_like(x)
    eps = 1e-7
    for j in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        f_plus = -_soft_min_logsumexp(
            np.concatenate([
                np.array([min(0.5 - x_plus.reshape(-1,2)[i,0], x_plus.reshape(-1,2)[i,0]+0.5, 
                             0.5 - x_plus.reshape(-1,2)[i,1], x_plus.reshape(-1,2)[i,1]+0.5) 
                          for i in range(n)]),
                np.array([np.sqrt(np.sum((x_plus.reshape(-1,2)[i]-x_plus.reshape(-1,2)[j])**2)) / 2.0 
                          for i in range(n) for j in range(i+1, n)])
            ]), beta
        )
        f_minus = -_soft_min_logsumexp(
            np.concatenate([
                np.array([min(0.5 - x_minus.reshape(-1,2)[i,0], x_minus.reshape(-1,2)[i,0]+0.5, 
                             0.5 - x_minus.reshape(-1,2)[i,1], x_minus.reshape(-1,2)[i,1]+0.5) 
                          for i in range(n)]),
                np.array([np.sqrt(np.sum((x_minus.reshape(-1,2)[i]-x_minus.reshape(-1,2)[j])**2)) / 2.0 
                          for i in range(n) for j in range(i+1, n)])
            ]), beta
        )
        grad[j] = (f_plus - f_minus) / (2 * eps)
    
    return -r_soft, grad


def _objective_and_grad_fast(x, beta):
    """
    Faster objective and gradient using vectorized operations.
    Maximizes soft-min of all relevant distances.
    """
    c = x.reshape(-1, 2)
    n = c.shape[0]
    
    # Wall distances
    wx_pos = 0.5 - c[:, 0]  # distance to right wall
    wx_neg = c[:, 0] + 0.5  # distance to left wall
    wy_pos = 0.5 - c[:, 1]  # distance to top wall
    wy_neg = c[:, 1] + 0.5  # distance to bottom wall
    wall_dists = np.minimum(np.minimum(wx_pos, wx_neg), np.minimum(wy_pos, wy_neg))
    
    # Pair distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists[np.triu_indices(n, k=1)]
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Soft min: -logsumexp(-beta * all_dists) / beta
    m = np.min(all_dists)
    shifted = -beta * (all_dists - m)
    shifted = np.clip(shifted, -300, 300)
    log_sum = np.log(np.sum(np.exp(shifted)))
    r_soft = m - log_sum / beta
    
    # Gradient: d/dx [-r_soft] = -d r_soft / dx
    # r_soft = m - log(sum(exp(-beta*(d_k - m))))/beta
    # For each distance d_k, weight w_k = exp(-beta*(d_k - m)) / sum(exp(-beta*(d_j - m)))
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    
    # Gradient of r_soft w.r.t. each coordinate
    grad = np.zeros_like(x)
    
    # Wall contribution
    for i in range(n):
        w = weights[i]  # weight for wall distance of point i
        # d(wall_dist_i)/dx_i
        if wx_pos[i] <= wx_neg[i] and wx_pos[i] <= wy_pos[i] and wx_pos[i] <= wy_neg[i]:
            grad[2*i] = -w * (-1)  # d(0.5-x)/dx = -1, so d(r)/dx = -w*(-1) = w... wait
            # r_soft includes wall_dist, d(r_soft)/d(wall_dist) = -w (from soft min)
            # d(wall_dist)/dx = -1 (for right wall)
            # So d(r_soft)/dx = -w * (-1) = w
            # We're minimizing -r_soft, so grad = -d(r_soft)/dx = -w
            grad[2*i] = -w
            grad[2*i+1] = 0
        elif wx_neg[i] <= wx_pos[i] and wx_neg[i] <= wy_pos[i] and wx_neg[i] <= wy_neg[i]:
            grad[2*i] = w  # d(x+0.5)/dx = 1, d(r)/d(dist) = -w, so d(r)/dx = -w*1 = -w, grad = w
            grad[2*i+1] = 0
        elif wy_pos[i] <= wx_pos[i] and wy_pos[i] <= wx_neg[i] and wy_pos[i] <= wy_neg[i]:
            grad[2*i] = 0
            grad[2*i+1] = -w
        else:
            grad[2*i] = 0
            grad[2*i+1] = w
    
    # Pair contribution
    # For pair (i,j) with distance d_ij, weight w_ij
    # d(d_ij)/dx_i = (x_i - x_j) / (2 * d_ij) * 2 / 2 ... let me be careful
    # d_ij = |x_i - x_j| / 2
    # d(d_ij)/dx_i = (x_i - x_j) / (2 * |x_i - x_j|) = (x_i - x_j) / (2 * 2 * d_ij) ... 
    # Actually |x_i - x_j| = 2 * d_ij, so d(d_ij)/dx_i = (x_i - x_j) / (2 * 2 * d_ij) = (x_i-x_j)/(4*d_ij)
    # Hmm, let me redo: d_ij = 0.5 * |x_i - x_j|
    # d(d_ij)/dx_i = 0.5 * (x_i - x_j)/|x_i - x_j|
    # d(r_soft)/dx_i = sum over pairs involving i of -w_ij * d(d_ij)/dx_i
    # grad = -d(r_soft)/dx_i
    
    for i in range(n):
        for j in range(i+1, n):
            idx = i * (n - 1) - i * (i+1) // 2 + (j - i - 1)  # index in triu array
            # Actually, let's just iterate properly
            pass
    
    # Simpler: build gradient from pair weights
    pair_grad = np.zeros((n, 2))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            w = weights[n + idx]
            diff_ij = c[i] - c[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij > 1e-12:
                # d(d_ij)/dx_i = 0.5 * diff_ij / dist_ij
                # d(r_soft)/dx_i = -w * 0.5 * diff_ij / dist_ij
                # grad contribution = -d(r_soft)/dx_i = w * 0.5 * diff_ij / dist_ij
                pair_grad[i] += w * 0.5 * diff_ij / dist_ij
                pair_grad[j] -= w * 0.5 * diff_ij / dist_ij
            idx += 1
    
    grad += pair_grad.flatten()
    
    return -r_soft, grad


def _generate_hex_initial(n_rows):
    """Generate hexagonal lattice initialization."""
    n_cols = math.ceil(N / n_rows)
    # Hexagonal spacing
    # Total height = (n_rows - 1) * d * sqrt(3)/2 + d = d * ((n_rows-1)*sqrt(3)/2 + 1)
    # Total width = (n_cols - 1) * d + d = n_cols * d (with alternating offsets)
    d_h = 1.0 / ((n_rows - 1) * math.sqrt(3) / 2 + 1) if n_rows > 1 else 1.0
    d_w = 1.0 / n_cols
    d = min(d_h, d_w)
    
    pts = []
    y_start = -0.5 + d / 2
    for row in range(n_rows):
        y = y_start + row * d * math.sqrt(3) / 2
        if row % 2 == 0:
            n_in_row = n_cols
            x_start = -0.5 + d / 2
        else:
            n_in_row = n_cols - 1
            x_start = -0.5 + d * 0.75
        
        for col in range(n_in_row):
            x = x_start + col * d
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Pad with random points
        while len(pts) < N:
            x = np.random.uniform(-0.5 + d/2, 0.5 - d/2)
            y = np.random.uniform(-0.5 + d/2, 0.5 - d/2)
            pts = np.vstack([pts, [[x, y]]])
    
    return pts


def _generate_jittered_hex(n_rows, jitter=0.01):
    """Generate hexagonal lattice with random jitter."""
    pts = _generate_hex_initial(n_rows)
    pts = pts + np.random.uniform(-jitter, jitter, size=pts.shape)
    pts = np.clip(pts, -0.5, 0.5)
    return pts


def _force_directed_phase(centers, n_iters=500, learning_rate=0.001):
    """Force-directed repulsion to spread points apart."""
    centers = centers.copy()
    n = len(centers)
    
    for it in range(n_iters):
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion (inverse square, capped)
        for i in range(n):
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    dist = 1e-8
                    diff = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)])
                # Force magnitude ~ 1/dist^2, but cap it
                force_mag = 1.0 / (dist * dist + 0.01)
                force = force_mag * diff / dist
                forces[i] += force
                forces[j] -= force
        
        # Wall attraction (keep points inside)
        for i in range(n):
            for axis in range(2):
                if centers[i, axis] > 0.4:
                    forces[i, axis] -= (centers[i, axis] - 0.4) * 10
                elif centers[i, axis] < -0.4:
                    forces[i, axis] -= (centers[i, axis] + 0.4) * 10
        
        centers += learning_rate * forces
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers


def _projected_gradient_ascent(x, beta_schedule, n_iters, lr=0.005, momentum=0.9):
    """
    Projected gradient ascent on the soft-min objective with adaptive beta.
    x is flat array of shape (2N,).
    """
    n = N
    v = np.zeros_like(x)  # momentum
    
    for it in range(n_iters):
        # Adaptive beta: start low (smooth), increase (sharpen)
        progress = it / max(n_iters - 1, 1)
        beta = beta_schedule(progress)
        
        obj, grad = _objective_and_grad_fast(x, beta)
        
        # Gradient ascent: x += lr * (-grad) since grad is for minimization
        v = momentum * v - lr * grad
        x_new = x + v
        
        # Project to square
        x_new = np.clip(x_new, -0.5, 0.5)
        
        # Check if improved (in terms of actual radius, not soft-min)
        c_new = x_new.reshape(-1, 2)
        r_new = _compute_radius(c_new)
        c_old = x.reshape(-1, 2)
        r_old = _compute_radius(c_old)
        
        if r_new >= r_old or np.random.random() < 0.1:
            x = x_new
        else:
            # Reject, reduce momentum
            v *= 0.5
    
    return x


def _polish_with_scipy(centers, max_iter=100):
    """Polish using scipy's L-BFGS-B on the negative radius objective."""
    x0 = centers.flatten()
    
    def neg_radius(x):
        c = x.reshape(-1, 2)
        r = _compute_radius(c)
        return -r
    
    try:
        result = minimize(neg_radius, x0, method='L-BFGS-B',
                          bounds=[(-0.5, 0.5)] * (2 * N),
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6})
        new_centers = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
        return new_centers
    except Exception:
        return centers


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    np.random.seed(42)
    
    best_centers = None
    best_radius = 0.0
    
    # Multi-start with different initializations
    initializations = []
    
    # Hexagonal lattices with different row counts
    for n_rows in [12, 13, 14, 15, 16]:
        if time.time() > deadline - 80:
            break
        initializations.append(_generate_hex_initial(n_rows))
        initializations.append(_generate_jittered_hex(n_rows, jitter=0.02))
    
    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        initializations.append(grid[:N])
    else:
        # Pad
        while len(grid) < N:
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.5, 0.5)
            grid = np.vstack([grid, [[x, y]]])
        initializations.append(grid)
    
    # Random initialization
    random_init = np.random.uniform(-0.45, 0.45, size=(N, 2))
    initializations.append(random_init)
    
    for init in initializations:
        if time.time() > deadline - 30:
            break
        
        # Phase 1: Force-directed spreading
        if time.time() < deadline - 40:
            centers_fd = _force_directed_phase(init, n_iters=200, learning_rate=0.002)
        else:
            centers_fd = init.copy()
        
        # Phase 2: Projected gradient ascent with adaptive beta
        if time.time() < deadline - 15:
            x_flat = centers_fd.flatten()
            # Beta schedule: start at 10, end at 100
            beta_schedule = lambda p: 10 + 90 * p
            x_opt = _projected_gradient_ascent(x_flat, beta_schedule, n_iters=300, lr=0.003, momentum=0.8)
            centers_opt = x_opt.reshape(-1, 2)
        else:
            centers_opt = centers_fd
        
        # Phase 3: Scipy polish
        if time.time() < deadline - 5:
            centers_polished = _polish_with_scipy(centers_opt, max_iter=50)
        else:
            centers_polished = centers_opt
        
        r = _compute_radius(centers_polished)
        if r > best_radius:
            best_radius = r
            best_centers = centers_polished.copy()
    
    # Fallback
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
