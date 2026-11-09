import time
import numpy as np
from scipy.optimize import minimize

N = 200
HALF = 0.5
SEED = 1234567

def _radius_of(c):
    """Compute packing radius for centers c (N,2)."""
    w = HALF - np.abs(c)
    wall = np.min(w)
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pair = np.sqrt(np.min(dist_sq)) / 2.0
    return float(min(wall, pair))

def _hexagon_seed(n):
    """Staggered hexagonal rows inside [-1/2,1/2]^2, return n centers."""
    # Try to fit a hexagonal packing
    # Target radius ~ 0.036
    # We will generate a grid and clip
    best_pts = None
    best_count = 0
    
    # Try a range of starting radii/spacings
    for r0 in np.linspace(0.030, 0.040, 100):
        h = 2 * r0
        v = r0 * np.sqrt(3)
        x0 = -HALF + r0
        y0 = -HALF + r0
        
        n_rows = int((1 - 2 * r0) / v) + 1
        pts = []
        for i in range(n_rows):
            y = y0 + i * v
            # Alternate offset
            offset = h / 2 if i % 2 else 0.0
            x_start = x0 + offset
            # Calculate how many columns fit
            # x ranges from -HALF + r0 to HALF - r0
            # Width = 1 - 2*r0
            # Number of steps = (Width - offset) / h + 1 ?
            # Let's just generate and filter
            n_cols = int((HALF - r0 - x_start) / h) + 1
            if n_cols < 1:
                continue
            for j in range(n_cols):
                x = x_start + j * h
                if x <= HALF - r0:
                    pts.append([x, y])
        
        if len(pts) > best_count:
            best_count = len(pts)
            best_pts = pts
            
    if best_pts is None or len(best_pts) < n:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
        
    best_pts = np.array(best_pts)
    if len(best_pts) > n:
        # Shuffle and take first n? Or just take first n?
        # Taking first n might bias towards one corner.
        # Let's center the selection?
        # For now, just take first n. The optimizer will fix it.
        return best_pts[:n]
    else:
        # Pad with random points if needed?
        # Or just return what we have? No, must return N.
        # If we don't have enough, we can duplicate and let optimizer separate?
        # Better: if we don't have enough, use a denser grid.
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]

def _soft_min_sq_radius(x, beta):
    """Smooth surrogate for min(wall^2, min_pair_dist_sq/4) using log-sum-exp."""
    c = x.reshape(-1, 2)
    # Wall clearances squared
    w = HALF - np.abs(c)
    wall_vals = w**2
    
    # Pairwise distances squared / 4
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pair_vals = dist_sq / 4.0
    
    all_margins = np.concatenate([wall_vals.ravel(), pair_vals.ravel()])
    
    # Soft min: -beta * logsumexp(-margins/beta)
    neg = -all_margins / beta
    max_neg = np.max(neg)
    # Handle case where max_neg is -inf (shouldn't happen if beta>0 and margins finite)
    if np.isinf(max_neg):
        return 0.0
    lse = max_neg + np.log(np.sum(np.exp(neg - max_neg)))
    return -lse * beta

def _soft_min_sq_grad(x, beta):
    """Gradient of -soft_min_sq_radius."""
    c = x.reshape(-1, 2)
    w = HALF - np.abs(c)
    wall_vals = w**2
    
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pair_vals = dist_sq / 4.0
    
    all_margins = np.concatenate([wall_vals.ravel(), pair_vals.ravel()])
    
    neg = -all_margins / beta
    max_neg = np.max(neg)
    if np.isinf(max_neg):
        return np.zeros_like(x)
    
    exps = np.exp(neg - max_neg)
    weights = exps / np.sum(exps)
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    # margin = (HALF - |c_ik|)^2
    # d(margin)/dc_ik = 2(HALF - |c_ik|) * d(HALF - |c_ik|)/dc_ik
    # d(HALF - |c_ik|)/dc_ik = -sign(c_ik)
    # So d(margin)/dc_ik = -2 * (HALF - |c_ik|) * sign(c_ik)
    # We are minimizing -soft_min, so we need -d(soft_min)/dx
    # d(soft_min)/dx = sum w_k d(margin_k)/dx
    # grad = - sum w_k d(margin_k)/dx
    
    wall_w = weights[:len(wall_vals)]
    for i in range(N):
        for k in range(2):
            idx = i * 2 + k
            cw = wall_w[idx]
            c_ik = c[i, k]
            if c_ik >= 0:
                # sign = 1, d(margin)/dc = -2 * w
                grad[idx] -= cw * (-2 * w[i, k]) # -cw * (-2w) = 2cw*w
            else:
                # sign = -1, d(margin)/dc = -2 * w * (-1) = 2w
                grad[idx] -= cw * (2 * w[i, k])  # -cw * (2w) = -2cw*w
                
    # Pair gradients
    # margin = dist_sq / 4 = |c_i - c_j|^2 / 4
    # d(margin)/dc_i = 2(c_i - c_j)/4 = (c_i - c_j)/2
    # d(margin)/dc_j = 2(c_j - c_i)/4 = (c_j - c_i)/2
    # grad contribution = -w_ij * d(margin)/dc_i
    
    pair_w = weights[len(wall_vals):].reshape(N, N)
    
    # We need:
    # grad_i = - sum_j pair_w[i,j] * (c_i - c_j)/2
    # grad_j = - sum_i pair_w[j,i] * (c_j - c_i)/2
    
    # Note pair_w is symmetric if margins are symmetric?
    # dist_sq is symmetric. So pair_vals is symmetric.
    # weights are symmetric.
    
    # grad_i = -0.5 * ( c_i * sum_j pair_w[i,j] - sum_j pair_w[i,j] * c_j )
    
    wsum = pair_w.sum(axis=1) # shape (N,)
    sum_wc = pair_w @ c # shape (N,2)
    
    grad_pair = -0.5 * (wsum[:, None] * c - sum_wc)
    
    grad[0::2] += grad_pair[:, 0]
    grad[1::2] += grad_pair[:, 1]
    
    return grad

def _optimize_lbgfs(c0, beta, deadline):
    """Run L-BFGS-B to maximize soft-min squared radius."""
    x0 = c0.ravel().copy()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    
    def obj(x):
        return -_soft_min_sq_radius(x, beta)
    
    def grad(x):
        return _soft_min_sq_grad(x, beta)
    
    t0 = time.time()
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxls': 50})
        if res.success or res.fun < obj(x0):
            best_x = res.x
        else:
            best_x = x0
    except Exception:
        best_x = x0
    
    best_c = best_x.reshape(N, 2)
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    r = _radius_of(best_c)
    return best_c, r

def construct():
    t_start = time.time()
    deadline = 115.0
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)

    restarts = 0
    while time.time() - t_start < deadline:
        restarts += 1
        
        # Beta schedule: Start large, decrease.
        # 0.05 is quite smooth. 0.005 is sharp.
        beta = max(0.002, 0.05 * (0.9 ** restarts))
        
        # Perturbation scale: Start moderate, decrease.
        scale = 0.02 / (1.0 + 0.1 * restarts)
        
        c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
        c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)
        
        # Check for coincident centers
        d = c_start[:, None, :] - c_start[None, :, :]
        dist_sq = np.sum(d * d, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        if np.min(dist_sq) < 1e-6:
            c_start = best_c.copy()
            
        rem = deadline - (time.time() - t_start)
        if rem < 5.0:
            break
            
        c_opt, r_opt = _optimize_lbgfs(c_start, beta, rem - 2.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt

    # Final polish
    rem = deadline - (time.time() - t_start)
    if rem > 10.0:
        c_polish, r_polish = _optimize_lbgfs(best_c, 0.001, rem - 2.0)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish

    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
        
    return best_c

if __name__ == "__main__":
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))
