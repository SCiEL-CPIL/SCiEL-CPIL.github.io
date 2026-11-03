"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def _radius(c):
    """Compute the maximum common radius for centers c of shape (N,2)."""
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(axis=1))
    rp = d.min() / 2.0
    rw = np.minimum(np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
                    np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)).min()
    return min(rp, rw)

def _soft_min(x, beta):
    """Differentiable approximation of min(x)."""
    # Log-sum-exp trick for stability
    m = np.min(x)
    if m < -100:
        return m
    s = np.sum(np.exp(beta * (x - m)))
    return m - np.log(s) / beta

def _soft_radius(c, beta):
    """Differentiable approximation of the maximum radius."""
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d_sq = (diff ** 2).sum(axis=1)
    # d = sqrt(d_sq), but we can work with d_sq? No, radius is min(d/2, w).
    # Let's compute d and w.
    d = np.sqrt(d_sq)
    pair_vals = d / 2.0
    
    # Wall distances
    x, y = c[:, 0], c[:, 1]
    w = np.minimum(np.minimum(0.5 - x, x + 0.5),
                   np.minimum(0.5 - y, y + 0.5))
    
    # Combine all constraints
    vals = np.concatenate([pair_vals, w])
    return _soft_min(vals, beta)

def _init_hexagonal():
    """Generate a hexagonal lattice initialization."""
    pts = []
    s = 0.142 # Slightly larger spacing to fit 50
    # We want ~50 points. 7x7=49. 8x8=64.
    # Let's try a grid that yields close to 50.
    # Or just use a 7x7 grid and jitter.
    # Actually, a 7x7 grid has 49 points. We need 50.
    # Let's use a 7x7 grid and add one point in a gap? Or just use 8x8 and drop one?
    # Better: Use a grid of 7x7 and add a 50th point optimally?
    # For simplicity and robustness, let's generate a 7x7 grid, and if we need 50,
    # we can split one cell or just use 8x8 and drop the worst one.
    # Let's stick to a robust 7x7 + 1 strategy or just 8x8 drop 1.
    # 8x8 grid:
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    # 64 points. We need 50.
    # Drop the 14 points farthest from center? Or drop random?
    # Dropping farthest from center might hurt boundary packing.
    # Let's try to keep a hex-like shape.
    # Or just use 7x7=49 and add one in the center? Center is occupied.
    # Let's use a 7x7 grid and duplicate the center with a tiny offset? No, that's bad.
    # Let's use a 7x7 grid and insert a 50th point in the largest empty space?
    # Simpler: Use a 7x7 grid, and add a 50th point at (0, 0.5 - s/2) if space allows?
    # Actually, the best known packing is not a simple grid.
    # Let's initialize with a 7x7 grid and one extra point placed randomly near the center,
    # then let the optimizer fix it.
    ticks7 = (np.arange(7) + 0.5) / 7 - 0.5
    g7 = np.array([(x, y) for y in ticks7 for x in ticks7])
    # Add a 50th point. Let's place it at (0,0) but shifted slightly?
    # (0,0) is already in g7 (index 24).
    # Let's place the 50th point at (0, 0.1) or similar, and let optimizer move it.
    # Or better: Start with 50 points on a slightly denser grid.
    # Let's use a 8x8 grid and drop the 4 corners? No, corners are good for walls.
    # Let's use a 7x7 grid and add a point at (0.5, 0) - wait, (0.5,0) is wall.
    # Let's just use a 7x7 grid and add a random point inside.
    c = np.vstack([g7, np.array([0.1, 0.1])])
    return c

def _delaunay_refine(c, beta, deadline, t0):
    """
    Refine using L-BFGS-B on the soft-min objective.
    We optimize -soft_radius(c).
    """
    n = c.shape[0]
    
    def objective(x):
        c_x = x.reshape(n, 2)
        # Clip to box to keep values reasonable during optimization?
        # L-BFGS-B has bounds, so we don't need to clip inside objective if bounds are set.
        # But soft_min is sensitive to large values.
        return -_soft_radius(c_x, beta)

    def gradient(x):
        c_x = x.reshape(n, 2)
        # Compute gradient of -soft_radius
        # soft_min(x) = m - log(sum(exp(beta*(x-m))))/beta
        # d(soft_min)/dx_i = (exp(beta*(x_i-m))/sum(exp(beta*(x-m))))
        # Let w_i = exp(beta*(x_i-m)) / sum(exp(beta*(x-m)))
        # grad_soft_min = w
        
        iu = np.triu_indices(n, k=1)
        diff = c_x[iu[0]] - c_x[iu[1]]
        d_sq = (diff ** 2).sum(axis=1)
        d = np.sqrt(d_sq)
        pair_vals = d / 2.0
        
        x, y = c_x[:, 0], c_x[:, 1]
        w = np.minimum(np.minimum(0.5 - x, x + 0.5),
                       np.minimum(0.5 - y, y + 0.5))
        
        vals = np.concatenate([pair_vals, w])
        m = np.min(vals)
        if m < -100:
            # Degenerate case
            return np.zeros_like(x)
        
        exp_vals = np.exp(beta * (vals - m))
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        grad = np.zeros_like(x)
        
        # Pair gradients
        # d(pair_val)/dc_i = 0.5 * (c_i - c_j) / d
        # d(pair_val)/dc_j = 0.5 * (c_j - c_i) / d = -0.5 * (c_i - c_j) / d
        # So d(soft_min)/dc = sum_k w_k * d(val_k)/dc
        
        # Initialize pair gradients
        # For each pair (i,j) with weight w_k:
        # grad_i += w_k * 0.5 * (c_i - c_j) / d_k
        # grad_j += w_k * 0.5 * (c_j - c_i) / d_k
        
        # Vectorized approach is tricky with triu indices.
        # Let's loop or use np.add.at
        # Number of pairs is ~1225. Looping is fine for N=50.
        
        for k in range(len(iu[0])):
            i, j = iu[0][k], iu[1][k]
            w_k = weights[k]
            if d[k] < 1e-10:
                continue
            vec = 0.5 * (c_x[i] - c_x[j]) / d[k]
            grad[2*i:2*i+2] += w_k * vec
            grad[2*j:2*j+2] -= w_k * vec
            
        # Wall gradients
        # w_i = min(0.5-x, x+0.5, 0.5-y, y+0.5)
        # This is non-smooth. The gradient is 0 except at the minimum.
        # At the minimum, if it's 0.5-x, grad_x = -1. If x+0.5, grad_x = 1.
        # If 0.5-y, grad_y = -1. If y+0.5, grad_y = 1.
        # If multiple are equal, we can average or pick one.
        # For soft-min, the weight w_wall is distributed.
        # But the wall constraint itself is a hard min.
        # The soft_min is over the set of values {d_ij/2, w_i}.
        # So the gradient of the wall part comes from the weight assigned to w_i in the soft_min.
        # d(w_i)/dc_i is the gradient of the wall distance function.
        
        for i in range(n):
            w_k = weights[len(iu[0]) + i]
            if w_k < 1e-10:
                continue
            cx, cy = c_x[i, 0], c_x[i, 1]
            dx1 = 0.5 - cx
            dx2 = cx + 0.5
            dy1 = 0.5 - cy
            dy2 = cy + 0.5
            
            # Determine which wall is active
            min_val = min(dx1, dx2, dy1, dy2)
            
            # If there's a tie, we should split the gradient.
            # For simplicity, let's just pick the first minimum.
            # Or better, compute the subgradient average.
            # Let's just use the one with the minimum value. If tie, pick arbitrary.
            # Actually, since it's a min, the gradient is a convex combination of the active ones.
            # If we just pick one, it might be noisy.
            # Let's check how many are close to min.
            active = []
            for val, g_x, g_y in [(dx1, -1, 0), (dx2, 1, 0), (dy1, 0, -1), (dy2, 0, 1)]:
                if abs(val - min_val) < 1e-8:
                    active.append((g_x, g_y))
            
            if active:
                avg_gx = sum(g[0] for g in active) / len(active)
                avg_gy = sum(g[1] for g in active) / len(active)
                grad[2*i] += w_k * avg_gx
                grad[2*i+1] += w_k * avg_gy

        return -grad # We minimize -soft_radius, so negate the gradient of soft_radius

    bounds = [(-0.5, 0.5)] * (2 * n)
    
    x0 = c.flatten()
    
    # Adaptive beta
    beta = 100.0
    
    # Run L-BFGS-B
    best_c = c.copy()
    best_r = _radius(c)
    
    start_time = time.time()
    max_iter = 200
    
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.x is not None:
            c_new = res.x.reshape(n, 2)
            r_new = _radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
    except Exception:
        pass
    
    # If time remains, increase beta and refine again
    if time.time() - t0 < deadline - 10:
        beta = 500.0
        x0 = best_c.flatten()
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.x is not None:
                c_new = res.x.reshape(n, 2)
                r_new = _radius(c_new)
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
        except Exception:
            pass
            
    return best_c

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Initialize
    c = _init_hexagonal()
    
    # Refine with L-BFGS-B on soft-min
    c = _delaunay_refine(c, 100.0, deadline, t0)
    
    # Ensure validity
    c = np.clip(c, -0.5 + 1e-15, 0.5 - 1e-15)
    
    if not np.all(np.isfinite(c)) or _radius(c) <= 0:
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return c
# EVOLVE-BLOCK-END
