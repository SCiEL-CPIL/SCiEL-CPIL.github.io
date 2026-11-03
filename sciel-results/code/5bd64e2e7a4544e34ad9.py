import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 118.0

def _radius(c):
    d = np.sqrt(((c[:,None,:]-c[None,:,:])**2).sum(-1))
    np.fill_diagonal(d, np.inf)
    rp = d.min()/2
    rw = np.minimum(np.minimum(0.5-c[:,0], c[:,0]+0.5),
                    np.minimum(0.5-c[:,1], c[:,1]+0.5)).min()
    return min(rp, rw)

def _relax(c, R, iters, eta0, eta1):
    # Kept for compatibility or fallback, but likely unused in new path
    c = c.copy()
    n = len(c)
    for it in range(iters):
        t = it/iters
        eta = eta0 + (eta1-eta0)*t
        diff = c[:,None,:]-c[None,:,:]
        d = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2*R - d, 0.0)/d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:,:,None]*diff).sum(1)
        f[:,0] += -np.maximum(c[:,0]-(0.5-R),0.0) + np.maximum((-0.5+R)-c[:,0],0.0)
        f[:,1] += -np.maximum(c[:,1]-(0.5-R),0.0) + np.maximum((-0.5+R)-c[:,1],0.0)
        c += eta*f
        c = np.clip(c, -0.5, 0.5)
    return c

def _grow(start, R0, R1, step, iters, eta0, eta1, t0):
    # Deprecated, replaced by _optimize_penalty
    return (start.copy(), _radius(start))

def _init_7x7_plus():
    ticks = (np.arange(7)+0.5)/7 - 0.5
    g = np.array([(x,y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])

def _init_jitter8(rng):
    ticks = (np.arange(8)+0.5)/8 - 0.5
    g = np.array([(x,y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03,0.03,c.shape)
    return c

def _init_rand(rng):
    return rng.uniform(-0.45,0.45,(N,2))

def _optimize_penalty(c_init, t0):
    """
    Optimizes the packing using a penalty method with L-BFGS-B.
    Variables: x (N*2) and r (1).
    Objective: -r + lambda * sum(max(0, 2r - d_ij)^2) + lambda * sum(max(0, r - w_ik)^2)
    """
    x0 = c_init.flatten()
    r0 = _radius(c_init)
    # Start r slightly lower to allow growth? Or start at current best.
    # Let's start at current best radius.
    r_start = r0
    
    # Bounds for x: [-0.5, 0.5]
    # Bounds for r: [0, 0.5]
    bounds_x = [(-0.5, 0.5)] * (N * 2)
    bounds_r = [(0.0, 0.5)]
    bounds = bounds_x + bounds_r
    
    def objective(z):
        if time.time() - t0 > DEADLINE:
            return 1e10
        x = z[:N*2].reshape(N, 2)
        r = z[N*2]
        
        # Compute distances
        diff = x[:,None,:] - x[None,:,:]
        d = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(d, 1e12)
        
        # Pairwise violations: max(0, 2r - d_ij)
        # We only care about i < j to avoid double counting, but for gradient symmetry it's easier to sum all i!=j and divide by 2?
        # Actually, let's just sum all i<j.
        # Mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_pairs = d[mask]
        viol_pairs = np.maximum(0, 2*r - d_pairs)
        
        # Wall violations: max(0, r - w_ik)
        # w_ik is min(0.5 - x, x + 0.5)
        w = np.minimum(np.minimum(0.5 - x[:,0], x[:,0] + 0.5),
                       np.minimum(0.5 - x[:,1], x[:,1] + 0.5))
        viol_walls = np.maximum(0, r - w)
        
        lam = 100.0 # Fixed lambda for now, or adaptively increase?
        # Let's use a fixed high lambda to enforce constraints strictly.
        # If lambda is too high, the problem becomes stiff.
        # Let's try lambda = 500.
        
        penalty = lam * (np.sum(viol_pairs**2) + np.sum(viol_walls**2))
        return -r + penalty

    def gradient(z):
        if time.time() - t0 > DEADLINE:
            return np.zeros_like(z)
        x = z[:N*2].reshape(N, 2)
        r = z[N*2]
        
        diff = x[:,None,:] - x[None,:,:]
        d = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(d, 1e12)
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_pairs = d[mask]
        viol_pairs = np.maximum(0, 2*r - d_pairs)
        
        w = np.minimum(np.minimum(0.5 - x[:,0], x[:,0] + 0.5),
                       np.minimum(0.5 - x[:,1], x[:,1] + 0.5))
        viol_walls = np.maximum(0, r - w)
        
        lam = 500.0
        
        # Gradient w.r.t r
        # d(-r)/dr = -1
        # d(penalty)/dr = lam * (2 * sum(viol_pairs * 2) + 2 * sum(viol_walls * 1))
        grad_r = -1 + lam * (4 * np.sum(viol_pairs) + 2 * np.sum(viol_walls))
        
        # Gradient w.r.t x
        # d(penalty)/dx_i = lam * sum_{j != i} d/dx_i (max(0, 2r - d_ij)^2)
        # + lam * sum_k d/dx_i (max(0, r - w_ik)^2)
        
        # Pairwise part
        # For each pair (i,j) with i<j, if viol > 0:
        # d/dx_i (2r - d_ij)^2 = 2(2r - d_ij) * (-d_ij/dx_i) = -2(2r - d_ij) * (x_i - x_j)/d_ij
        # d/dx_j (2r - d_ij)^2 = 2(2r - d_ij) * (d_ij/dx_j) = 2(2r - d_ij) * (x_j - x_i)/d_ij = -2(2r - d_ij) * (x_i - x_j)/d_ij
        # Wait, d_ij = ||x_i - x_j||. d/dx_i d_ij = (x_i - x_j)/d_ij.
        # So d/dx_i (2r - d_ij) = -(x_i - x_j)/d_ij.
        # d/dx_i (2r - d_ij)^2 = 2(2r - d_ij) * (-(x_i - x_j)/d_ij).
        
        # We can vectorize this.
        # Create a matrix of forces F where F_ij is the force on i from j.
        # F_ij = -2 * viol_ij * (x_i - x_j) / d_ij
        # Note: viol_ij = max(0, 2r - d_ij).
        
        # Let's construct the force matrix.
        # diff = x_i - x_j
        # d = dist
        # viol_matrix = max(0, 2r - d)
        # Force on i from j: -2 * viol_ij * diff_ij / d_ij
        # Total force on i = sum_j Force_ij
        
        # Be careful with d=0 (should not happen with distinct centers, but safe to check)
        d_safe = np.where(d < 1e-10, 1e-10, d)
        
        # viol_matrix is N x N
        viol_matrix = np.maximum(0, 2*r - d)
        np.fill_diagonal(viol_matrix, 0)
        
        # factor = 2 * viol_matrix / d_safe
        # force_matrix = -factor[:,:,None] * diff
        # grad_x = lam * sum_j force_matrix[i,j]
        
        factor = 2 * viol_matrix / d_safe
        force_matrix = -factor[:, :, None] * diff
        grad_x_pairs = lam * np.sum(force_matrix, axis=1)
        
        # Wall part
        # w_ik = min(...)
        # This is non-smooth at the boundaries of the min.
        # We can approximate by treating the active constraint.
        # Or just use numerical gradient for walls? No, let's try analytical.
        # If x_ik is the min, then d(w_ik)/dx_ik = +/- 1.
        # Let's determine which wall is active.
        
        # w_ik is min of 4 distances.
        # Let's compute gradients for each wall distance and select the active one.
        # dist_left = x_ik + 0.5, grad = 1
        # dist_right = 0.5 - x_ik, grad = -1
        # dist_bottom = x_ik + 0.5, grad = 1
        # dist_top = 0.5 - x_ik, grad = -1
        
        # Actually, w_ik = min(0.5 - x_ik, x_ik + 0.5) for each dimension.
        # Let w_x = min(0.5 - x_i0, x_i0 + 0.5)
        # Let w_y = min(0.5 - x_i1, x_i1 + 0.5)
        # w = min(w_x, w_y)
        
        # If w_x < w_y, w = w_x.
        #   If 0.5 - x_i0 < x_i0 + 0.5 (i.e., x_i0 > 0), w_x = 0.5 - x_i0, grad = -1.
        #   Else w_x = x_i0 + 0.5, grad = 1.
        # Similarly for w_y.
        
        # Let's compute grad_w for each point.
        grad_w = np.zeros((N, 2))
        
        # For x dimension
        active_x = (0.5 - x[:,0] <= x[:,0] + 0.5) # x >= 0
        grad_w_x = np.where(active_x, -1.0, 1.0)
        
        # For y dimension
        active_y = (0.5 - x[:,1] <= x[:,1] + 0.5) # y >= 0
        grad_w_y = np.where(active_y, -1.0, 1.0)
        
        w_x = np.minimum(0.5 - x[:,0], x[:,0] + 0.5)
        w_y = np.minimum(0.5 - x[:,1], x[:,1] + 0.5)
        
        # w = min(w_x, w_y)
        # If w_x < w_y, active is x.
        # If w_y < w_x, active is y.
        # If equal, take average? Or just one?
        
        active_dim_x = (w_x <= w_y)
        
        # grad_w[:, 0] is non-zero only if x is active
        grad_w[:, 0] = np.where(active_dim_x, grad_w_x, 0.0)
        # grad_w[:, 1] is non-zero only if y is active
        grad_w[:, 1] = np.where(~active_dim_x, grad_w_y, 0.0)
        
        # Penalty term: sum(max(0, r - w)^2)
        # d/dx_i = 2 * viol_wall_i * (-grad_w_i)
        # grad_x_walls = lam * 2 * viol_walls[:, None] * (-grad_w)
        
        grad_x_walls = lam * 2 * viol_walls[:, None] * (-grad_w)
        
        grad_x = grad_x_pairs + grad_x_walls
        
        return np.concatenate([grad_x.flatten(), [grad_r]])

    z0 = np.concatenate([x0, [r_start]])
    
    best_z = z0.copy()
    best_val = objective(z0)
    
    # Run L-BFGS-B
    try:
        res = minimize(objective, z0, jac=gradient, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
        if res.fun < best_val:
            best_z = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Extract best solution
    x_best = best_z[:N*2].reshape(N, 2)
    r_best = best_z[N*2]
    
    # Clip x to be strictly inside
    x_best = np.clip(x_best, -0.5, 0.5)
    
    # Calculate true radius
    r_true = _radius(x_best)
    
    # If r_true is better, use it.
    # Note: The penalty method maximizes r subject to constraints.
    # The resulting r_best should be close to r_true.
    
    return x_best, r_true

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    
    inits = [
        _init_7x7_plus(),
        _init_jitter8(rng),
        _init_rand(rng),
        _init_jitter8(rng),
        _init_rand(rng)
    ]
    
    best_c = None
    best_r = 0.0
    
    for start in inits:
        if (time.time() - t0) > DEADLINE - 10:
            break
            
        c, r = _optimize_penalty(start, t0)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
    if best_c is None:
        best_c = _init_7x7_plus()
        best_r = _radius(best_c)
        
    # Final polish: Run one more optimization from the best found
    if (time.time() - t0) < DEADLINE - 5:
        c, r = _optimize_penalty(best_c, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
    return best_c
