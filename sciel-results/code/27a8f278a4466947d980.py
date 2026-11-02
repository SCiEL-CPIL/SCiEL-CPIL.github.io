import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
        return 0.0
        
    dist_to_wall = 0.5 - np.abs(centers)
    r_wall = np.min(dist_to_wall)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    return min(r_wall, r_pair)

def _soft_min_obj_and_grad(centers, alpha):
    """
    Compute the soft-min objective and its gradient.
    Objective: -soft_min(constraints)
    Constraints: wall distances and half-pair-distances.
    """
    centers = centers.reshape(N, 3)
    
    # Wall constraints: 0.5 - |c_ik|
    abs_c = np.abs(centers)
    wall_vals = 0.5 - abs_c
    # Gradient of wall_vals w.r.t centers: -sign(centers)
    wall_grads = -np.sign(centers)
    
    # Pair constraints: 0.5 * ||c_i - c_j||
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Create a flat array of all constraint values
    # We need to be careful with the diagonal for pairs, but we set inf there.
    # For soft-min, inf values contribute 0 to the sum if alpha is positive? 
    # e^{-alpha * inf} = 0. So they are naturally excluded.
    
    # Flatten values
    all_vals = np.concatenate([wall_vals.flatten(), pair_vals[~np.eye(N, dtype=bool)].flatten()])
    
    # To compute gradient, we need the weights for each constraint
    # w_k = exp(-alpha * v_k) / sum(exp(-alpha * v))
    
    # Numerical stability: subtract max
    max_val = np.max(all_vals)
    exp_vals = np.exp(-alpha * (all_vals - max_val))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Soft-min value: - (1/alpha) * (log(sum_exp) - alpha * max_val) 
    # Actually: soft_min = -1/alpha * log(sum(exp(-alpha * v)))
    # = -1/alpha * (log(sum(exp(-alpha * (v - max)))) - alpha * max)
    # = -1/alpha * log(sum_exp) + max_val
    
    soft_min_val = max_val - (1.0 / alpha) * np.log(sum_exp)
    obj = -soft_min_val
    
    # Gradient calculation
    # grad_obj = - grad(soft_min)
    # grad(soft_min) = - sum(w_k * grad(v_k))
    # So grad_obj = sum(w_k * grad(v_k))
    
    grad_centers = np.zeros((N, 3))
    
    # Wall part
    # wall_vals is (N, 3), weights for walls are the first N*3 of 'weights'
    wall_weights = weights[:N*3].reshape(N, 3)
    grad_centers += wall_weights * wall_grads
    
    # Pair part
    # pair_vals is (N, N). We need to map weights back to pairs.
    # The order of flattening pair_vals[~np.eye(N, dtype=bool)] is row-major.
    # We need to accumulate gradients for each center.
    # For pair (i, j), val = 0.5 * dist.
    # grad_c_i = (c_i - c_j) / dist
    # grad_c_j = (c_j - c_i) / dist = -grad_c_i
    
    # It is more efficient to loop or use vectorized accumulation if possible.
    # Given N=50, N^2=2500, a simple loop might be okay, but vectorization is better.
    
    # Let's reconstruct the weight matrix for pairs
    pair_weights_flat = weights[N*3:]
    pair_weights_matrix = np.zeros((N, N))
    # Fill the matrix. The mask ~eye is upper+lower triangle.
    # We need to map flat index to (i, j).
    # This is tricky to do vectorized without generating indices.
    # Let's use indices.
    row_idx, col_idx = np.where(~np.eye(N, dtype=bool))
    pair_weights_matrix[row_idx, col_idx] = pair_weights_flat
    
    # Now compute pair gradients
    # dists is (N, N)
    # diff is (N, N, 3)
    # grad_pair_ij = diff[i, j] / dists[i, j]
    # Note: dists[i, j] is the full distance. val = dist/2.
    # d(val)/d(c_i) = (c_i - c_j) / dist
    # So we need diff / dists
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pair_grads = diff / dists[..., np.newaxis]
        
    # Set diagonal to 0 to avoid NaNs from inf/inf
    np.fill_diagonal(pair_grads, 0.0)
    
    # Accumulate gradients
    # For each i, sum over j: w_ij * grad_c_i
    # grad_centers[i] += sum_j (pair_weights_matrix[i, j] * pair_grads[i, j])
    
    # Vectorized sum:
    # (N, N, 3) * (N, N, 1) -> (N, N, 3) -> sum over axis 1 -> (N, 3)
    pair_grad_contrib = pair_weights_matrix[..., np.newaxis] * pair_grads
    grad_centers += np.sum(pair_grad_contrib, axis=1)
    
    return obj, grad_centers

def _init_fcc():
    """FCC lattice initialization."""
    pts = []
    # FCC unit cell has 4 points.
    # We can generate a 3x3x3 grid of FCC cells?
    # Or just take points from a larger FCC grid and filter.
    
    # FCC points in unit cell [0,1]^3:
    # (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Lattice constant a.
    # We want to fit 50 spheres in [-0.5, 0.5]^3.
    # Volume of cube = 1. Volume of 50 spheres = 50 * 4/3 * pi * r^3.
    # If r ~ 0.136, V ~ 50 * 1.33 * 0.0025 ~ 0.16. Packing fraction ~ 16%.
    # FCC packing fraction is ~ 0.74.
    # So we have plenty of space.
    
    # Let's just generate a grid of FCC points and pick the best N.
    # Scale FCC so that it fits.
    # Let's try a 4x4x4 grid of FCC cells?
    # That would be 4*4*4*4 = 256 points.
    
    a = 1.0 / 4.0 # Lattice parameter
    pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                base = np.array([i*a, j*a, k*a])
                pts.append(base + np.array([0,0,0]))
                pts.append(base + np.array([a/2, a/2, 0]))
                pts.append(base + np.array([a/2, 0, a/2]))
                pts.append(base + np.array([0, a/2, a/2]))
                
    pts = np.array(pts)
    # Center at 0
    pts -= 0.5
    # Filter points inside [-0.5, 0.5]
    mask = np.all(np.abs(pts) <= 0.5, axis=1)
    pts = pts[mask]
    
    # We need exactly N points.
    # If we have more, pick the ones closest to the center? Or random?
    # Let's pick the first N.
    if len(pts) >= N:
        return pts[:N].copy()
    else:
        # Pad with random points
        rng = np.random.RandomState(42)
        extra = rng.uniform(-0.4, 0.4, size=(N - len(pts), 3))
        return np.vstack([pts, extra])

def _init_bcc():
    """BCC lattice initialization."""
    pts = []
    a = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                base = np.array([i*a, j*a, k*a])
                pts.append(base + np.array([0,0,0]))
                pts.append(base + np.array([a/2, a/2, a/2]))
                
    pts = np.array(pts)
    pts -= 0.5
    mask = np.all(np.abs(pts) <= 0.5, axis=1)
    pts = pts[mask]
    
    if len(pts) >= N:
        return pts[:N].copy()
    else:
        rng = np.random.RandomState(42)
        extra = rng.uniform(-0.4, 0.4, size=(N - len(pts), 3))
        return np.vstack([pts, extra])

def _init_random(seed=0):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _repulsion_anneal(centers, time_budget=10.0):
    """
    Simple repulsion-based relaxation.
    """
    start = time.time()
    best = centers.copy()
    best_r = _compute_radius(centers)
    current = centers.copy()
    
    step = 0.05
    for it in range(200):
        if time.time() - start > time_budget:
            break
        step *= 0.98
        
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        forces = np.zeros_like(current)
        # Vectorized repulsion
        # Force on i from j: k * (c_i - c_j) / dist^2 if dist < cutoff
        # Let's use a soft repulsion potential: V = 1/d^2
        # F = -grad V = 2/d^3 * (c_i - c_j)
        
        # To avoid division by zero, add epsilon
        dists_safe = np.where(dists < 0.1, dists, 0.1) # Cap distance for force calc?
        # Actually, we only want repulsion when close.
        # Let's use a cutoff.
        mask = dists < 0.2
        mask &= ~np.eye(N, dtype=bool)
        
        # Compute forces
        # F_i = sum_j F_ij
        # F_ij = step * (c_i - c_j) / dists_ij^2
        # This is O(N^2) per step.
        
        for i in range(N):
            for j in range(i+1, N):
                if mask[i, j]:
                    d = dists[i, j]
                    f_mag = step * (0.2 - d) / (d * d + 1e-6)
                    f_vec = f_mag * diff[i, j]
                    forces[i] += f_vec
                    forces[j] -= f_vec
                    
        current = current + forces
        current = np.clip(current, -0.49, 0.49)
        
        r = _compute_radius(current)
        if r > best_r:
            best_r = r
            best = current.copy()
            
    return best

def _smooth_polish(centers, time_budget=15.0):
    """
    Polish using L-BFGS-B with soft-min objective.
    """
    start = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Increase alpha over time to sharpen the soft-min
    alphas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for alpha in alphas:
        if time.time() - start > time_budget - 2.0:
            break
            
        def fun(x):
            obj, grad = _soft_min_obj_and_grad(x, alpha)
            return obj, grad
        
        try:
            res = minimize(
                fun,
                current.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                new_centers = res.x.reshape(N, 3)
                r = _compute_radius(new_centers)
                if r > best_r:
                    best_r = r
                    best = new_centers.copy()
                current = new_centers
        except Exception:
            pass
            
    return best

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    best_centers = None
    best_radius = -np.inf
    
    def _evaluate(centers):
        nonlocal best_centers, best_radius
        if centers is None:
            return
        if not np.all(np.isfinite(centers)):
            return
        if np.any(np.abs(centers) > 0.5):
            return
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    # Seeds
    seeds = [
        _init_fcc(),
        _init_bcc(),
        _init_random(0),
        _init_random(1)
    ]
    
    for seed in seeds:
        if time.time() - start_time > time_limit - 40:
            break
            
        current = seed.copy()
        _evaluate(current)
        
        # 1. Repulsion Anneal
        if time.time() - start_time < time_limit - 30:
            current = _repulsion_anneal(current, time_budget=10.0)
            _evaluate(current)
            
        # 2. Smooth Polish
        if time.time() - start_time < time_limit - 10:
            current = _smooth_polish(current, time_budget=20.0)
            _evaluate(current)
            
    if best_centers is None:
        return _init_fcc()
        
    # Final check and clip
    best_centers = np.clip(best_centers, -0.5 + 1e-15, 0.5 - 1e-15)
    return best_centers

if __name__ == "__main__":
    centers = construct()
    r = _compute_radius(centers)
    print(f"Radius: {r}")
