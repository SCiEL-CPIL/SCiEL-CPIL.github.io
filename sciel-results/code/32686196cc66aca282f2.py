import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 21
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def _compute_radii(x, y, W, H):
    """
    Compute the maximum radius for each circle given centers x, y and box W, H.
    r_i = min( dist_to_nearest_neighbor_i, dist_to_boundary_i )
    """
    # Distances to boundary
    db = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
    
    # Distances to nearest neighbor
    tree = cKDTree(np.column_stack((x, y)))
    dists, _ = tree.query(np.column_stack((x, y)), k=2)
    dn = dists[:, 1]
    
    # Radius is min of boundary distance and neighbor distance
    r = np.minimum(db, dn)
    return r

def _soft_min(a, b, t=0.01):
    """
    Smooth approximation of min(a, b) using log-sum-exp.
    """
    # min(a,b) = -max(-a, -b)
    # max(u, v) ≈ -log(exp(-u/t) + exp(-v/t)) * t for small t
    # To avoid overflow, use stable log-sum-exp
    m = np.maximum(a, b)
    # min(a,b) = m - max(m-a, m-b)
    # max(p, q) ≈ t * log(exp(p/t) + exp(q/t)) for small t... 
    # Actually: min(a,b) ≈ -t*log(exp(-a/t) + exp(-b/t))
    # But this can underflow. Let's use:
    # min(a,b) = a + b - max(a,b) - |a-b|
    # Smooth: |x| ≈ t*log(1+exp(x/t)) - x... this is getting complex.
    
    # Simpler: use the fact that min(a,b) = (a+b - |a-b|)/2
    # And |x| ≈ sqrt(x^2 + eps^2) for smoothness, but that's not great.
    
    # Better: use the log-sum-exp trick properly
    # min(a,b) = -max(-a, -b)
    # max(u,v) = t * log(exp(u/t) + exp(v/t)) for t -> 0+ gives max
    # So min(a,b) = -t * log(exp(-a/t) + exp(-b/t))
    # For numerical stability:
    # = -t * (max(-a,-b)/t + log(exp(-a/t - max(-a,-b)/t) + exp(-b/t - max(-a,-b)/t)))
    # = max(-a,-b) - t*log(exp(-a/t - max(-a,-b)/t) + exp(-b/t - max(-a,-b)/t))
    
    # Actually, let me just use a simpler smooth min:
    # min(a,b) ≈ (a+b - sqrt((a-b)^2 + eps))/2
    eps = t*t
    return 0.5 * (a + b - np.sqrt((a - b)**2 + eps))

def _objective_smooth(v, W, H, t=0.01):
    """
    Smooth objective: minimize -sum(soft_min(boundary_dist, neighbor_dist))
    """
    x = v[:N]
    y = v[N:2*N]
    
    # Distances to boundary
    db = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
    
    # Distances to nearest neighbor
    tree = cKDTree(np.column_stack((x, y)))
    dists, _ = tree.query(np.column_stack((x, y)), k=2)
    dn = dists[:, 1]
    
    # Soft min
    r = _soft_min(db, dn, t)
    return -np.sum(r)

def _objective(v, W, H):
    """
    Objective function to minimize: -sum(radii)
    """
    x = v[:N]
    y = v[N:2*N]
    r = _compute_radii(x, y, W, H)
    return -np.sum(r)

def _validate(x, y, W, H, tol=1e-6):
    """
    Check if the configuration is valid.
    """
    # Check boundaries
    if np.any(x < -tol) or np.any(x > W + tol):
        return False
    if np.any(y < -tol) or np.any(y > H + tol):
        return False
    
    # Check pairwise distances
    # r_i = min(dist_to_bound, dist_to_neighbor)
    # If we just check dist >= r_i + r_j, it's sufficient if r_i, r_j are derived correctly.
    # But let's explicitly check disjointness with the computed radii.
    r = _compute_radii(x, y, W, H)
    
    # Check boundaries with radii
    if np.any(x - r < -tol) or np.any(x + r > W + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > H + tol):
        return False
        
    # Check pairwise
    tree = cKDTree(np.column_stack((x, y)))
    pairs = tree.query_pairs(r.max() + tol) # conservative check
    for i, j in pairs:
        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
        if dist < r[i] + r[j] - tol:
            return False
    return True

def _initial_grid(W, H):
    """
    Generate a grid-like initial configuration.
    """
    cols = int(np.ceil(np.sqrt(N * W / H))) if W > H else int(np.ceil(np.sqrt(N * H / W)))
    # Simple 5x5 grid for 21 circles
    cols = 5
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    k = 0
    for i in range(cols):
        for j in range(rows):
            if k >= N:
                break
            x[k] = (i + 0.5) * W / cols
            y[k] = (j + 0.5) * H / rows
            k += 1
    return np.concatenate([x, y])

def _initial_random(W, H, seed=42):
    """
    Generate a random initial configuration.
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, W, N)
    y = rng.uniform(0, H, N)
    return np.concatenate([x, y])

def _sa_optimize(v0, W, H, max_iter=2000, t0=1.0, t_end=0.01, alpha=0.999):
    """
    Simulated Annealing to find a good local minimum.
    """
    v = v0.copy()
    curr_val = _objective(v, W, H)
    best_val = curr_val
    best_v = v.copy()
    
    T = t0
    for i in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Propose a move
        step = 0.1 * T
        idx = np.random.randint(0, 2*N)
        delta = np.random.uniform(-step, step)
        
        v_new = v.copy()
        v_new[idx] += delta
        
        v_new[:N] = np.clip(v_new[:N], 0, W)
        v_new[N:] = np.clip(v_new[N:], 0, H)
        
        new_val = _objective(v_new, W, H)
        delta_val = new_val - curr_val
        
        if delta_val < 0 or np.random.rand() < np.exp(-delta_val / T):
            v = v_new
            curr_val = new_val
            if curr_val < best_val:
                best_val = curr_val
                best_v = v.copy()
                
        T *= alpha
        if T < t_end:
            break
            
    return best_v, best_val

def _polish_smooth(v, W, H, t=0.01):
    """
    Use L-BFGS-B with smooth objective to polish.
    """
    bounds = []
    for i in range(N):
        bounds.append((0, W))
        bounds.append((0, H))
        
    try:
        res = minimize(lambda vv: _objective_smooth(vv, W, H, t), v, 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
        return res.x, _objective(res.x, W, H)
    except Exception:
        return v, _objective(v, W, H)

def _polish(v, W, H):
    """
    Use L-BFGS-B to polish the solution found by SA.
    First use smooth objective, then exact.
    """
    # First polish with smooth objective
    v1, val1 = _polish_smooth(v, W, H, t=0.01)
    
    # Then polish with exact objective using tighter settings
    bounds = []
    for i in range(N):
        bounds.append((0, W))
        bounds.append((0, H))
        
    try:
        res = minimize(_objective, v1, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        return res.x, res.fun
    except Exception:
        return v1, val1

def _optimize_for_aspect(W, H):
    """
    Run SA + Polish for a specific aspect ratio.
    """
    best_val = np.inf
    best_v = None
    
    # Try multiple initializations
    inits = []
    inits.append(_initial_grid(W, H))
    for seed in [1, 2, 3]:
        inits.append(_initial_random(W, H, seed))
        
    for v0 in inits:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # SA
        v_sa, val_sa = _sa_optimize(v0, W, H)
        
        # Polish with smooth then exact
        v_pol, val_pol = _polish(v_sa, W, H)
        
        if val_pol < best_val:
            best_val = val_pol
            best_v = v_pol
            
    return best_v, -best_val # Return negative value as sum of radii

def construct():
    best_val = -1
    best_res = None
    
    # Aspect ratios W, H=2-W
    # Focus on promising aspect ratios with more samples near 1.0
    widths = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4])
    
    for W in widths:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        H = 2.0 - W
        if H <= 0:
            continue
            
        v, val = _optimize_for_aspect(W, H)
        
        if v is None:
            continue
            
        x = v[:N]
        y = v[N:]
        
        # Validate
        if _validate(x, y, W, H):
            if val > best_val:
                best_val = val
                best_res = np.column_stack((x, y, _compute_radii(x, y, W, H)))
                
    if best_res is None:
        # Fallback to seed: 1x1 square, grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_res
