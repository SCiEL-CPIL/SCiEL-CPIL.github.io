"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0  # Safety margin for 120s limit

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        norms = np.linalg.norm(centers, axis=1)
        return np.min(1.0 - norms)
    
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _get_hex_init(n):
    """Generate hexagonal lattice points inside unit circle."""
    r_est = 0.125
    s = 2 * r_est * 1.01  # slight gap
    sqrt3 = np.sqrt(3)
    
    points = []
    max_range = int(1.5 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * sqrt3 / 2.0
            if np.hypot(x, y) <= 0.92:
                points.append([x, y])
                
    if len(points) == 0:
        return _get_random_init(n, seed=42)
        
    points = np.array(points)
    
    if len(points) < n:
        return _get_random_init(n, seed=42)
    
    # Pick n points closest to center for densest packing
    dists_to_center = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists_to_center)[:n]
    return points[idx]

def _get_ring_init(n):
    """Generate ring-based initialization with 1 center + rings."""
    centers = [[0.0, 0.0]]
    count = 1
    
    # Ring configurations: (num_points, radius)
    rings = [
        (6, 0.22),
        (12, 0.44),
        (12, 0.66),
        (10, 0.85),
        (10, 0.95)
    ]
    
    for num_pts, radius in rings:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        for a in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(a), radius * np.sin(a)])
            count += 1
            
    return np.array(centers[:n])

def _get_random_init(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = np.sqrt(rng.uniform(0, 0.9))
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _soft_min_objective(params, n, beta):
    """
    Objective: -soft_min of all constraint values.
    Constraint values: d_ij/2 for pairs, 1-|c_i| for boundary.
    """
    centers = params.reshape(n, 2)
    
    # Pair distances
    dists = cdist(centers, centers)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    vals = np.concatenate([pair_dists, bound_dists])
    m = np.min(vals)
    
    # Soft min via log-sum-exp
    exp_terms = np.exp(-beta * (vals - m))
    S = np.sum(exp_terms)
    soft_min = m - (1.0 / beta) * np.log(S)
    obj = -soft_min
    
    # Gradient
    grad = np.zeros_like(centers)
    weights = exp_terms / S
    
    pair_w = weights[:n*(n-1)//2]
    bound_w = weights[n*(n-1)//2:]
    
    for k in range(len(pair_w)):
        w = pair_w[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        d = np.linalg.norm(centers[i] - centers[j])
        if d > 1e-10:
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
            
    for i in range(n):
        w = bound_w[i]
        if w < 1e-10:
            continue
        if norms[i] > 1e-10:
            grad[i] -= w * centers[i] / norms[i]
            
    return obj, -grad

def _polish(centers, start_time, max_iters=3000, base_step=0.001):
    """
    Sub-gradient ascent polish: identify tightest constraint, push apart.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        n = len(current)
        dists = cdist(current, current)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - norms
        
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(bound_dists)
        
        r = min(min_pair, min_bound)
        
        # Adaptive step size
        step = base_step * (1.0 + 0.5 * np.sin(it * 0.01))
        
        if min_pair < min_bound:
            # Bottleneck is a pair
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            if i > j:
                i, j = j, i
            d = dists[i, j]
            if d > 1e-10:
                vec = (current[i] - current[j]) / d
                current[i] += vec * step * 0.5
                current[j] -= vec * step * 0.5
        else:
            # Bottleneck is boundary
            i = np.argmin(bound_dists)
            if norms[i] > 1e-10:
                vec = current[i] / norms[i]
                current[i] -= vec * step
        
        # Clamp inside
        new_norms = np.linalg.norm(current, axis=1)
        mask = new_norms > 0.999
        if np.any(mask):
            current[mask] *= 0.999 / new_norms[mask]
            
        new_score = _compute_score(current)
        if new_score > best_score + 1e-12:
            best_score = new_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 500:
                break
                
    return current, best_score

def _optimize(centers, start_time):
    """
    Multi-stage optimization:
    1. L-BFGS-B with increasing beta
    2. Sub-gradient polish
    """
    current = centers.copy()
    
    # Stage 1: L-BFGS-B with soft-min
    betas = [30.0, 100.0, 300.0, 1000.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 15:
            break
        try:
            res = minimize(
                _soft_min_objective,
                current.flatten(),
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            if np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate)):
                    # Verify improvement
                    if _compute_score(candidate) >= _compute_score(current) - 1e-6:
                        current = candidate
        except Exception:
            pass
            
    # Stage 2: Polish
    if time.time() - start_time < DEADLINE - 3:
        current, _ = _polish(current, start_time, max_iters=5000, base_step=0.0005)
        
    return current

def construct():
    start_time = time.time()
    best_score = -1.0
    best_centers = None
    
    # Multiple diverse initializations
    inits = [
        _get_hex_init(N),
        _get_ring_init(N),
        _get_random_init(N, seed=1),
        _get_random_init(N, seed=7),
        _get_random_init(N, seed=13),
        _get_random_init(N, seed=100),
    ]
    
    # Optimize each initialization
    for init in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Ensure inside unit circle
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.9
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.9 / norms[mask]
            
        optimized = _optimize(init, start_time)
        
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
                
        # Early stop if we're very close to known best
        if best_score > 0.1257:
            break
            
    if best_centers is None:
        return _get_hex_init(N)
        
    # Final high-precision polish on the best
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _polish(best_centers, start_time, max_iters=20000, base_step=0.0001)
        
    # Final clamp
    norms = np.linalg.norm(best_centers, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_centers = best_centers.copy()
        best_centers[mask] *= 0.9999 / norms[mask]
        
    return best_centers
# EVOLVE-BLOCK-END
