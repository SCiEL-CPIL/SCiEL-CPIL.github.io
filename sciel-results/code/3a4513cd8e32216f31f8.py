import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def _compute_radius(centers):
    # Exact radius calculation
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Use cdist for efficiency, but for N=100, it's fast enough
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    centers = centers_flat.reshape(-1, 3)
    
    # Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # LSE
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_halton_init(n, dims=3, seed=42):
    # Halton sequence
    rng = np.random.default_rng(seed)
    points = []
    # Prime bases
    bases = [2, 3, 5]
    for i in range(1, n+1):
        p = []
        for b in bases:
            f = 1.0
            x = i
            while x > 0:
                f /= b
                x = x // b
            p.append(f)
        points.append(p)
    points = np.array(points)
    # Randomize order and shift slightly to avoid symmetry
    points = points[rng.permutation(n)]
    points += rng.uniform(-0.01, 0.01, points.shape)
    # Map to [-0.5, 0.5]
    points = points - 0.5
    return np.clip(points, -0.499, 0.499)

def _build_bcc_init():
    # BCC lattice approximation
    # 5x5x4 simple grid is 100.
    # BCC has 2 per cell.
    # Let's try a 4x4x4 BCC block (4*4*4*2 = 128 points) and trim?
    # Or just use a distorted grid.
    # Let's stick to a simple grid with noise for reliability, 
    # but the prompt says inspiration uses BCC.
    # Let's generate a 5x5x4 grid and add a "center" offset to half of them?
    # Actually, a 5x5x4 grid is Simple Cubic.
    # BCC is denser.
    # Let's just use Halton and Grid. Halton is often better for optimization starts.
    
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, 5)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, 5)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, 4)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    rng = np.random.default_rng(123)
    centers += rng.normal(0, 0.01, centers.shape)
    return np.clip(centers, -0.499, 0.499)

def _polish_coordinate_descent(centers, max_iters=500, tol=1e-10):
    """
    Coordinate descent to maximize the exact radius.
    Finds the bottleneck pair/wall and pushes apart.
    """
    centers = centers.copy()
    r = _compute_radius(centers)
    
    for _ in range(max_iters):
        # Find current radius and bottleneck
        dists_to_walls = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(dists_to_walls)
        min_wall_val = np.min(dists_to_walls)
        
        dists = cdist(centers, centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        current_r = min(min_wall_val, min_pair_val / 2.0)
        
        if abs(current_r - r) < tol:
            break
            
        r = current_r
        
        # Determine move
        if min_pair_val / 2.0 < min_wall_val:
            # Push pair apart
            i, j = min_pair_idx
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-10: continue
            vec = vec / norm
            
            # Check if moving i+vec and j-vec violates walls
            # We can only move by delta such that new_r > old_r
            # Simple step: move by 0.5 * (min_pair_val - 2*min_wall_val) ? No.
            # Just move a small fixed step or adaptive step.
            step = 0.1 * (min_pair_val - 2.0 * min_wall_val) if min_pair_val > 2.0 * min_wall_val else 0.05 * min_pair_val
            # Actually, we want to increase the min distance.
            # The limiting factor is the pair.
            # Move them apart by a small epsilon.
            eps = 1e-4
            new_i = centers[i] + eps * vec
            new_j = centers[j] - eps * vec
            
            # Check walls for new_i and new_j
            if np.all(np.abs(new_i) < 0.5 - 1e-9) and np.all(np.abs(new_j) < 0.5 - 1e-9):
                # Also check if this move reduces other distances? 
                # For local polish, we assume small step is ok.
                centers[i] = new_i
                centers[j] = new_j
        else:
            # Push wall point in
            idx = min_wall_idx // 3
            axis = min_wall_idx % 3
            # Move inward
            sign = -np.sign(centers[idx, axis])
            eps = 1e-4
            new_center = centers[idx].copy()
            new_center[axis] += sign * eps
            if np.all(np.abs(new_center) < 0.5 - 1e-9):
                centers[idx] = new_center
                
    return centers

def _optimize_from_initial(centers, deadline, iu):
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Beta schedule
    betas = [50, 200, 1000, 5000]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                _soft_min_objective,
                best_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    # Polish
    if time.time() < deadline - 2:
        polished = _polish_coordinate_descent(best_centers, max_iters=200)
        r_polished = _compute_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    iu = np.triu_indices(N, k=1)
    
    inits = [
        _build_halton_init(N, seed=1),
        _build_halton_init(N, seed=2),
        _build_bcc_init(),
    ]
    
    best_centers = None
    best_radius = 0.0
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        opt_centers, opt_radius = _optimize_from_initial(init, deadline, iu)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    if best_centers is None:
        best_centers = _build_bcc_init()
        
    return best_centers
