import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
        
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    # Using cdist is efficient for N=100
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    """Build a grid initialization."""
    # Ensure we get exactly N points
    # For N=100, 5x5x4 is a good start
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        # Pad or truncate
        if len(centers) < N:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
        else:
            centers = centers[:N]

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    pair_vals = dists / 2.0
    # Only use upper triangle to avoid duplicates, but we need indices
    # iu is precomputed indices of upper triangle
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # This is the log-sum-exp of -beta * x, which approximates min(x) as beta -> inf
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _optimize_from_initial(centers, deadline, iu, time_budget=20.0):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    
    # Phase 1: Soft-min optimization
    # Start with low beta (smooth), increase to high beta (sharp)
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_centers = centers.copy()
    
    for beta in betas:
        if _time_left() < 2.0 or (time.time() - start_t) > time_budget - 2.0:
            break
            
        x0 = current_centers.flatten()
        
        try:
            # Reduce iterations as we run out of time
            max_iter = 100 if beta < 100 else 50
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 0: # res.fun should be negative radius
                new_centers = res.x.reshape(-1, 3)
                new_radius = _compute_radius(new_centers)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
                    current_centers = new_centers.copy()
                else:
                    # Even if not better, maybe it's a good start for next beta?
                    # Usually yes, keep the trajectory if it's valid
                    if np.all(np.isfinite(new_centers)):
                        current_centers = new_centers.copy()
                        
        except Exception as e:
            # If optimization fails, keep current
            pass
            
    # Phase 2: Local Repulsion Polish
    # Use a simple force-directed method to nudge points apart
    # This helps escape shallow local minima where soft-min gets stuck
    
    current_centers = best_centers.copy()
    r_current = _compute_radius(current_centers)
    
    # Precompute upper triangle indices for distance checks
    # We will use KDTree for efficiency in finding close pairs
    
    step_size = 0.005
    decay = 0.95
    min_step = 1e-6
    margin = 1.05 # We want distance > 2*r*margin
    
    no_improve_count = 0
    max_no_improve = 100
    
    while _time_left() > 1.0 and (time.time() - start_t) < time_budget - 0.5 and step_size > min_step and no_improve_count < max_no_improve:
        r = _compute_radius(current_centers)
        if r > best_radius + 1e-12:
            best_radius = r
            best_centers = current_centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Compute forces
        forces = np.zeros_like(current_centers)
        
        # Wall forces: push away from walls if too close
        for k in range(3):
            dists = 0.5 - np.abs(current_centers[:, k])
            limit = max(r_current * margin, 0.01)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(current_centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # Pairwise forces: push apart if too close
        # Use KDTree to find neighbors within limit
        tree = cKDTree(current_centers)
        limit_dist = 2 * max(r_current * margin, 0.01)
        
        for i in range(N):
            # Find neighbors within limit_dist
            neighbors = tree.query_ball_point(current_centers[i], limit_dist)
            for j in neighbors:
                if j <= i:
                    continue
                diff = current_centers[i] - current_centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    # Random push if coincident
                    forces[i] += np.random.normal(0, 0.01, 3)
                    forces[j] -= np.random.normal(0, 0.01, 3)
                elif dist < limit_dist:
                    strength = (limit_dist - dist) / limit_dist
                    unit_diff = diff / dist
                    forces[i] += unit_diff * strength
                    forces[j] -= unit_diff * strength
        
        # Normalize and apply
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        current_centers += step_size * forces
        current_centers = np.clip(current_centers, -0.4999, 0.4999)
        
        step_size *= decay
        r_current = _compute_radius(current_centers)
        
    return best_centers, best_radius

def construct():
    start_time = time.time()
    
    # Precompute upper triangle indices for soft-min objective
    iu = np.triu_indices(N, k=1)
    
    # Strategy:
    # 1. Try multiple initializations (Grid, Halton, Random)
    # 2. For each, run Soft-Min + Repulsion Polish
    # 3. Keep the best result
    
    initializations = [
        _build_grid_init(5, 5, 4, noise_scale=0.01, seed=42),
        _build_grid_init(5, 5, 4, noise_scale=0.05, seed=123),
        _build_halton_init(seed=42),
        _build_halton_init(seed=99),
        np.random.default_rng(1).uniform(-0.45, 0.45, (N, 3)),
        np.random.default_rng(2).uniform(-0.45, 0.45, (N, 3)),
    ]
    
    best_centers_overall = None
    best_radius_overall = -1.0
    
    # Calculate time per initialization
    # We have ~145s total.
    # Let's try 4 initializations with ~30s each, leaving 25s for a final long polish on the best.
    
    n_inits = len(initializations)
    time_per_init = 25.0
    
    for i, init_centers in enumerate(initializations):
        if _time_left() < time_per_init + 5.0:
            break
            
        # Ensure valid start
        if not np.all(np.isfinite(init_centers)):
            continue
            
        centers, radius = _optimize_from_initial(
            init_centers, 
            _get_deadline(), 
            iu, 
            time_budget=time_per_init
        )
        
        if radius > best_radius_overall:
            best_radius_overall = radius
            best_centers_overall = centers.copy()
            
    # Final polish on the best result found
    if best_centers_overall is not None and _time_left() > 5.0:
        # Use remaining time for one more long run on the best
        time_remaining = _time_left() - 2.0
        if time_remaining > 10.0:
            centers, radius = _optimize_from_initial(
                best_centers_overall,
                _get_deadline(),
                iu,
                time_budget=time_remaining
            )
            if radius > best_radius_overall:
                best_radius_overall = radius
                best_centers_overall = centers.copy()
                
    # Fallback if something went wrong
    if best_centers_overall is None:
        best_centers_overall = _build_grid_init(5, 5, 4, noise_scale=0.0, seed=42)
        
    return best_centers_overall
