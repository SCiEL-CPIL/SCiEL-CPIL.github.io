import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def _get_deadline():
    return time.time() + 140.0

def _time_left(deadline):
    return max(0.0, deadline - time.time())

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _get_bottleneck_info(centers):
    """
    Get the bottleneck: which constraint is limiting the radius.
    Returns (radius, bottleneck_type, bottleneck_indices)
    """
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    half_pair = min_pair_dist / 2.0
    
    if min_wall_dist <= half_pair:
        # Bottleneck is wall
        idx = np.argmin(dists_to_walls)
        # Find which coordinate
        coord = np.argmin(dists_to_walls[idx])
        return min_wall_dist, 'wall', (idx, coord)
    else:
        # Bottleneck is pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        return half_pair, 'pair', (i, j)

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
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid_init(nx, ny, nz, noise_scale=0.0):
    """Build a grid initialization."""
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        return None
    
    if noise_scale > 0:
        rng = np.random.default_rng(int(np.sum(np.round(centers*1000))) % 10000)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_bcc_init():
    """
    Build a BCC lattice initialization.
    """
    a = (2.0 / N) ** (1/3.0)
    n_cells = int(np.ceil(1.0 / a)) + 2
    
    points = []
    for i in range(-1, n_cells):
        for j in range(-1, n_cells):
            for k in range(-1, n_cells):
                points.append([i*a, j*a, k*a])
                points.append([i*a + a/2, j*a + a/2, k*a + a/2])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) > N:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        return _build_grid_init(5, 5, 4, 0.01)
        
    return points

def _build_sobol_init():
    """
    Build a Sobol sequence initialization for better space-filling.
    """
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True)
    m = 10
    points = sampler.random(2 ** m)  # 1024 points in [0,1]^3
    # Scale to [-0.45, 0.45] to stay safely inside
    points = points * 0.9 - 0.45
    # Select N points that maximize minimum distance (greedy)
    selected = [points[0]]
    for _ in range(N - 1):
        d = np.min(cdist(points, np.array(selected), metric='euclidean'), axis=1)
        selected.append(points[np.argmax(d)])
    return np.array(selected)

def _build_halton_init():
    """
    Build a Halton sequence initialization.
    """
    def halton(index, base):
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f /= base
        return result
    
    points = []
    for i in range(1, N + 1):
        points.append([
            halton(i, 2) * 0.9 - 0.45,
            halton(i, 3) * 0.9 - 0.45,
            halton(i, 5) * 0.9 - 0.45
        ])
    return np.array(points)

def _simulated_annealing(centers, deadline, iu, max_iter=2000):
    """
    Simulated annealing to escape local minima.
    Uses the soft-min objective with decreasing beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current = centers.copy()
    r_current = best_radius
    
    # Temperature schedule
    T_start = 0.03
    T_end = 0.0005
    n_steps = max_iter
    
    for i in range(n_steps):
        if _time_left(deadline) < 2.0:
            break
        
        T = T_start * (T_end / T_start) ** (i / max(1, n_steps - 1))
        
        # Generate perturbation
        rng = np.random.default_rng(42 + i)
        perturbation = rng.normal(0, T, current.shape)
        candidate = np.clip(current + perturbation, -0.499, 0.499)
        
        # Evaluate
        r_candidate = _compute_radius(candidate)
        
        # Accept if better, or with probability exp(-(r_current - r_candidate)/T)
        if r_candidate >= r_current:
            current = candidate
            r_current = r_candidate
        else:
            delta = r_current - r_candidate
            if delta > 0 and rng.random() < np.exp(-delta / max(T, 1e-10)):
                current = candidate
                r_current = r_candidate
        
        if r_current > best_radius:
            best_radius = r_current
            best_centers = current.copy()
    
    return best_centers, best_radius

def _polish_repulsion(centers, deadline, iu):
    """
    Polish by focusing on repulsion between the closest pairs.
    Uses a weighted objective that emphasizes the most constrained pairs.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    if _time_left(deadline) < 3.0:
        return best_centers, best_radius
    
    # Find the k closest pairs and create a weighted objective
    dists = cdist(best_centers, best_centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    
    # Get the 20 closest pairs
    flat_dists = dists[iu]
    k = min(20, len(flat_dists))
    closest_idx = np.argsort(flat_dists)[:k]
    
    # Create weighted soft-min: weight the closest pairs more heavily
    beta = 2000.0
    
    def weighted_objective(centers_flat):
        c = centers_flat.reshape(-1, 3)
        d = cdist(c, c, metric='euclidean')
        np.fill_diagonal(d, np.inf)
        pair_vals = d[iu] / 2.0
        wall_vals = (0.5 - np.abs(c)).flatten()
        
        # Weight: closest pairs get higher weight
        weights = np.ones(len(pair_vals))
        for idx in closest_idx:
            weights[idx] = 3.0
        
        # Weighted log-sum-exp for pairs
        neg_beta_pair = -beta * pair_vals
        max_pair = np.max(neg_beta_pair)
        sum_exp_pair = np.sum(weights * np.exp(neg_beta_pair - max_pair))
        pair_obj = (max_pair + np.log(sum_exp_pair)) / beta
        
        # Standard soft-min for walls
        neg_beta_wall = -beta * wall_vals
        max_wall = np.max(neg_beta_wall)
        sum_exp_wall = np.sum(np.exp(neg_beta_wall - max_wall))
        wall_obj = (max_wall + np.log(sum_exp_wall)) / beta
        
        return min(pair_obj, wall_obj)
    
    try:
        res = minimize(
            weighted_objective,
            best_centers.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-9, 'maxls': 25}
        )
        
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        r = _compute_radius(new_centers)
        if r > best_radius:
            best_radius = r
            best_centers = new_centers
            
    except Exception:
        pass
    
    return best_centers, best_radius

def _polish_bottleneck(centers, deadline, iu):
    """
    Polish by specifically targeting the bottleneck constraint.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    if _time_left(deadline) < 3.0:
        return best_centers, best_radius
    
    # Iterate: find bottleneck, push it out
    for iteration in range(5):
        if _time_left(deadline) < 2.0:
            break
        
        r, btype, bidx = _get_bottleneck_info(best_centers)
        
        if btype == 'pair':
            i, j = bidx
            # Push spheres i and j apart
            direction = best_centers[i] - best_centers[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                # Try pushing both apart
                step = 0.001
                for trial in range(3):
                    candidate = best_centers.copy()
                    candidate[i] += direction * step
                    candidate[j] -= direction * step
                    candidate = np.clip(candidate, -0.499, 0.499)
                    r_new = _compute_radius(candidate)
                    if r_new > best_radius:
                        best_radius = r_new
                        best_centers = candidate
                    else:
                        break
                    step *= 0.5
        elif btype == 'wall':
            idx, coord = bidx
            # Push sphere idx away from wall
            sign = np.sign(best_centers[idx, coord])
            if sign == 0:
                sign = 1.0
            direction = np.zeros(3)
            direction[coord] = sign
            step = 0.001
            for trial in range(3):
                candidate = best_centers.copy()
                candidate[idx] += direction * step
                candidate = np.clip(candidate, -0.499, 0.499)
                r_new = _compute_radius(candidate)
                if r_new > best_radius:
                    best_radius = r_new
                    best_centers = candidate
                else:
                    break
                step *= 0.5
    
    return best_centers, best_radius

def _polish_high_beta(centers, deadline, iu):
    """
    High-precision polish pass with very high beta and tight tolerances.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Very high beta for near-exact min
    beta = 10000.0
    
    if _time_left(deadline) < 3.0:
        return best_centers, best_radius
    
    try:
        res = minimize(
            _soft_min_objective,
            best_centers.flatten(),
            args=(beta, iu),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 30}
        )
        
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        r = _compute_radius(new_centers)
        if r > best_radius:
            best_radius = r
            best_centers = new_centers
            
    except Exception:
        pass
    
    return best_centers, best_radius

def _optimize_from_initial(centers, deadline, iu):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Beta schedule
    beta_start = 100.0
    beta_end = 5000.0
    n_beta_steps = 10
    
    if _time_left(deadline) < 5.0:
        return best_centers, best_radius

    for step in range(n_beta_steps):
        if _time_left(deadline) < 1.0:
            break
            
        if n_beta_steps == 1:
            beta = beta_start
        else:
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 20}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
                
        except Exception:
            pass
            
    # Simulated annealing phase to escape local minima
    if _time_left(deadline) > 10.0:
        sa_centers, sa_radius = _simulated_annealing(best_centers, deadline, iu, max_iter=1000)
        if sa_radius > best_radius:
            best_radius = sa_radius
            best_centers = sa_centers
    
    # Repulsion polish
    if _time_left(deadline) > 5.0:
        rep_centers, rep_radius = _polish_repulsion(best_centers, deadline, iu)
        if rep_radius > best_radius:
            best_radius = rep_radius
            best_centers = rep_centers
    
    # Bottleneck polish
    if _time_left(deadline) > 3.0:
        bn_centers, bn_radius = _polish_bottleneck(best_centers, deadline, iu)
        if bn_radius > best_radius:
            best_radius = bn_radius
            best_centers = bn_centers
    
    # High-precision polish
    if _time_left(deadline) > 3.0:
        polish_centers, polish_radius = _polish_high_beta(best_centers, deadline, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = _get_deadline()
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    # 1. Grid Initializations
    grid_configs = [
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 4),
        (10, 10, 1),
        (10, 5, 2),
    ]
    
    noise_scales = [0.0, 0.01, 0.02]
    
    for nx, ny, nz in grid_configs:
        if _time_left(deadline) < 10.0:
            break
        
        for noise in noise_scales:
            if _time_left(deadline) < 5.0:
                break
            
            centers = _build_grid_init(nx, ny, nz, noise)
            if centers is None:
                continue
                
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
    
    # 2. BCC Initialization
    if _time_left(deadline) > 15.0:
        centers = _build_bcc_init()
        if centers is not None and len(centers) == N:
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # 3. Sobol Initialization
    if _time_left(deadline) > 15.0:
        centers = _build_sobol_init()
        if centers is not None and len(centers) == N:
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # 4. Halton Initialization
    if _time_left(deadline) > 15.0:
        centers = _build_halton_init()
        if centers is not None and len(centers) == N:
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # 5. Random Initializations
    rng = np.random.default_rng(123)
    n_random_starts = 3
    
    if _time_left(deadline) > 10.0:
        for i in range(n_random_starts):
            if _time_left(deadline) < 5.0:
                break
            
            centers = rng.uniform(-0.45, 0.45, (N, 3))
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # Final polish on the best solution
    if best_centers is not None and _time_left(deadline) > 3.0:
        final_centers, final_radius = _polish_high_beta(best_centers, deadline, iu)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = final_centers
        
        # Additional bottleneck polish
        if _time_left(deadline) > 2.0:
            bn_centers, bn_radius = _polish_bottleneck(best_centers, deadline, iu)
            if bn_radius > best_radius:
                best_radius = bn_radius
                best_centers = bn_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
    
    return best_centers
