"""Pack 50 equal circles in a unit square, maximising the common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# Known best radius (Boll et al. 2000)
R_KNOWN = 0.071377103865


def _true_radius(centers):
    """Compute the true common radius for a given set of centers."""
    centers = np.asarray(centers)
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(centers), 0.5)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_wall, min_pair / 2.0)


def _softmin(x, beta):
    """Soft minimum of an array using log-sum-exp."""
    x = np.asarray(x)
    min_val = np.min(x)
    # Avoid overflow
    shifted = x - min_val
    log_sum = min_val + np.log(np.sum(np.exp(-beta * shifted)))
    return -log_sum / beta


def _negative_soft_radius(centers_flat, beta):
    """
    Negative of the soft-min radius surrogate.
    We minimize this to maximize the radius.
    centers_flat: 1D array of shape (2N,)
    """
    centers = centers_flat.reshape(N, 2)
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centers), 0.5)
    min_wall = _softmin(wall_dists, beta)
    
    # Pairwise distances / 2
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_radii = dists / 2.0
    # Only take upper triangle to avoid duplicates
    iu = np.triu_indices(N, k=1)
    pair_vals = pair_radii[iu]
    min_pair = _softmin(pair_vals, beta)
    
    # Combined soft min
    combined = _softmin(np.array([min_wall, min_pair]), beta)
    return -combined


def _force_relax(centers, max_iter=5000, deadline=None):
    """
    Force-directed relaxation to resolve overlaps.
    Pushes overlapping circles apart and pulls them away from walls.
    """
    centers = centers.copy()
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall repulsion
        for i in range(N):
            for dim in range(2):
                if centers[i, dim] > 0:
                    dist_to_wall = 0.5 - centers[i, dim]
                    if dist_to_wall < 0.05:
                        forces[i, dim] += 10.0 * (0.05 - dist_to_wall)
                else:
                    dist_to_wall = 0.5 + centers[i, dim]
                    if dist_to_wall < 0.05:
                        forces[i, dim] -= 10.0 * (0.05 - dist_to_wall)
        
        # Pair repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 0.15:  # Repel if closer than 0.15
                    if dist < 1e-10:
                        force_vec = np.random.normal(0, 0.01, size=2)
                    else:
                        force_vec = diff / dist
                    strength = 50.0 * (0.15 - dist)
                    forces[i] += strength * force_vec
                    forces[j] -= strength * force_vec
        
        # Apply forces with damping
        step_size = 0.01 / (1.0 + 0.01 * it)
        centers += step_size * forces
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers


def _hex_init():
    """Create a hexagonal-like initial layout."""
    centers = np.zeros((N, 2))
    # Try to arrange in rows with hexagonal offset
    # 7 rows: some with 7, some with 8 circles
    row_counts = [7, 8, 7, 8, 7, 8, 5]  # Total = 50
    row_spacing = 1.0 / (len(row_counts) + 0.5)
    idx = 0
    for r, count in enumerate(row_counts):
        y = -0.5 + (r + 0.5) * row_spacing
        offset = 0 if r % 2 == 0 else 0.5 / count
        for c in range(count):
            x = -0.5 + (c + 0.5 + offset) * (1.0 / count)
            centers[idx] = [x, y]
            idx += 1
    return centers


def _random_init(seed=42):
    """Random initialization."""
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, size=(N, 2))


def _grid_init():
    """Grid initialization with jitter."""
    np.random.seed(123)
    # 5x10 grid
    xs = np.linspace(-0.45, 0.45, 10)
    ys = np.linspace(-0.45, 0.45, 5)
    centers = np.array([[x, y] for y in ys for x in xs])
    # Add small jitter
    centers += np.random.normal(0, 0.01, size=(N, 2))
    return centers


def _gradient_optimize(centers, deadline=None):
    """
    Progressive L-BFGS-B optimization with increasing beta (sharper soft-min).
    """
    best = centers.copy()
    best_score = _true_radius(best)
    
    # Progressive beta values (start smooth, get sharper)
    betas = [10, 30, 100, 300, 1000]
    
    current = centers.copy()
    for beta in betas:
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Flatten for optimization
        x0 = current.flatten()
        
        # Bounds: each coordinate in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                _negative_soft_radius,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
            )
            new_centers = result.x.reshape(N, 2)
            new_score = _true_radius(new_centers)
            
            if new_score > best_score:
                best = new_centers.copy()
                best_score = new_score
            current = new_centers
        except Exception:
            pass
    
    return best, best_score


def _anneal_refine(centers, deadline=None, max_iter=5000):
    """
    Simple simulated annealing refinement on top of gradient result.
    """
    current = centers.copy()
    current_score = _true_radius(current)
    best = current.copy()
    best_score = current_score
    
    t = 0.005
    alpha = 0.999
    
    np.random.seed(7)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Propose move
        idx = np.random.randint(N)
        delta = np.random.normal(0, t, size=2)
        proposed = current.copy()
        proposed[idx] += delta
        proposed[idx] = np.clip(proposed[idx], -0.5, 0.5)
        
        proposed_score = _true_radius(proposed)
        
        # Accept if better or with probability exp((new-old)/t)
        if proposed_score > current_score or np.random.random() < np.exp((proposed_score - current_score) / max(t, 1e-10)):
            current = proposed
            current_score = proposed_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        t *= alpha
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_overall = None
    best_score_overall = -np.inf
    
    # Multiple initialization strategies
    inits = [
        ("hex", _hex_init()),
        ("random", _random_init(42)),
        ("grid", _grid_init()),
        ("random2", _random_init(123)),
    ]
    
    for name, init_centers in inits:
        if time.time() > deadline - 30:
            break
        
        # Step 1: Force relaxation to resolve overlaps
        relaxed = _force_relax(init_centers, max_iter=3000, deadline=deadline)
        
        # Step 2: Gradient optimization with progressive soft-min
        opt_centers, opt_score = _gradient_optimize(relaxed, deadline=deadline)
        
        if opt_score > best_score_overall:
            best_overall = opt_centers
            best_score_overall = opt_score
        
        # Step 3: Annealing refinement if time permits
        if time.time() < deadline - 10:
            anneal_centers, anneal_score = _anneal_refine(opt_centers, deadline=deadline, max_iter=3000)
            if anneal_score > best_score_overall:
                best_overall = anneal_centers
                best_score_overall = anneal_score
    
    if best_overall is None:
        best_overall = _hex_init()
    
    # Final clip to ensure validity
    best_overall = np.clip(best_overall, -0.5, 0.5)
    
    return best_overall
