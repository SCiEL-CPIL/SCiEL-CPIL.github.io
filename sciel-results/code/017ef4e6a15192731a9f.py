"""Packing 50 equal circles in a unit square using force-directed layout + simulated annealing + polish."""
N = 50
import time
import numpy as np
from scipy.spatial import cKDTree

DEADLINE_FRACTION = 0.85  # safety margin: stop at 85% of 150s


def _compute_radius(centers):
    """Compute the common radius for a given set of centers."""
    # Boundary constraint
    boundary_r = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_r)
    
    # Pairwise constraint
    tree = cKDTree(centers)
    pairs = tree.query_pairs(0.5)  # pairs within 0.5 (generous upper bound)
    if len(pairs) == 0:
        return min_boundary
    dists = np.array([np.linalg.norm(centers[i] - centers[j]) for i, j in pairs])
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)


def _force_layout(centers, n_iters=5000, dt=0.001):
    """Force-directed layout: repulsive forces between circles, attractive toward center."""
    centers = centers.copy()
    n = len(centers)
    
    for it in range(n_iters):
        forces = np.zeros_like(centers)
        
        # Repulsive forces between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random small perturbation
                    d = np.random.randn(2) * 1e-6
                    dist = np.linalg.norm(d)
                # Force magnitude ~ 1/dist^2 (strong repulsion at close range)
                f = 1.0 / (dist * dist + 1e-8)
                f_vec = f * d / dist
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Attractive force toward center (weak)
        forces -= 0.01 * centers
        
        # Damping
        dt_eff = dt * (1.0 - 0.5 * it / n_iters)
        centers += dt_eff * forces
        
        # Clamp to square
        centers = np.clip(centers, -0.499, 0.499)
    
    return centers


def _simulated_annealing(centers, n_iters=20000, t_start=0.05, t_end=0.001, seed=42):
    """Simulated annealing to maximize the minimum radius."""
    rng = np.random.RandomState(seed)
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    for it in range(n_iters):
        # Cooling schedule
        t = t_start * (t_end / t_start) ** (it / n_iters)
        
        # Propose a move: perturb a random center
        idx = rng.randint(N)
        step_size = t * 0.5
        delta = rng.randn(2) * step_size
        new_centers = centers.copy()
        new_centers[idx] += delta
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_r = _compute_radius(new_centers)
        delta_r = new_r - best_r
        
        if delta_r > 0 or rng.random() < np.exp(delta_r / t):
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
    
    return best_centers, best_r


def _polish(centers, n_iters=500, tol=1e-10):
    """Gradient-based polish: iteratively adjust centers to maximize min radius."""
    centers = centers.copy()
    current_r = _compute_radius(centers)
    
    for it in range(n_iters):
        # Find the critical constraint (which pair or boundary limits the radius)
        # Compute all pairwise distances and boundary distances
        tree = cKDTree(centers)
        pairs = tree.query_pairs(0.2)
        
        min_val = 0.5
        min_type = None  # ('pair', i, j) or ('boundary', i, k)
        
        # Check boundary
        for i in range(N):
            for k in range(2):
                val = 0.5 - abs(centers[i, k])
                if val < min_val:
                    min_val = val
                    min_type = ('boundary', i, k)
        
        # Check pairs
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            if d < min_val:
                min_val = d
                min_type = ('pair', i, j)
        
        if min_type is None:
            break
        
        # Compute gradient of the min constraint w.r.t. centers
        grad = np.zeros_like(centers)
        if min_type[0] == 'pair':
            i, j = min_type[1], min_type[2]
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                continue
            # gradient of |c_i - c_j|/2 w.r.t. c_i is d/dist, w.r.t. c_j is -d/dist
            grad[i] += d / dist
            grad[j] -= d / dist
            # We want to INCREASE the distance, so move in direction of gradient
            step = 0.001
            centers[i] += step * grad[i]
            centers[j] += step * grad[j]
        elif min_type[0] == 'boundary':
            i, k = min_type[1], min_type[2]
            # Move center i away from boundary k
            if centers[i, k] > 0:
                centers[i, k] -= 0.001
            else:
                centers[i, k] += 0.001
        
        # Clamp
        centers = np.clip(centers, -0.499, 0.499)
        
        new_r = _compute_radius(centers)
        if new_r > current_r:
            current_r = new_r
        elif abs(new_r - current_r) < tol and it > 100:
            break
        else:
            # Revert if it got worse
            centers = np.clip(centers - 0.001 * (grad if min_type[0] == 'pair' else 0), -0.499, 0.499)
            # Actually let's just keep it simple: if worse, revert the move
            break
    
    return centers, current_r


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRACTION
    
    # Initial guess: hexagonal-ish grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Step 1: Force-directed layout
    centers = _force_layout(centers, n_iters=3000, dt=0.0005)
    r = _compute_radius(centers)
    
    if time.time() > deadline:
        return centers
    
    # Step 2: Simulated annealing
    centers, r = _simulated_annealing(centers, n_iters=15000, t_start=0.03, t_end=0.0005, seed=42)
    
    if time.time() > deadline:
        return centers
    
    # Step 3: Polish
    centers, r = _polish(centers, n_iters=300)
    
    # Final check: ensure all centers are within the square
    centers = np.clip(centers, -0.499999999, 0.499999999)
    
    return centers
