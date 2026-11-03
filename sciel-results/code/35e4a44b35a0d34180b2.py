"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) is None or len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return float(min_dist_to_wall)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _repulsion_energy(centers, r_target=0.11):
    """
    Compute repulsion energy: sum of 1/d^2 for pairs closer than 2*r_target,
    plus wall violations. Lower is better.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Only count pairs that are too close
    too_close = sq_dists < (2.0 * r_target)**2
    if np.any(too_close):
        pair_energy = np.sum(1.0 / sq_dists[too_close])
    else:
        pair_energy = 0.0
    
    # Wall violations: penalty for being outside or too close to walls
    wall_dist = 0.5 - np.abs(centers)
    wall_violation = np.sum(np.maximum(0, r_target - wall_dist)**2)
    
    return pair_energy + 100.0 * wall_violation

def _simulated_annealing(centers, r_target, T_init=0.05, T_final=0.0001, 
                          max_steps=50000, time_limit=30.0, seed=42):
    """
    Simulated annealing to maximize minimum radius.
    Uses repulsion energy as the objective to minimize.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    centers = centers.copy()
    current_energy = _repulsion_energy(centers, r_target)
    best_centers = centers.copy()
    best_energy = current_energy
    best_radius = _compute_radius(centers)
    
    step_size = 0.02
    cooling = (T_final / T_init) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        T = T_init * (cooling ** step)
        
        # Random perturbation
        idx = rng.randint(N)
        delta = rng.normal(0, step_size, size=3)
        new_centers = centers.copy()
        new_centers[idx] = np.clip(new_centers[idx] + delta, -0.4999, 0.4999)
        
        new_energy = _repulsion_energy(new_centers, r_target)
        delta_E = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_E/T)
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
            centers = new_centers
            current_energy = new_energy
            
            r_new = _compute_radius(centers)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = centers.copy()
                # Reduce step size as we improve
                step_size *= 0.999
        
        # Occasionally try larger moves to escape local minima
        if step % 100 == 0:
            idx1 = rng.randint(N)
            idx2 = rng.randint(N)
            if idx1 != idx2:
                delta1 = rng.normal(0, step_size * 2, size=3)
                delta2 = -delta1  # Keep center of mass roughly fixed
                new_centers = centers.copy()
                new_centers[idx1] = np.clip(new_centers[idx1] + delta1, -0.4999, 0.4999)
                new_centers[idx2] = np.clip(new_centers[idx2] + delta2, -0.4999, 0.4999)
                
                new_energy = _repulsion_energy(new_centers, r_target)
                delta_E = new_energy - current_energy
                
                if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
                    centers = new_centers
                    current_energy = new_energy
                    r_new = _compute_radius(centers)
                    if r_new > best_radius:
                        best_radius = r_new
                        best_centers = centers.copy()
    
    return best_centers, best_radius

def _gradient_polish(centers, max_iter=2000, time_limit=15.0):
    """
    Final gradient-based polish using L-BFGS on a smooth repulsion objective.
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        c = x.reshape(N, 3)
        # Smooth repulsion: sum of exp(-d^2/sigma^2) for close pairs
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        sigma = 0.05
        weights = np.exp(-sq_dists / sigma**2)
        np.fill_diagonal(weights, 0.0)
        
        energy = np.sum(weights)
        
        # Wall penalty
        wall_dist = 0.5 - np.abs(c)
        wall_penalty = np.sum(np.exp(-wall_dist**2 / (sigma/2)**2))
        
        total = energy + 10.0 * wall_penalty
        
        # Gradient
        grad_c = np.zeros_like(c)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = weights[i, j]
                if w > 1e-10:
                    d = c[i] - c[j]
                    grad_c[i] -= 2.0 * w * d / sigma**2
                    grad_c[j] += 2.0 * w * d / sigma**2
            
            # Wall gradient
            for k in range(3):
                wd = wall_dist[i, k]
                wp = np.exp(-wd**2 / (sigma/2)**2)
                if wp > 1e-10:
                    sign = np.sign(c[i, k])
                    if sign == 0:
                        sign = 1.0
                    grad_c[i, k] += 10.0 * wp * 2.0 * wd * sign / (sigma/2)**2
        
        grad = grad_c.flatten()
        return total, grad
    
    try:
        result = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return centers

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _jitter_init(n, seed=None, jitter=0.01):
    """Generate jittered grid initial configuration to break symmetry."""
    grid = _grid_init(n)
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-jitter, jitter, size=grid.shape)
    return np.clip(grid + noise, -0.49, 0.49)

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Stage 0: Initial grid
    grid = _grid_init(N)
    r_grid = _compute_radius(grid)
    best_radius = r_grid
    best_centers = grid.copy()
    
    # Stage 1: Simulated annealing from multiple starts
    # Target radius slightly above current best to guide the search
    r_target = max(best_radius * 1.05, 0.10)
    
    inits = [grid]
    for seed in range(3):
        inits.append(_jitter_init(N, seed=seed, jitter=0.03))
    for seed in range(2):
        inits.append(_random_init(N, seed=seed))
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 50.0:
            break
        
        time_remaining = deadline - time.time() - 40.0
        if time_remaining < 5.0:
            break
        
        try:
            # Use a portion of the remaining time for SA
            sa_time = min(25.0, time_remaining * 0.6)
            sa_centers, sa_radius = _simulated_annealing(
                init, r_target, 
                T_init=0.03, T_final=0.00005,
                max_steps=30000, 
                time_limit=sa_time,
                seed=100 + init_idx
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers.copy()
                # Update target for subsequent runs
                r_target = max(best_radius * 1.02, 0.105)
        except Exception:
            continue
    
    # Stage 2: Gradient polish from best SA result
    if time.time() < deadline - 20.0:
        try:
            time_remaining = deadline - time.time() - 10.0
            if time_remaining > 3.0:
                polished = _gradient_polish(best_centers, max_iter=1500, 
                                             time_limit=min(15.0, time_remaining))
                r_polished = _compute_radius(polished)
                if r_polished > best_radius:
                    best_radius = r_polished
                    best_centers = polished.copy()
        except Exception:
            pass
    
    # Stage 3: More SA rounds with tighter target, starting from polished result
    if time.time() < deadline - 15.0:
        r_target = max(best_radius * 1.01, 0.11)
        for seed in range(3):
            if time.time() > deadline - 8.0:
                break
            
            time_remaining = deadline - time.time() - 5.0
            if time_remaining < 3.0:
                break
            
            try:
                sa_time = min(8.0, time_remaining * 0.7)
                sa_centers, sa_radius = _simulated_annealing(
                    best_centers, r_target,
                    T_init=0.01, T_final=0.00002,
                    max_steps=15000,
                    time_limit=sa_time,
                    seed=200 + seed
                )
                if sa_radius > best_radius:
                    best_radius = sa_radius
                    best_centers = sa_centers.copy()
            except Exception:
                continue
    
    # Stage 4: Final gradient polish
    if time.time() < deadline - 5.0:
        try:
            time_remaining = deadline - time.time() - 2.0
            if time_remaining > 2.0:
                polished = _gradient_polish(best_centers, max_iter=1000,
                                             time_limit=min(5.0, time_remaining))
                r_polished = _compute_radius(polished)
                if r_polished > best_radius:
                    best_radius = r_polished
                    best_centers = polished.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
# EVOLVE-BLOCK-END
