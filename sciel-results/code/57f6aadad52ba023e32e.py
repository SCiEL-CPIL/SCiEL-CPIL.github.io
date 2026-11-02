"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

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
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_initial_grid(nx, ny, nz, noise_scale=0.005, seed=42):
    """Build a grid initialization with small noise to break symmetry."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add small noise to break perfect symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init(noise_scale=0.005, seed=42):
    """
    Build a BCC-inspired initialization.
    Creates a dense grid of BCC-like points and selects 100.
    """
    # Create a fine grid and add body-center offsets
    nx, ny, nz = 5, 5, 4
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add body-center offsets
    dx = xs[1] - xs[0] if nx > 1 else 0.1
    dy = ys[1] - ys[0] if ny > 1 else 0.1
    dz = zs[1] - zs[0] if nz > 1 else 0.1
    
    all_centers = [grid_centers]
    
    # Body centers for interior cells
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                bc = np.array([
                    (xs[i] + xs[i+1]) / 2,
                    (ys[j] + ys[j+1]) / 2,
                    (zs[k] + zs[k+1]) / 2
                ])
                all_centers.append(bc.reshape(1, 3))
    
    all_centers = np.vstack(all_centers)
    
    rng = np.random.default_rng(seed)
    if len(all_centers) >= N:
        idx = rng.choice(len(all_centers), N, replace=False)
        centers = all_centers[idx]
    else:
        centers = all_centers
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(seed=42):
    """
    Build a Halton sequence based initialization for good space-filling properties.
    """
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    centers = np.zeros((N, 3))
    for i in range(1, N + 1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _coordinate_descent_minmax(centers, deadline, max_iters=2000, step_size=0.005):
    """
    Coordinate-descent minmax refinement.
    Directly pushes apart the binding pair and inward wall-violating points.
    This provides a structurally different local search that can escape
    the soft-min L-BFGS basin.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    iu = np.triu_indices(N, k=1)
    rng = np.random.default_rng(123)
    
    n_done = 0
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current constraints
        dists_to_walls = 0.5 - np.abs(current_centers)
        min_wall_idx = np.argmin(dists_to_walls)
        min_wall_val = np.min(dists_to_walls)
        
        dists = cdist(current_centers, current_centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        # The binding constraint is the minimum of wall and pair
        if min_wall_val <= min_pair_val / 2.0:
            # Wall is binding: push inward
            i = min_wall_idx // 3
            k = min_wall_idx % 3
            current_centers[i, k] += step_size * np.sign(current_centers[i, k])
        else:
            # Pair is binding: push apart
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            diff = current_centers[i] - current_centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-12:
                direction = diff / norm
                current_centers[i] += step_size * 0.5 * direction
                current_centers[j] -= step_size * 0.5 * direction
        
        # Clip to valid range
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(current_centers)
        
        if new_radius >= current_radius:
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = current_centers.copy()
        else:
            # Revert if worse
            current_centers = best_centers.copy()
            current_radius = best_radius
        
        # Decrease step size
        step_size *= 0.995
        
        n_done += 1
    
    return best_centers, best_radius

def _anneal(centers, deadline, T_start=0.005, T_end=0.00001, n_steps=2000):
    """
    Simulated annealing to explore the configuration space.
    Uses Metropolis-Hastings with energy = -radius (we want to maximize radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_centers = centers.copy()
    
    iu = np.triu_indices(N, k=1)
    rng = np.random.default_rng(42)
    
    n_done = 0
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Annealing schedule
        progress = step / max(1, n_steps - 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Random perturbation
        idx = rng.integers(N)
        # Smaller step size than typical to stay in local neighborhood but explore
        delta = rng.normal(0, 0.005, 3)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
        
        # Compute new radius
        new_radius = _compute_radius(new_centers)
        
        # Metropolis criterion
        if new_radius >= current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            delta_E = current_radius - new_radius  # positive means worse
            if delta_E < 1e-15:
                accept = True
            else:
                accept = np.log(rng.uniform()) > -delta_E / T
            if accept:
                current_centers = new_centers
                current_radius = new_radius
        
        n_done += 1
    
    return best_centers, best_radius

def _optimize_from_initial(centers, deadline, beta_start=30.0, beta_end=3000.0, n_beta_steps=6):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    # Create beta schedule: geometric progression from beta_start to beta_end
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    # Phase 1: L-BFGS-B with increasing beta
    for beta in betas:
        if time.time() > deadline - 20.0: # Leave time for other phases
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            # Check if we improved
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 20.0:
            break
    
    # Phase 2: Coordinate Descent Min-Max Refinement
    if time.time() < deadline - 15.0:
        try:
            cd_centers, cd_radius = _coordinate_descent_minmax(
                best_centers, deadline - 10.0, max_iters=1500, step_size=0.005
            )
            if cd_radius > best_radius:
                best_radius = cd_radius
                best_centers = cd_centers
        except Exception:
            pass
    
    # Phase 3: Simulated Annealing
    if time.time() < deadline - 10.0:
        try:
            # Use a smaller number of steps for annealing to fit in time
            # Adjust n_steps based on remaining time? Fixed 1000 steps is ~0.1s-0.2s usually
            an_centers, an_radius = _anneal(
                best_centers, deadline - 5.0, T_start=0.002, T_end=0.00001, n_steps=1000
            )
            if an_radius > best_radius:
                best_radius = an_radius
                best_centers = an_centers
        except Exception:
            pass
            
    # Phase 4: Final Polish Pass with very high beta
    if time.time() < deadline - 2.0:
        try:
            x0 = best_centers.flatten()
            res = minimize(
                _soft_min_objective,
                x0,
                args=(20000.0, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
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

def construct():
    start_time = time.time()
    # Use 145s deadline to leave safety margin for the 150s limit
    deadline = start_time + 145.0
    
    # Try multiple different initializations
    initializers = []
    
    # Grid configurations
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
    ]
    
    for nx, ny, nz in grid_configs:
        for noise_scale in [0.003, 0.005]:
            initializers.append(_build_initial_grid(nx, ny, nz, noise_scale))
    
    # BCC initialization
    initializers.append(_build_bcc_init(noise_scale=0.005))
    
    # Halton initialization
    initializers.append(_build_halton_init())
    
    best_centers = None
    best_radius = 0.0
    
    for centers in initializers:
        if time.time() > deadline - 20.0:
            break
        
        r_init = _compute_radius(centers)
        
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
        
        # Optimize from this initial
        opt_centers, opt_radius = _optimize_from_initial(
            centers, deadline,
            beta_start=30.0, beta_end=3000.0, n_beta_steps=6
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # If no valid result, fallback to simple grid
    if best_centers is None:
        best_centers = _build_initial_grid(5, 5, 4, 0.01)
    
    return best_centers
# EVOLVE-BLOCK-END
