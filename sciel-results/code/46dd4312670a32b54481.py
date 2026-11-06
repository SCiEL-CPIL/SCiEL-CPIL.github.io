"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
TIME_LIMIT = 115.0  # Safety margin before 120s kill

# EVOLVE-BLOCK-START
import numpy as np

def _compute_radius(centres):
    """Compute the maximum radius for given centres."""
    n = len(centres)
    if n < 2:
        return (0.5 - np.abs(centres)).min()
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0
    r_boundary = (0.5 - np.abs(centres)).min()
    return min(r_pair, r_boundary)

def _initial_grid():
    """Create initial cubic grid packing 5x5x4 = 100."""
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid.copy()

def _perturb_grid(centres, rng, scale=0.02):
    """Apply small random perturbations to break symmetry."""
    c = centres.copy()
    c += rng.normal(0, scale, size=c.shape)
    c = np.clip(c, -0.49, 0.49)
    return c

def _relax(centres, time_limit=10.0):
    """Force-directed relaxation: repel nearest neighbors and push away from boundaries."""
    start_time = time.time()
    n = len(centres)
    c = centres.copy().astype(float)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Parameters
    step_size = 0.005
    decay = 0.995
    repulsion_cutoff = 0.25  # Only consider pairs within this distance
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Boundary forces: push away from cube walls
        for k in range(3):
            # Left wall
            viol = np.maximum(0, -0.5 + 0.01 - c[:, k])
            forces[:, k] += viol * 10
            # Right wall
            viol = np.maximum(0, c[:, k] - (0.5 - 0.01))
            forces[:, k] -= viol * 10
        
        # Pairwise repulsion: for each pair, if too close, push apart
        # Use vectorized computation
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Only repel pairs that are too close (within cutoff)
        mask = dist < repulsion_cutoff
        # Strength inversely proportional to distance
        strength = np.where(mask, 1.0 / np.maximum(dist, 0.001) ** 2, 0.0)
        
        # Net force on each point from repulsion
        for i in range(n):
            for j in range(n):
                if i != j and mask[i, j]:
                    # Force direction: from j to i
                    dir_vec = c[i] - c[j]
                    dist_ij = np.linalg.norm(dir_vec)
                    if dist_ij > 1e-10:
                        force_mag = strength[i, j] * 0.001
                        forces[i] += (dir_vec / dist_ij) * force_mag
        
        # Update positions
        c += step_size * forces
        c = np.clip(c, -0.49, 0.49)
        
        # Decay step size
        step_size *= decay
        
        # Check improvement
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    return best_c, best_r

def _refine_with_optimization(centres, time_limit=15.0):
    """Further refine using scipy optimization on the best solution."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    n = len(centres)
    c = centres.copy().astype(float)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e10
        c_local = x.reshape(n, 3)
        if not np.all(np.isfinite(c_local)):
            return 1e10
        # Penalty for being outside cube
        if np.any(np.abs(c_local) > 0.5):
            return 1e10
        r = _compute_radius(c_local)
        return -r
    
    # Try Powell method (good for non-smooth objectives)
    try:
        x0 = c.flatten()
        res = minimize(objective, x0, method='Powell',
                      options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                               'niter': 50})
        new_c = res.x.reshape(n, 3)
        new_r = _compute_radius(new_c)
        if new_r > best_r:
            best_r = new_r
            best_c = new_c
    except Exception:
        pass
    
    return best_c, best_r

def construct():
    start_time = time.time()
    
    # Start with grid initialization
    best_centres = _initial_grid()
    best_radius = _compute_radius(best_centres)
    
    # Try multiple perturbations of the grid
    rng = np.random.RandomState(42)
    
    # First, relax the unperturbed grid
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 30:
        relaxed, r = _relax(best_centres, time_limit=min(10.0, TIME_LIMIT - elapsed - 20))
        if r > best_radius:
            best_radius = r
            best_centres = relaxed
    
    # Try several perturbed grids
    for trial in range(8):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 25:
            break
        
        # Create perturbed version
        scale = 0.01 + 0.01 * trial
        perturbed = _perturb_grid(_initial_grid(), rng, scale=scale)
        
        # Relax
        remaining = TIME_LIMIT - elapsed - 10
        if remaining < 2:
            break
        relaxed, r = _relax(perturbed, time_limit=min(5.0, remaining))
        
        if r > best_radius:
            best_radius = r
            best_centres = relaxed
    
    # Final refinement with scipy optimization
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 5:
        refined, r = _refine_with_optimization(best_centres, time_limit=min(10.0, TIME_LIMIT - elapsed - 2))
        if r > best_radius:
            best_radius = r
            best_centres = refined
    
    # Ensure validity: clip to cube with small margin
    best_centres = np.clip(best_centres, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centres
# EVOLVE-BLOCK-END
