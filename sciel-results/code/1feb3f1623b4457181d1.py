"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Deadline guard
START_TIME = time.time()
DEADLINE = 145.0  # 5s safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    # Distance between all pairs
    d = cdist(centres, centres)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = d.min()
    
    # Distance to walls
    # Cube is [-0.5, 0.5]^3
    wall_dists = []
    for k in range(3):
        wall_dists.append(0.5 - np.abs(centres[:, k]))
    min_wall_dist = min(wd.min() for wd in wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_objective(x, beta=50.0):
    """
    Smooth surrogate for the negative packing radius.
    We minimize -r, but use a soft-min to make it differentiable.
    
    r = min(min_pair_dist/2, min_wall_dist)
    -r = -min(...) = max(-min_pair_dist/2, -min_wall_dist)
    
    Actually, let's think about this more carefully.
    We want to maximize r = min(d_min/2, w_min)
    So we want to minimize -r = -min(d_min/2, w_min)
    
    -min(a, b) = max(-a, -b)
    
    So we need a soft-max of the "badness" terms.
    
    Let's define:
    For each pair (i,j): badness_pair = -d_ij/2
    For each centre i and wall k: badness_wall = -(0.5 - |c_ik|)
    
    Then -r ≈ soft_max(all badnesses)
    
    soft_max(x) = (1/beta) * log(sum(exp(beta * x_i)))
    
    But this has numerical issues. Let's use a different formulation.
    
    Alternative: minimize a smooth approximation of the constraint violation.
    We want all pair distances >= 2r and all wall distances >= r.
    
    Let's use a penalty-based approach with smooth min.
    """
    centres = x.reshape(N, 3)
    
    # Pairwise distances
    d = cdist(centres, centres)
    np.fill_diagonal(d, np.inf)
    
    # Wall distances
    wall_dists = []
    for k in range(3):
        wall_dists.append(0.5 - np.abs(centres[:, k]))
    wall_dists = np.concatenate(wall_dists)
    
    # We want to maximize the minimum of (d_ij/2) and (wall_dist)
    # Let's compute a smooth version of the minimum
    
    # Collect all "radius candidates"
    pair_radii = d[d < np.inf] / 2.0
    all_radii = np.concatenate([pair_radii, wall_dists])
    
    # Soft min using log-sum-exp trick
    # soft_min(x) = -soft_max(-x)
    # soft_max(x) = (1/beta)*log(sum(exp(beta*x_i)))
    
    # To avoid overflow, subtract max
    neg_all = -all_radii
    max_neg = np.max(neg_all)
    if max_neg > 50:  # avoid overflow
        exp_vals = np.exp(beta * (neg_all - max_neg))
    else:
        exp_vals = np.exp(beta * neg_all)
    
    log_sum = np.log(np.sum(exp_vals))
    soft_max_val = (1.0 / beta) * (log_sum + beta * max_neg)
    soft_min_val = -soft_max_val
    
    # Return negative of soft min (we want to minimize -r)
    return -soft_min_val

def _gradient(x, beta=50.0):
    """Compute gradient of the soft-min objective via finite differences or analytic."""
    # Use numerical gradient for simplicity, but this is slow.
    # Let's try analytic gradient.
    
    centres = x.reshape(N, 3)
    
    d = cdist(centres, centres)
    np.fill_diagonal(d, np.inf)
    
    wall_dists = []
    for k in range(3):
        wall_dists.append(0.5 - np.abs(centres[:, k]))
    wall_dists = np.concatenate(wall_dists)
    
    pair_radii = d[d < np.inf] / 2.0
    all_radii = np.concatenate([pair_radii, wall_dists])
    
    neg_all = -all_radii
    max_neg = np.max(neg_all)
    exp_vals = np.exp(beta * (neg_all - max_neg))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_max(-all_radii) w.r.t. all_radii is -weights
    # So gradient of -soft_min = -soft_max(-all_radii) w.r.t. all_radii is weights
    # Wait, let me redo this.
    
    # f = -soft_min(all_radii) = soft_max(-all_radii) = (1/beta)*log(sum(exp(beta*(-all_radii))))
    # df/d(all_radii_i) = (1/beta) * exp(beta*(-all_radii_i)) / sum(exp(beta*(-all_radii_j))) * (-beta)
    #                    = -exp(beta*(-all_radii_i)) / sum(exp(beta*(-all_radii_j)))
    #                    = -weights_i
    
    # So df/d(all_radii) = -weights
    
    # Now we need df/dx.
    # all_radii consists of pair_radii and wall_dists.
    
    grad = np.zeros_like(x)
    
    # Pair contributions
    # pair_radii[i] = d_ij/2 where i is the flat index into the upper triangle
    # This is complex. Let's just use numerical gradient for now.
    
    return None  # Fallback to numerical

def _construct_bcc_init():
    """Body-centered cubic lattice initialization."""
    # BCC lattice: corners + body centers
    # In a unit cube, BCC has 8 corners and 1 body center per unit cell.
    # But we need to fit 100 spheres.
    
    # Let's try a different approach: start with a grid and add perturbations.
    m = 5  # 5x5x5 = 125 positions, take 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add small random perturbations to break symmetry
    np.random.seed(42)
    perturbation = np.random.randn(len(grid), 3) * 0.01
    grid = grid + perturbation
    
    # Clip to stay within bounds
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid[:N]

def _construct_fcc_init():
    """Face-centered cubic lattice initialization."""
    # FCC has 4 atoms per unit cell.
    # Let's create a lattice of FCC unit cells.
    
    # FCC basis vectors in a unit cell of side a:
    # (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    
    # For 100 spheres, we need about 25 unit cells.
    # Let's use a 3x3x3 grid of FCC cells = 27 cells * 4 = 108 spheres, take 100.
    
    n_cells = 3
    a = 1.0 / n_cells  # cell size
    
    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                base = np.array([ix, iy, iz]) * a
                # FCC positions in cell
                fcc_pos = [
                    np.array([0, 0, 0]),
                    np.array([0.5, 0.5, 0]) * a,
                    np.array([0.5, 0, 0.5]) * a,
                    np.array([0, 0.5, 0.5]) * a,
                ]
                for fp in fcc_pos:
                    positions.append(base + fp)
    
    positions = np.array(positions)
    
    # Center the whole thing
    positions = positions - 0.5  # shift to [-0.5, 0.5]
    
    # Add small perturbations
    np.random.seed(42)
    perturbation = np.random.randn(len(positions), 3) * 0.005
    positions = positions + perturbation
    
    # Clip
    positions = np.clip(positions, -0.49, 0.49)
    
    return positions[:N]

def _optimize_from_init(init_centres, beta_start=20.0, beta_end=100.0, max_iter=500):
    """Run L-BFGS-B optimization from given initialization."""
    x0 = init_centres.flatten()
    
    best_x = x0.copy()
    best_r = _compute_radius(init_centres)
    
    # Anneal beta
    n_anneal_steps = 10
    betas = np.linspace(beta_start, beta_end, n_anneal_steps)
    
    for beta in betas:
        if _time_left() < 10:
            break
            
        result = minimize(
            _soft_min_objective,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-0.499, 0.499)] * (N * 3),
            options={'maxiter': max(50, int(200 * _time_left() / 60)), 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < _soft_min_objective(best_x, beta):
            # Check if this is actually better in terms of true radius
            new_centres = result.x.reshape(N, 3)
            new_r = _compute_radius(new_centres)
            if new_r > best_r:
                best_r = new_r
                best_x = result.x.copy()
        
        x0 = result.x
    
    return best_x.reshape(N, 3), best_r

def construct():
    """Main entry point."""
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 145.0  # 5s safety margin
    
    best_centres = None
    best_r = -np.inf
    
    # Try multiple initializations
    inits = [
        _construct_bcc_init(),
        _construct_fcc_init(),
    ]
    
    # Also try a random initialization
    np.random.seed(123)
    random_init = np.random.uniform(-0.45, 0.45, (N, 3))
    inits.append(random_init)
    
    for i, init in enumerate(inits):
        if _time_left() < 20:
            break
        
        centres, r = _optimize_from_init(init)
        
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
    
    # Final polish with tighter tolerance
    if best_centres is not None and _time_left() > 15:
        x0 = best_centres.flatten()
        result = minimize(
            _soft_min_objective,
            x0,
            args=(200.0,),
            method='L-BFGS-B',
            bounds=[(-0.499, 0.499)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        new_centres = result.x.reshape(N, 3)
        new_r = _compute_radius(new_centres)
        
        if new_r > best_r:
            best_r = new_r
            best_centres = new_centres.copy()
    
    if best_centres is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N]
    
    # Ensure centres are strictly inside the cube
    best_centres = np.clip(best_centres, -0.499, 0.499)
    
    return best_centres
# EVOLVE-BLOCK-END
