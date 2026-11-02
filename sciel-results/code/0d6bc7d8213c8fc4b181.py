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
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))

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
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid(nx, ny, nz, noise_scale=0.005, seed=42):
    """Build a grid of points with noise."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers[:N]

def _build_bcc_init(n_per_axis=4, seed=42):
    """
    Build a BCC (body-centered cubic) initialization.
    BCC has better packing density than simple cubic.
    """
    # Create a BCC lattice: corners + body centers
    margin = 0.02
    step = (1.0 - 2*margin) / (n_per_axis - 1) if n_per_axis > 1 else 1.0
    
    centers_list = []
    for i in range(n_per_axis):
        for j in range(n_per_axis):
            for k in range(n_per_axis):
                # Corner
                cx = -0.5 + margin + i * step
                cy = -0.5 + margin + j * step
                cz = -0.5 + margin + k * step
                centers_list.append([cx, cy, cz])
                
                # Body center (only if not on boundary)
                if i < n_per_axis - 1 and j < n_per_axis - 1 and k < n_per_axis - 1:
                    bx = -0.5 + margin + (i + 0.5) * step
                    by = -0.5 + margin + (j + 0.5) * step
                    bz = -0.5 + margin + (k + 0.5) * step
                    centers_list.append([bx, by, bz])
    
    centers = np.array(centers_list)
    
    # Add small noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    # Take first N points
    return centers[:N]

def _build_halton_init(n=100, seed=42):
    """
    Build a Halton sequence initialization for better space-filling.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        points.append([x - 0.5, y - 0.5, z - 0.5])
    
    centers = np.array(points)
    
    # Add small noise
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _optimize_from_initial(centers, deadline, iu):
    """
    Optimize from a given initial configuration using soft-min with increasing beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    beta = 10.0
    max_beta = 10000.0
    
    while time.time() < deadline - 10.0 and beta < max_beta:
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
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
        # Increase beta
        beta *= 1.4
        
    return best_centers, best_radius

def _simulated_annealing(centers, deadline, iu, initial_temp=0.01, cooling_rate=0.999, min_temp=1e-5):
    """
    Simulated Annealing to escape local minima.
    Uses soft-min objective with temperature.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_centers = best_centers.copy()
    
    temp = initial_temp
    rng = np.random.default_rng(42)
    
    # Precompute indices for efficiency
    # We will use a fixed beta for SA to keep it simple and fast, 
    # or a beta that scales with temp. Let's use beta = 1/temp for the soft min.
    
    steps = 0
    max_steps = 5000 # Hard cap to prevent infinite loops if time check fails
    
    while time.time() < deadline and temp > min_temp and steps < max_steps:
        # Generate a random perturbation
        # Scale perturbation with temperature
        scale = 0.01 * temp / initial_temp
        perturbation = rng.normal(0, scale, (N, 3))
        new_centers = current_centers + perturbation
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        # Compute new radius
        new_radius = _compute_radius(new_centers)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta > 0:
            # Always accept improvement
            current_radius = new_radius
            current_centers = new_centers
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            # Accept worse solution with probability exp(delta/temp)
            # Note: delta is negative
            prob = np.exp(delta / temp)
            if rng.random() < prob:
                current_radius = new_radius
                current_centers = new_centers
        
        # Cool down
        temp *= cooling_rate
        steps += 1
        
        # Periodic local refinement to help escape
        if steps % 100 == 0 and time.time() < deadline - 5.0:
             # Quick L-BFGS polish
             try:
                 res = minimize(
                     _soft_min_objective,
                     current_centers.flatten(),
                     args=(100.0, iu),
                     method='L-BFGS-B',
                     bounds=[(-0.5, 0.5)] * (N * 3),
                     options={'maxiter': 50, 'ftol': 1e-10}
                 )
                 polished = res.x.reshape(-1, 3)
                 polished = np.clip(polished, -0.5, 0.5)
                 r_pol = _compute_radius(polished)
                 if r_pol > best_radius:
                     best_radius = r_pol
                     best_centers = polished.copy()
                     current_centers = polished.copy()
                     current_radius = r_pol
             except:
                 pass

    return best_centers, best_radius

def _polish(centers, deadline, iu):
    """
    High-precision polish pass with very high beta and tight tolerances.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Use very high beta for near-exact min
    for beta in [5000.0, 10000.0, 20000.0, 50000.0]:
        if time.time() > deadline:
            break
            
        x0 = best_centers.flatten()
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
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
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    # Generate multiple initial configurations
    best_centers = None
    best_radius = 0.0
    
    # Try different initialization strategies
    inits = []
    
    # Grid initializations
    grids = [
        (5, 5, 4),
        (5, 4, 5),
        (4, 5, 5),
    ]
    for nx, ny, nz in grids:
        inits.append(_build_grid(nx, ny, nz, noise_scale=0.005, seed=42))
    
    # BCC initialization
    inits.append(_build_bcc_init(n_per_axis=4, seed=42))
    inits.append(_build_bcc_init(n_per_axis=5, seed=43))
    
    # Halton initialization
    inits.append(_build_halton_init(n=N, seed=44))
    
    # Evaluate all initial configurations
    for centers in inits:
        if time.time() > deadline - 30.0:
            break
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    if best_centers is None:
        best_centers = _build_grid(5, 5, 4)
        best_radius = _compute_radius(best_centers)
    
    # Phase 1: Initial Local Optimization
    if time.time() < deadline - 40.0:
        opt_centers, opt_radius = _optimize_from_initial(
            best_centers,
            deadline - 40.0,
            iu
        )
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # Phase 2: Simulated Annealing to escape local minima
    if time.time() < deadline - 20.0:
        sa_centers, sa_radius = _simulated_annealing(
            best_centers,
            deadline - 20.0,
            iu,
            initial_temp=0.05,
            cooling_rate=0.998
        )
        if sa_radius > best_radius:
            best_radius = sa_radius
            best_centers = sa_centers
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 5.0:
        polish_centers, polish_radius = _polish(
            best_centers,
            deadline - 2.0,
            iu
        )
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
    
    return best_centers

# EVOLVE-BLOCK-END
