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
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
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
    
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
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
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n=100, seed=42):
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
    
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n + 1):
        centers[i - 1, 0] = -0.5 + halton(i, 2)
        centers[i - 1, 1] = -0.5 + halton(i, 3)
        centers[i - 1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init(n=100, seed=42):
    """
    Build a BCC-inspired initialization.
    Creates a 4x5x5 grid with body-center points, selects 100.
    """
    nx, ny, nz = 4, 5, 5
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    all_centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                all_centers.append([xs[i], ys[j], zs[k]])
                if i < nx - 1 and j < ny - 1 and k < nz - 1:
                    all_centers.append([(xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2, (zs[k] + zs[k + 1]) / 2])
                elif i < nx - 1 and j < ny - 1:
                    all_centers.append([(xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2, zs[k]])
                elif i < nx - 1:
                    all_centers.append([(xs[i] + xs[i + 1]) / 2, ys[j], zs[k]])
                elif j < ny - 1:
                    all_centers.append([xs[i], (ys[j] + ys[j + 1]) / 2, zs[k]])
    
    all_centers = np.array(all_centers)
    rng = np.random.default_rng(seed)
    
    if len(all_centers) >= n:
        idx = rng.choice(len(all_centers), n, replace=False)
        centers = all_centers[idx]
    else:
        centers = all_centers
        while len(centers) < n:
            centers = np.vstack([centers, centers[0]])
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _simulated_anneal(centers, deadline, T_start=0.008, T_end=0.00005, max_steps=3000):
    """
    Simulated annealing to escape local minima.
    Uses Metropolis-Hastings with energy = -radius.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    rng = np.random.default_rng(12345)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(1, max_steps - 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Move one random center
        idx = rng.integers(N)
        delta = rng.normal(0, 0.008, 3)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius >= current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            delta_E = current_radius - new_radius
            if delta_E < 1e-15:
                accept = True
            else:
                accept = np.log(rng.uniform()) > -delta_E / T
            if accept:
                current_centers = new_centers
                current_radius = new_radius
    
    return best_centers, best_radius

def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=6):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
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
                options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
    
    return best_centers, best_radius

def _high_precision_polish(centers, deadline):
    """
    Final high-precision polish pass using very high beta and tight tolerances.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    
    # Try several high-beta values with increasing precision
    for beta in [10000.0, 50000.0, 200000.0]:
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
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
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
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Build diverse initial configurations
    initial_configs = []
    
    # Grid configurations: 5x5x4, 4x5x5, 5x4x5, 10x5x2
    grid_configs = [(5, 5, 4), (4, 5, 5), (5, 4, 5), (10, 5, 2)]
    noise_scales = [0.003, 0.005, 0.01]
    
    for nx, ny, nz in grid_configs:
        for ns in noise_scales:
            seed = hash((nx, ny, nz, ns)) % 2**32
            centers = _build_initial_grid(nx, ny, nz, ns, seed)
            initial_configs.append(centers)
    
    # Halton sequence
    initial_configs.append(_build_halton_init(100, 42))
    initial_configs.append(_build_halton_init(100, 123))
    
    # BCC-inspired
    initial_configs.append(_build_bcc_init(100, 42))
    initial_configs.append(_build_bcc_init(100, 999))
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Quick SA + L-BFGS-B on each initial config
    for i, centers in enumerate(initial_configs):
        if time.time() > deadline - 20.0:
            break
        
        # Quick SA to escape local minima
        sa_centers, sa_radius = _simulated_anneal(centers, deadline - 15.0, 
                                                   T_start=0.005, T_end=0.0001, 
                                                   max_steps=800)
        
        if sa_radius > best_radius:
            best_radius = sa_radius
            best_centers = sa_centers.copy()
        
        # L-BFGS-B optimization
        opt_centers, opt_radius = _optimize_from_initial(
            sa_centers, deadline - 15.0,
            beta_start=50.0, beta_end=5000.0, n_beta_steps=5
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # Phase 2: High-precision polish on the best configuration
    if best_centers is not None and time.time() < deadline - 10.0:
        polished_centers, polished_radius = _high_precision_polish(best_centers, deadline - 5.0)
        
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers
    
    # Fallback if no valid result
    if best_centers is None:
        best_centers = _build_initial_grid(5, 5, 4, 0.01, 42)
    
    return best_centers

# EVOLVE-BLOCK-END
