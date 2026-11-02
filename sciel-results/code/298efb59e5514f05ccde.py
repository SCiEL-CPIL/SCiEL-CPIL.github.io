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

def _build_grid(nx, ny, nz, noise_scale=0.005):
    """Build a grid of points with noise."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers[:N]

def _local_refinement_nm(centers, max_iters=200):
    """
    Perform local refinement using Nelder-Mead on the raw radius (negative).
    """
    def neg_radius(c_flat):
        c = c_flat.reshape(-1, 3)
        return -_compute_radius(c)
    
    try:
        res = minimize(
            neg_radius,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        return new_centers
    except Exception:
        return centers

def _coord_descent_minmax(centers, deadline, max_sweeps=200):
    """
    Coordinate-descent minmax refinement:
    - Find the binding pair (minimum distance) and push them apart.
    - Find wall-violating points and push them inward.
    This is a structurally different local search from gradient-based methods.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        
        c = best_centers.copy()
        improved = False
        
        # Compute all pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Find binding pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        min_dist = dists[i, j]
        
        # Push apart the binding pair
        if min_dist < 1e-10:
            # Coincident: separate randomly
            rng = np.random.default_rng(sweep)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            step = 0.001
            c[i] -= direction * step
            c[j] += direction * step
        else:
            direction = (c[i] - c[j]) / min_dist
            # Push apart by a small amount
            step = min_dist * 0.01
            c[i] += direction * step
            c[j] -= direction * step
        
        # Handle wall violations
        for k in range(N):
            for dim in range(3):
                if c[k, dim] > 0.5:
                    c[k, dim] = 0.5
                elif c[k, dim] < -0.5:
                    c[k, dim] = -0.5
        
        # Push wall-violating points inward
        wall_dists = 0.5 - np.abs(c)
        min_wall_idx = np.argmin(wall_dists)
        k = min_wall_idx // 3
        dim = min_wall_idx % 3
        if wall_dists[k, dim] < best_radius * 0.5:
            # Push inward
            sign = -np.sign(c[k, dim])
            c[k, dim] += sign * 0.001
        
        # Clip to bounds
        c = np.clip(c, -0.5, 0.5)
        
        new_radius = _compute_radius(c)
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = c
            improved = True
        else:
            # Try smaller step
            c = best_centers.copy()
            i, j = min_idx
            min_dist = dists[i, j]
            if min_dist > 1e-10:
                direction = (c[i] - c[j]) / min_dist
                step = min_dist * 0.001
                c[i] += direction * step
                c[j] -= direction * step
                c = np.clip(c, -0.5, 0.5)
                new_radius = _compute_radius(c)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = c
                    improved = True
        
        if not improved and sweep > 50:
            break
    
    return best_centers, best_radius

def _simulated_annealing(centers, deadline, initial_temp=0.01, cooling_rate=0.9995, min_temp=0.0001):
    """
    Simulated annealing to explore the configuration space.
    Uses Metropolis-Hastings moves.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    temp = initial_temp
    
    rng = np.random.default_rng(123)
    
    while time.time() < deadline and temp > min_temp:
        # Generate a random move: perturb one or a few spheres
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        new_centers = best_centers.copy()
        # Perturb positions
        step_size = temp * 0.5
        new_centers[idx] += rng.normal(0, step_size, (n_perturb, 3))
        
        # Keep within bounds
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        # Metropolis criterion
        delta = new_radius - current_radius
        if delta > 0:
            # Accept improvement
            best_centers = new_centers
            best_radius = new_radius
            current_radius = new_radius
        else:
            # Accept with probability exp(delta / temp)
            prob = np.exp(delta / temp)
            if rng.random() < prob:
                best_centers = new_centers
                current_radius = new_radius
        
        # Cool down
        temp *= cooling_rate
        
        # If we've been stuck for a while, maybe reheat slightly or break
        if current_radius < best_radius * 0.95 and temp < 0.005:
            break
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Precompute upper triangle indices for soft-min
    iu = np.triu_indices(N, k=1)
    
    # Initial guess: Grid
    best_centers = None
    best_radius = 0.0
    
    grids = [
        (5, 5, 4),
        (5, 4, 5),
        (4, 5, 5),
        (4, 4, 7),
        (7, 4, 4),
        (4, 7, 4),
    ]
    
    for nx, ny, nz in grids:
        if time.time() > deadline - 10.0:
            break
        centers = _build_grid(nx, ny, nz, noise_scale=0.005)
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    if best_centers is None:
        best_centers = _build_grid(5, 5, 4)
        best_radius = _compute_radius(best_centers)
        
    # Phase 1: Soft-min optimization with aggressive beta schedule
    # Use fewer, larger steps: 30 -> 3000 over 6 steps
    beta_schedule = [30.0, 60.0, 150.0, 400.0, 1000.0, 3000.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 25.0:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 80}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
    # Phase 2: Coordinate-descent minmax refinement
    if time.time() < deadline - 20.0:
        try:
            cd_centers, cd_radius = _coord_descent_minmax(
                best_centers,
                deadline - 20.0,
                max_sweeps=150
            )
            if cd_radius > best_radius:
                best_radius = cd_radius
                best_centers = cd_centers
        except Exception:
            pass
        
    # Phase 3: Simulated Annealing to escape local minima
    if time.time() < deadline - 15.0:
        try:
            sa_centers, sa_radius = _simulated_annealing(
                best_centers, 
                deadline - 15.0,
                initial_temp=0.005,
                cooling_rate=0.9998
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers
        except Exception:
            pass
            
    # Phase 4: Final local refinement with Nelder-Mead
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        try:
            refined = _local_refinement_nm(best_centers, max_iters=300)
            r_ref = _compute_radius(refined)
            if r_ref > best_radius:
                best_radius = r_ref
                best_centers = refined
        except Exception:
            pass
            
    # Phase 5: Final coordinate-descent polish
    if time.time() < deadline - 3.0:
        try:
            cd_centers, cd_radius = _coord_descent_minmax(
                best_centers,
                deadline - 3.0,
                max_sweeps=100
            )
            if cd_radius > best_radius:
                best_radius = cd_radius
                best_centers = cd_centers
        except Exception:
            pass
            
    # Phase 6: One more soft-min pass if time permits
    if time.time() < deadline - 2.0:
        try:
            x0 = best_centers.flatten()
            res = minimize(
                _soft_min_objective,
                x0,
                args=(10000.0, iu),
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
            
    return best_centers

# EVOLVE-BLOCK-END
