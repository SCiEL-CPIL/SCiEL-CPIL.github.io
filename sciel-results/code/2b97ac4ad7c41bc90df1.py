import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.stats import qmc

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
        
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    # Using cdist is efficient for N=100
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    """Build a grid initialization."""
    # Ensure we get exactly N points
    # For N=100, 5x5x4 is a good start
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        # Pad or truncate
        if len(centers) < N:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
        else:
            centers = centers[:N]

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_sobol_init(seed=42):
    """
    Generate N points using a Sobol sequence (low discrepancy).
    """
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    points = sampler.random(N)
    # Map from [0,1]^3 to [-0.5, 0.5]^3
    centers = points - 0.5
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_bcc_init(seed=42):
    """
    Generate N points using a body-centered cubic (BCC) lattice initialization.
    BCC has better local packing density than simple cubic.
    """
    # BCC unit cell: 8 corners + 1 center
    # For a cube of side 1, we can tile with BCC cells
    # Each BCC cell has 2 atoms (corners shared, center unique)
    # We need 100 points, so ~50 BCC cells
    
    # Start with a grid of BCC unit cells
    # Each BCC unit cell in our coordinate system:
    # Corners at (0,0,0), (1,0,0), etc. and center at (0.5, 0.5, 0.5)
    # Scale to fit in [-0.5, 0.5]^3
    
    # Use a 4x4x4 grid of BCC cells (64 cells * 2 points = 128 points, take 100)
    nx = ny = nz = 4
    cell_size = 1.0 / max(nx, ny, nz)
    
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corner of this cell
                cx = -0.5 + i * cell_size
                cy = -0.5 + j * cell_size
                cz = -0.5 + k * cell_size
                
                # Add corner point (only if it's a unique corner in the tiling)
                # For simplicity, add all corners and centers, then deduplicate
                points.append([cx, cy, cz])
                # Add center point
                points.append([cx + cell_size/2, cy + cell_size/2, cz + cell_size/2])
    
    points = np.array(points)
    
    # Deduplicate points that are very close (corners shared between cells)
    if len(points) > 0:
        tree = cKDTree(points)
        pairs = tree.query_pairs(r=cell_size * 0.1)
        # Keep only unique points
        keep = np.ones(len(points), dtype=bool)
        for i, j in pairs:
            if keep[j]:
                keep[j] = False
        points = points[keep]
    
    # If we have more than N points, take the first N
    # If fewer, pad with random
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        rng = np.random.default_rng(seed)
        extra = rng.uniform(-0.45, 0.45, (N - len(points), 3))
        points = np.vstack([points, extra])
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    points += rng.normal(0, 0.005, points.shape)
    points = np.clip(points, -0.499, 0.499)
    
    return points

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
    # Only use upper triangle to avoid duplicates, but we need indices
    # iu is precomputed indices of upper triangle
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # This is the log-sum-exp of -beta * x, which approximates min(x) as beta -> inf
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _simulated_annealing(centers, iu, time_budget=10.0, initial_temp=0.05):
    """
    Simulated annealing to escape local minima.
    Uses the soft-min objective with varying temperature.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    start_t = time.time()
    rng = np.random.default_rng(42)
    
    temp = initial_temp
    cooling_rate = 0.995
    min_temp = 0.001
    
    while _time_left() > 2.0 and (time.time() - start_t) < time_budget - 1.0 and temp > min_temp:
        # Generate a random perturbation
        perturbation = rng.normal(0, temp, centers.shape)
        new_centers = current_centers + perturbation
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_radius = _compute_radius(new_centers)
        
        # Accept if better, or with probability exp(-(delta)/temp) if worse
        if new_radius > current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            delta = current_radius - new_radius
            if delta < 1e-15:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / max(temp, 1e-10))
            if accept:
                current_centers = new_centers
                current_radius = new_radius
        
        temp *= cooling_rate
    
    return best_centers, best_radius

def _polish_high_beta(centers, iu, time_budget=10.0):
    """
    High-precision polish using very high beta soft-min.
    This extracts the last bits of radius from a good configuration.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    
    # Use very high beta values for high precision
    betas = [10000.0, 50000.0, 200000.0]
    
    current_centers = centers.copy()
    
    for beta in betas:
        if _time_left() < 2.0 or (time.time() - start_t) > time_budget - 1.0:
            break
            
        x0 = current_centers.flatten()
        
        try:
            max_iter = 200
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_radius = _compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
                current_centers = new_centers.copy()
            else:
                if np.all(np.isfinite(new_centers)):
                    current_centers = new_centers.copy()
                    
        except Exception:
            pass
    
    return best_centers, best_radius

def _optimize_from_initial(centers, deadline, iu, time_budget=20.0):
    """
    Run optimization from initial centers.
    Multi-phase: soft-min -> simulated annealing -> repulsion polish -> high-beta polish
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    
    # Phase 1: Soft-min optimization with increasing beta
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_centers = centers.copy()
    
    for beta in betas:
        if _time_left() < 2.0 or (time.time() - start_t) > time_budget * 0.5:
            break
            
        x0 = current_centers.flatten()
        
        try:
            max_iter = 100 if beta < 100 else 50
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_radius = _compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
                current_centers = new_centers.copy()
            else:
                if np.all(np.isfinite(new_centers)):
                    current_centers = new_centers.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Simulated annealing to escape local minima
    if _time_left() > 5.0 and (time.time() - start_t) < time_budget * 0.7:
        sa_time = min(5.0, time_budget * 0.2)
        centers_sa, radius_sa = _simulated_annealing(
            best_centers, iu, time_budget=sa_time
        )
        if radius_sa > best_radius:
            best_radius = radius_sa
            best_centers = centers_sa.copy()
    
    # Phase 3: Local Repulsion Polish
    if _time_left() > 3.0 and (time.time() - start_t) < time_budget * 0.85:
        current_centers = best_centers.copy()
        r_current = _compute_radius(current_centers)
        
        step_size = 0.003
        decay = 0.95
        min_step = 1e-6
        margin = 1.05
        
        no_improve_count = 0
        max_no_improve = 50
        polish_start = time.time()
        
        while _time_left() > 1.5 and (time.time() - polish_start) < time_budget * 0.15 and step_size > min_step and no_improve_count < max_no_improve:
            r = _compute_radius(current_centers)
            if r > best_radius + 1e-12:
                best_radius = r
                best_centers = current_centers.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Compute forces
            forces = np.zeros_like(current_centers)
            
            # Wall forces
            for k in range(3):
                dists = 0.5 - np.abs(current_centers[:, k])
                limit = max(r_current * margin, 0.01)
                mask = dists < limit
                if np.any(mask):
                    dirs = -np.sign(current_centers[mask, k])
                    dirs[dirs == 0] = 1.0
                    strengths = (limit - dists[mask]) / limit
                    forces[mask, k] += dirs * strengths
                    
            # Pairwise forces
            tree = cKDTree(current_centers)
            limit_dist = 2 * max(r_current * margin, 0.01)
            
            for i in range(N):
                neighbors = tree.query_ball_point(current_centers[i], limit_dist)
                for j in neighbors:
                    if j <= i:
                        continue
                    diff = current_centers[i] - current_centers[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-9:
                        forces[i] += np.random.normal(0, 0.01, 3)
                        forces[j] -= np.random.normal(0, 0.01, 3)
                    elif dist < limit_dist:
                        strength = (limit_dist - dist) / limit_dist
                        unit_diff = diff / dist
                        forces[i] += unit_diff * strength
                        forces[j] -= unit_diff * strength
            
            # Normalize and apply
            norms = np.linalg.norm(forces, axis=1, keepdims=True)
            norms[norms == 0] = 1
            forces = forces / norms
            
            current_centers += step_size * forces
            current_centers = np.clip(current_centers, -0.4999, 0.4999)
            
            step_size *= decay
            r_current = _compute_radius(current_centers)
    
    # Phase 4: High-beta polish for final precision
    if _time_left() > 3.0 and (time.time() - start_t) < time_budget - 1.0:
        hb_time = min(8.0, _time_left() - 2.0)
        if hb_time > 2.0:
            centers_hb, radius_hb = _polish_high_beta(
                best_centers, iu, time_budget=hb_time
            )
            if radius_hb > best_radius:
                best_radius = radius_hb
                best_centers = centers_hb.copy()
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    
    # Precompute upper triangle indices for soft-min objective
    iu = np.triu_indices(N, k=1)
    
    # Strategy:
    # 1. Try multiple initializations (Grid, Halton, Random)
    # 2. For each, run Soft-Min + Repulsion Polish
    # 3. Keep the best result
    
    initializations = [
        _build_bcc_init(seed=42),
        _build_sobol_init(seed=42),
        _build_grid_init(5, 5, 4, noise_scale=0.01, seed=42),
        _build_halton_init(seed=42),
        _build_sobol_init(seed=123),
        np.random.default_rng(1).uniform(-0.45, 0.45, (N, 3)),
    ]
    
    best_centers_overall = None
    best_radius_overall = -1.0
    
    # We have ~145s total.
    # Try 4-5 initializations with ~25s each, leaving ~20s for final polish.
    
    n_inits = len(initializations)
    time_per_init = 22.0
    
    for i, init_centers in enumerate(initializations):
        if _time_left() < time_per_init + 5.0:
            break
            
        # Ensure valid start
        if not np.all(np.isfinite(init_centers)):
            continue
            
        centers, radius = _optimize_from_initial(
            init_centers, 
            _get_deadline(), 
            iu, 
            time_budget=time_per_init
        )
        
        if radius > best_radius_overall:
            best_radius_overall = radius
            best_centers_overall = centers.copy()
            
    # Final polish on the best result found
    if best_centers_overall is not None and _time_left() > 5.0:
        time_remaining = _time_left() - 2.0
        if time_remaining > 10.0:
            centers, radius = _optimize_from_initial(
                best_centers_overall,
                _get_deadline(),
                iu,
                time_budget=time_remaining
            )
            if radius > best_radius_overall:
                best_radius_overall = radius
                best_centers_overall = centers.copy()
    
    # Fallback if something went wrong
    if best_centers_overall is None:
        best_centers_overall = _build_grid_init(5, 5, 4, noise_scale=0.0, seed=42)
        
    return best_centers_overall
