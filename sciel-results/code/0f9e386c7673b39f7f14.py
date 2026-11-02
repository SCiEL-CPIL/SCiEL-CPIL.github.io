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
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    """Build a grid initialization."""
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
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
    """Generate N points using a Halton sequence (low discrepancy)."""
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
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_sobol_init(seed=42):
    """Generate N points using a Sobol sequence (low discrepancy)."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    points = sampler.random(N)
    centers = points - 0.5
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_bcc_init(seed=42, noise_scale=0.005):
    """
    Generate N points using a body-centered cubic (BCC) lattice initialization.
    The 5x5x2 grid produces 100 points exactly, providing a structured starting point
    that respects the cube geometry better than random or simple cubic grids.
    """
    nx, ny, nz = 5, 5, 2
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    
    centers = np.array(points[:N])
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _simulated_annealing(centers, iu, time_budget=10.0, initial_temp=0.05):
    """
    Simulated annealing to escape local minima.
    Uses a faster cooling schedule and larger steps to explore the landscape more effectively.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    start_t = time.time()
    rng = np.random.default_rng(42)
    
    temp = initial_temp
    cooling_rate = 0.95
    min_temp = 0.0001
    
    while _time_left() > 2.0 and (time.time() - start_t) < time_budget - 1.0 and temp > min_temp:
        perturbation = rng.normal(0, temp, centers.shape)
        new_centers = current_centers + perturbation
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_radius = _compute_radius(new_centers)
        
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

def _polish_repulsion(centers, iu, time_budget=10.0):
    """
    Force-based repulsion polish.
    Uses cKDTree for efficient neighbor queries to apply repulsive forces
    to pairs that are too close and attractive forces to walls that are too close.
    This is a structural change from gradient-based methods to a physics-based simulation.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = centers.copy()
    
    start_t = time.time()
    step_size = 0.005
    decay = 0.97
    min_step = 1e-8
    margin = 1.05
    no_improve_count = 0
    max_no_improve = 150
    
    while _time_left() > 1.0 and (time.time() - start_t) < time_budget - 1.0 and step_size > min_step and no_improve_count < max_no_improve:
        r = _compute_radius(current_centers)
        if r > best_radius + 1e-12:
            best_radius = r
            best_centers = current_centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        forces = np.zeros_like(current_centers)
        
        # Wall forces: push points away from walls if they are too close
        for k in range(3):
            dists = 0.5 - np.abs(current_centers[:, k])
            limit = max(r * margin, 0.01)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(current_centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
        
        # Pair forces: push apart points that are too close
        tree = cKDTree(current_centers)
        search_radius = 2 * max(r * margin, 0.01)
        neighbor_pairs = tree.query_pairs(r=search_radius)
        
        for i, j in neighbor_pairs:
            if i == j:
                continue
            diff = current_centers[i] - current_centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9 and dist < search_radius:
                strength = (search_radius - dist) / search_radius
                unit_diff = diff / dist
                forces[i] += strength * unit_diff
                forces[j] -= strength * unit_diff
        
        # Normalize forces to ensure consistent step size
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        current_centers += step_size * forces
        current_centers = np.clip(current_centers, -0.4999, 0.4999)
        
        step_size *= decay
    
    return best_centers, best_radius

def _polish_bottleneck(centers, iu, time_budget=5.0):
    """
    Focus optimization on the bottleneck constraints (closest pairs and nearest walls).
    Combines targeted force application with local gradient polish.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    
    current_centers = centers.copy()
    
    for iteration in range(50):
        if _time_left() < 2.0 or (time.time() - start_t) > time_budget - 1.0:
            break
        
        r = _compute_radius(current_centers)
        
        # Find bottleneck pairs and walls
        dists = cdist(current_centers, current_centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_dist = dists[min_pair_idx]
        
        dists_to_walls = 0.5 - np.abs(current_centers)
        min_wall_idx = np.unravel_index(np.argmin(dists_to_walls), dists_to_walls.shape)
        min_wall_dist = dists_to_walls[min_wall_idx]
        
        # Try to move the bottleneck points apart
        if min_pair_dist / 2.0 < min_wall_dist:
            # Move the closest pair apart
            i, j = min_pair_idx
            diff = current_centers[i] - current_centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit_diff = diff / dist
                # Move them apart slightly
                move = 0.001
                current_centers[i] += unit_diff * move
                current_centers[j] -= unit_diff * move
            else:
                # Random kick
                rng = np.random.default_rng(iteration)
                current_centers[i] += rng.normal(0, 0.01, 3)
                current_centers[j] -= rng.normal(0, 0.01, 3)
        else:
            # Move the wall-constrained point inward
            idx, axis = min_wall_idx
            current_centers[idx, axis] += np.sign(current_centers[idx, axis]) * 0.001 if current_centers[idx, axis] != 0 else 0.001
        
        current_centers = np.clip(current_centers, -0.4999, 0.4999)
        
        # Quick local polish
        try:
            x0 = current_centers.flatten()
            res = minimize(
                _soft_min_objective,
                x0,
                args=(500.0, iu),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
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

def _polish_high_beta(centers, iu, time_budget=10.0):
    """
    High-precision polish using very high beta soft-min.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    
    betas = [10000.0, 50000.0, 200000.0, 500000.0]
    
    current_centers = centers.copy()
    
    for beta in betas:
        if _time_left() < 2.0 or (time.time() - start_t) > time_budget - 1.0:
            break
            
        x0 = current_centers.flatten()
        
        try:
            max_iter = 300
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

def _optimize_from_initial(centers, iu, time_budget=20.0):
    """
    Run optimization from initial centers.
    Multi-phase: soft-min -> repulsion polish -> simulated annealing -> bottleneck polish -> high-beta polish
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    
    # Phase 1: Soft-min optimization with increasing beta
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_centers = centers.copy()
    
    for beta in betas:
        if _time_left() < 2.0 or (time.time() - start_t) > time_budget * 0.3:
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
    
    # Phase 2: Repulsion-based polish (new structural component)
    if _time_left() > 5.0 and (time.time() - start_t) < time_budget * 0.5:
        rep_time = min(5.0, time_budget * 0.2)
        centers_rep, radius_rep = _polish_repulsion(
            best_centers, iu, time_budget=rep_time
        )
        if radius_rep > best_radius:
            best_radius = radius_rep
            best_centers = centers_rep.copy()
    
    # Phase 3: Simulated annealing to escape local minima
    if _time_left() > 5.0 and (time.time() - start_t) < time_budget * 0.7:
        sa_time = min(5.0, time_budget * 0.15)
        centers_sa, radius_sa = _simulated_annealing(
            best_centers, iu, time_budget=sa_time
        )
        if radius_sa > best_radius:
            best_radius = radius_sa
            best_centers = centers_sa.copy()
    
    # Phase 4: Bottleneck-focused polish
    if _time_left() > 3.0 and (time.time() - start_t) < time_budget * 0.85:
        bottleneck_time = min(4.0, time_budget * 0.1)
        centers_bn, radius_bn = _polish_bottleneck(
            best_centers, iu, time_budget=bottleneck_time
        )
        if radius_bn > best_radius:
            best_radius = radius_bn
            best_centers = centers_bn.copy()
    
    # Phase 5: High-beta polish for final precision
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
    # 1. Try multiple initializations (BCC, Sobol, Grid, Halton, Random)
    # 2. For each, run the full optimization pipeline
    # 3. Keep the best result
    
    # BCC is the most promising initialization for this problem, so we give it priority
    initializations = [
        _build_bcc_init(seed=42),
        _build_bcc_init(seed=123),
        _build_sobol_init(seed=42),
        _build_grid_init(5, 5, 4, noise_scale=0.01, seed=42),
        _build_halton_init(seed=42),
        _build_sobol_init(seed=123),
        _build_grid_init(5, 4, 5, noise_scale=0.01, seed=99),
        np.random.default_rng(1).uniform(-0.45, 0.45, (N, 3)),
    ]
    
    best_centers_overall = None
    best_radius_overall = -1.0
    
    # We have ~145s total.
    # Try 5-6 initializations with ~18s each, leaving ~15s for final polish.
    
    n_inits = len(initializations)
    time_per_init = 16.0
    
    for i, init_centers in enumerate(initializations):
        if _time_left() < time_per_init + 5.0:
            break
            
        # Ensure valid start
        if not np.all(np.isfinite(init_centers)):
            continue
            
        centers, radius = _optimize_from_initial(
            init_centers, 
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
