"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

# Deadline guard
START_TIME = time.time()
DEADLINE = 145.0  # 5s safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _compute_radius(centres):
    """
    Compute the packing radius for a set of centres.
    Uses cKDTree for efficient nearest neighbor search.
    """
    if len(centres) < 2:
        return 0.5 - np.max(np.abs(centres))
    
    # Wall distance
    wall_dists = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distance
    # We only need the minimum distance, so we can use KDTree
    tree = cKDTree(centres)
    # query_pairs returns all pairs within distance r. 
    # We don't know the min distance yet, so we estimate or use a generous bound.
    # A better approach for just min distance: query each point for its nearest neighbor.
    # dists, _ = tree.query(centres, k=2) -> dists[:, 1] is distance to nearest neighbor.
    # This is O(N log N).
    try:
        dists, _ = tree.query(centres, k=2)
        min_pair_dist = np.min(dists[:, 1])
    except Exception:
        # Fallback to cdist if KDTree fails for some reason
        d = cdist(centres, centres)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = d.min()
        
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_objective(x, beta=50.0):
    """
    Smooth surrogate for the negative packing radius.
    Used for the coarse optimization phase.
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
    pair_radii = d[d < np.inf] / 2.0
    all_radii = np.concatenate([pair_radii, wall_dists])
    
    # Soft min using log-sum-exp trick
    # soft_min(x) = -soft_max(-x)
    neg_all = -all_radii
    max_neg = np.max(neg_all)
    # Avoid overflow
    if max_neg > 50:
        exp_vals = np.exp(beta * (neg_all - max_neg))
    else:
        exp_vals = np.exp(beta * neg_all)
    
    log_sum = np.log(np.sum(exp_vals))
    soft_max_val = (1.0 / beta) * (log_sum + beta * max_neg)
    soft_min_val = -soft_max_val
    
    # Return negative of soft min (we want to minimize -r)
    return -soft_min_val

def _construct_bcc_init():
    """Body-centered cubic lattice initialization."""
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

def _construct_grid_init():
    """Standard grid initialization."""
    # 5x5x4 = 100
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5, 0.5, nx)
    ys = np.linspace(-0.5, 0.5, ny)
    zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        # Adjust to get exactly N
        if len(centers) > N:
            centers = centers[:N]
        else:
            # Pad with random
            rng = np.random.default_rng(42)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
            
    # Add noise
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _repulsion_search(centers, time_budget=10.0):
    """
    Local search using repulsion forces with KD-tree acceleration.
    This is the primary structural improvement: fast local refinement.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    # Parameters
    step_size = 0.005
    decay = 0.98
    min_step = 1e-8
    margin = 1.02 # Search radius multiplier
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500
    
    while _time_left() > 5.0 and (time.time() - start_time) < time_budget and step_size > min_step:
        r = _compute_radius(centers)
        if r > best_r + 1e-12:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # 1. Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.01)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # 2. Pairwise forces using KD-tree
        tree = cKDTree(centers)
        search_radius = 2 * max(r_current * margin, 0.01)
        
        # Query neighbors
        neighbor_pairs = tree.query_pairs(r=search_radius)
        
        for i, j in neighbor_pairs:
            if i == j:
                continue
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9 and dist < search_radius:
                # Repulsion strength increases as they get closer
                strength = (search_radius - dist) / search_radius
                unit_diff = diff / dist
                forces[i] += strength * unit_diff
                forces[j] -= strength * unit_diff
        
        # Normalize forces to prevent huge jumps
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.4999, 0.4999)
        
        step_size *= decay
        r_current = r # Update current radius for next loop's margin calc
        
    return best_centers, best_r

def _optimize_from_init(init_centres, time_budget=20.0):
    """
    Run optimization from given initialization.
    Phase 1: Coarse L-BFGS-B (optional, short)
    Phase 2: Repulsion Local Search (main work)
    Phase 3: Fine Polish
    """
    best_centres = init_centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Phase 1: Brief L-BFGS to smooth out initial noise
    if _time_left() > 10:
        x0 = best_centres.flatten()
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                args=(50.0,),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            new_centres = result.x.reshape(N, 3)
            new_r = _compute_radius(new_centres)
            if new_r > best_r:
                best_r = new_r
                best_centres = new_centres.copy()
        except Exception:
            pass

    # Phase 2: Repulsion Search
    remaining = max(0, time_budget - (time.time() - (START_TIME if 'START_TIME' in locals() else time.time()))) 
    # Simpler: just check global time
    if _time_left() > 10:
        rep_centres, rep_r = _repulsion_search(best_centres, time_budget=15.0)
        if rep_r > best_r:
            best_r = rep_r
            best_centres = rep_centres.copy()

    # Phase 3: Final Polish with smaller steps
    if _time_left() > 5:
        # Run repulsion again with smaller step size / higher precision
        # Or a final L-BFGS with high beta
        x0 = best_centres.flatten()
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                args=(200.0,),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_centres = result.x.reshape(N, 3)
            new_r = _compute_radius(new_centres)
            if new_r > best_r:
                best_r = new_r
                best_centres = new_centres.copy()
        except Exception:
            pass
            
        # One last quick repulsion pass
        if _time_left() > 2:
            rep_centres, rep_r = _repulsion_search(best_centres, time_budget=2.0)
            if rep_r > best_r:
                best_r = rep_r
                best_centres = rep_centres.copy()

    return best_centres, best_r

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
        _construct_grid_init(),
    ]
    
    # Also try a random initialization
    np.random.seed(123)
    random_init = np.random.uniform(-0.45, 0.45, (N, 3))
    inits.append(random_init)
    
    # Allocate time budget
    # We have ~145s. 
    # Let's give each init ~20-30s if it looks promising, or just run them sequentially with time checks.
    
    for i, init in enumerate(inits):
        if _time_left() < 15:
            break
        
        # Give more time to structured inits, less to random
        budget = 30.0 if i < 3 else 10.0
        
        centres, r = _optimize_from_init(init, time_budget=budget)
        
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
            
        # If we have a very good result, we can stop early or just polish the best
        # For now, continue to see if another init beats it, but check time.
        
    # Final global polish on the best candidate found
    if best_centres is not None and _time_left() > 10:
        # Run one more high-quality repulsion + L-BFGS on the best
        polished_centres, polished_r = _optimize_from_init(best_centres, time_budget=_time_left() - 5)
        if polished_r > best_r:
            best_r = polished_r
            best_centres = polished_centres.copy()
            
    if best_centres is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N]
    
    # Ensure centres are strictly inside the cube
    best_centres = np.clip(best_centres, -0.4999, 0.4999)
    
    return best_centres
# EVOLVE-BLOCK-END
