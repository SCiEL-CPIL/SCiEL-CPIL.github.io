"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _initial_config(n):
    """
    Generate initial configuration using a jittered grid.
    We use a 4x4x4 grid (64 points) and select 50, or just use 50 random points
    with some structure to avoid immediate overlaps.
    A better start: Place points in a 5x5x2 grid? No, 50 is prime-ish.
    Let's use a random start but ensure minimum separation by rejection sampling
    or just start with a grid and jitter.
    """
    np.random.seed(42)
    
    # Start with a 4x4x4 grid, take first 50, add large noise
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Select 50 points. To be more symmetric, let's shuffle and take 50.
    indices = np.random.permutation(64)[:50]
    c = grid[indices].copy()
    
    # Add significant noise to break the grid symmetry
    c += np.random.uniform(-0.1, 0.1, size=c.shape)
    c = np.clip(c, -0.49, 0.49)
    
    return c

def _forces(c, r_target=0.13):
    """
    Compute repulsive forces.
    We want to maximize the minimum distance.
    A common trick is to maximize the sum of inverse powers of distances,
    but that favors many small distances over one large one.
    
    Better: Use a potential that is steep when distances are small.
    Let's define force on i from j as:
    F_ij = (c_i - c_j) / |c_i - c_j|^3  (Coulomb-like)
    But this only pushes apart. We need to handle walls.
    
    Actually, for packing, we want to push spheres apart until they touch.
    If we simply minimize the sum of (1/d_ij)^k for large k, we maximize min d.
    However, gradient descent on this is unstable.
    
    Let's use a Simulated Annealing approach with a move set.
    The "energy" we want to minimize is -r (maximize r).
    But r is non-smooth.
    
    Alternative: Minimize E = sum_{i<j} exp(-d_ij / T) + sum_i exp(-w_i / T)?
    No, let's stick to SA with a clear objective: Maximize min(d_ij/2, w_i).
    
    We will define a move as a random displacement of one or a few spheres.
    The acceptance criterion is based on the change in the objective function (radius).
    """
    pass # Placeholder, we'll implement SA directly

def _simulated_annealing(c, t_start, t_end, n_iter=None, time_limit=100.0):
    """
    Simulated Annealing to maximize the minimum clearance.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    current_r = best_r
    T = t_start
    t0 = time.time()
    
    # Precompute indices for efficiency? N=50 is small, so full recompute is fast enough.
    # 50 points -> 1225 pairs.
    
    step_size = 0.05
    cooling_rate = 0.9995
    
    while time.time() - t0 < time_limit:
        # Randomly select a sphere to move
        i = np.random.randint(N)
        
        # Random displacement
        # Scale displacement with temperature to allow large jumps early on
        move = np.random.normal(0, step_size * (T / t_start + 0.1), size=3)
        
        c_new = c.copy()
        c_new[i] += move
        c_new[i] = np.clip(c_new[i], -0.5 + 1e-9, 0.5 - 1e-9)
        
        r_new = _compute_radius(c_new)
        
        # Acceptance criterion
        if r_new > current_r:
            c = c_new
            current_r = r_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        else:
            # Metropolis criterion: accept with probability exp((r_new - current_r) / T)
            # Note: r is small, T should be scaled appropriately.
            # Let's scale T to be in the range of radius differences.
            # Initial T ~ 0.01, final T ~ 1e-6
            delta_r = r_new - current_r
            if delta_r < 0:
                prob = np.exp(delta_r / T)
                if np.random.rand() < prob:
                    c = c_new
                    current_r = r_new
        
        # Cool down
        T *= cooling_rate
        
        # Adapt step size?
        # If we are stuck, maybe increase step size?
        # Simple heuristic: if no improvement in last 1000 steps, increase step size slightly
        # For now, keep it simple.
        
    return best_c, best_r

def _local_refine(c, time_limit=10.0):
    """
    Perform a local refinement using gradient-free optimization (Nelder-Mead)
    on a smoothed objective or just fine-tuning SA.
    Since N=50, Nelder-Mead is too slow (150 dimensions).
    Instead, do a high-precision SA with very small steps.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    t0 = time.time()
    T = 1e-5
    step_size = 0.001
    
    while time.time() - t0 < time_limit:
        i = np.random.randint(N)
        move = np.random.normal(0, step_size, size=3)
        c_new = c.copy()
        c_new[i] += move
        c_new[i] = np.clip(c_new[i], -0.5 + 1e-9, 0.5 - 1e-9)
        
        r_new = _compute_radius(c_new)
        
        if r_new > current_r if 'current_r' in locals() else r_new > best_r:
             pass # Logic fix below
        
        # Correct logic:
        if r_new > best_r:
            c = c_new
            best_r = r_new
            best_c = c_new.copy()
        else:
            # Accept worse moves with low probability to escape local minima
            delta_r = r_new - best_r # This is wrong, should be current state
            # We need to track current state separately from best state in SA
            # Let's rewrite the loop properly.
            
    # Rewrite local refine properly
    c_curr = c.copy()
    r_curr = _compute_radius(c_curr)
    best_c_local = c_curr.copy()
    best_r_local = r_curr
    T = 1e-4
    step = 0.002
    
    while time.time() - t0 < time_limit:
        i = np.random.randint(N)
        move = np.random.normal(0, step, size=3)
        c_trial = c_curr.copy()
        c_trial[i] += move
        c_trial[i] = np.clip(c_trial[i], -0.5 + 1e-9, 0.5 - 1e-9)
        
        r_trial = _compute_radius(c_trial)
        
        if r_trial > r_curr:
            c_curr = c_trial
            r_curr = r_trial
            if r_curr > best_r_local:
                best_r_local = r_curr
                best_c_local = c_curr.copy()
        else:
            delta = r_trial - r_curr
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    c_curr = c_trial
                    r_curr = r_trial
        
        T *= 0.999
        if T < 1e-8:
            T = 1e-8
            
    return best_c_local, best_r_local

def construct():
    t0 = time.time()
    deadline = t0 + 145.0 # Safety margin
    
    np.random.seed(42)
    
    # 1. Initial Configuration
    c = _initial_config(N)
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    # 2. Main Simulated Annealing
    # Start with high temperature to explore, cool down to exploit
    # Time budget: ~100s for SA, ~20s for local refine, ~10s buffer
    time_left_sa = deadline - time.time() - 30.0
    if time_left_sa > 5.0:
        c_sa, r_sa = _simulated_annealing(c, t_start=0.05, t_end=1e-6, time_limit=time_left_sa)
        if r_sa > best_r:
            best_r = r_sa
            best_c = c_sa.copy()
            
    # 3. Local Refinement
    time_left_refine = deadline - time.time() - 5.0
    if time_left_refine > 2.0:
        c_ref, r_ref = _local_refine(best_c, time_limit=time_left_refine)
        if r_ref > best_r:
            best_r = r_ref
            best_c = c_ref.copy()
            
    # 4. Final Check
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
