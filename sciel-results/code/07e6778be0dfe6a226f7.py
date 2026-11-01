import numpy as np
import time
import sys

N = 50

def _score(centers):
    """
    Compute the maximum common radius for given centers.
    Returns the minimum of (min_pair_dist / 2, min_wall_dist).
    """
    centers = np.asarray(centers, dtype=np.float64)
    
    # Check for non-finite values or out of bounds
    if not np.all(np.isfinite(centers)):
        return -np.inf
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return -np.inf

    # Distance to walls
    # The distance from point c to the boundary [-0.5, 0.5] is min(0.5 - |c_x|, 0.5 - |c_y|, 0.5 - |c_z|)
    # Actually, for a cube [-0.5, 0.5]^3, the distance to the nearest wall for point (x,y,z) is:
    # min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y, 0.5 - z, 0.5 + z)
    # which is 0.5 - max(|x|, |y|, |z|)
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dists)
    
    if r_wall < 0:
        return -np.inf

    # Pairwise distances
    # Compute all pairwise squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    
    # Exclude self-distances
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = np.min(dists_sq)
    min_pair_dist = np.sqrt(min_dist_sq)
    
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def _initial_grid_jittered(seed=42):
    """Jittered grid initialization."""
    # 4x4x4 = 64 points, we need 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(grid))
    centers = grid[perm[:N]].copy()
    # Small jitter to break symmetry
    jitter = rng.uniform(-0.01, 0.01, size=centers.shape)
    centers += jitter
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _initial_fcc_like(seed=123):
    """
    Initialize using an FCC lattice structure clipped to the cube.
    FCC has higher packing density than SC.
    """
    rng = np.random.RandomState(seed)
    # FCC lattice vectors
    # Basis: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0) scaled by a
    # We want to fit ~50 points in [-0.5, 0.5]^3.
    # Volume of cube = 1. Volume per sphere ~ 1/50 = 0.02.
    # r ~ 0.136 -> d ~ 0.27.
    # Let's generate points on an FCC lattice with spacing ~0.28 and keep those inside the cube.
    
    a = 0.28
    points = []
    # Range needs to cover [-0.5, 0.5]
    # FCC points are at (i*a, j*a, k*a) + offsets
    offsets = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ]
    
    # Generate grid of lattice sites
    # i, j, k from -2 to 2 (approx)
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                base = np.array([i*a, j*a, k*a])
                for off in offsets:
                    p = base + np.array(off)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    
    points = np.array(points)
    if len(points) < N:
        # If we don't have enough, add random points
        extra = rng.uniform(-0.4, 0.4, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        # Randomly select N points
        perm = rng.permutation(len(points))
        points = points[perm[:N]]
        
    # Jitter
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    points = np.clip(points, -0.499, 0.499)
    return points

def _simulated_annealing(centers, t_start, deadline, seed=0):
    """
    Perform Simulated Annealing to maximize the radius.
    """
    rng = np.random.RandomState(seed)
    centers = centers.copy()
    current_r = _score(centers)
    
    best_centers = centers.copy()
    best_r = current_r
    
    # Temperature schedule
    # Start with a temperature that allows some bad moves
    # Delta r is typically small. If r ~ 0.1, a change of 0.001 is significant.
    # exp(-0.001 / T). If T=0.001, prob ~ 0.37.
    T = 0.002 
    T_min = 1e-6
    cooling_rate = 0.999
    
    # Step size for perturbations
    # Initially large, then shrink
    step_size = 0.05
    step_cooling = 0.999
    
    time_limit = deadline - 2.0 # Safety margin
    
    while time.time() < time_limit and T > T_min:
        # Generate a move
        # Pick 1 or 2 random centers to move
        n_moves = 1 if rng.random() < 0.8 else 2
        indices = rng.choice(N, size=n_moves, replace=False)
        
        original_positions = centers[indices].copy()
        
        # Perturb
        # Move in random direction with magnitude step_size
        for idx in indices:
            direction = rng.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction /= norm
            # Random distance
            dist = step_size * rng.uniform(0.5, 1.5)
            centers[idx] += direction * dist
        
        # Clip to cube
        centers[indices] = np.clip(centers[indices], -0.4999, 0.4999)
        
        new_r = _score(centers)
        
        if not np.isfinite(new_r):
            new_r = -np.inf
            
        delta_r = new_r - current_r
        
        # Acceptance criterion
        if delta_r > 0:
            # Always accept improvement
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            # Accept with probability exp(delta_r / T)
            # If delta_r is very negative, prob is ~0
            if T > 0:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current_r = new_r
                    if new_r > best_r:
                        best_r = new_r
                        best_centers = centers.copy()
                else:
                    # Reject move
                    centers[indices] = original_positions
                    # Note: current_r remains unchanged as it was calculated from original centers
            
        # Cool down
        T *= cooling_rate
        step_size *= step_cooling
        
        # If step_size becomes too small, we might be stuck. 
        # Reheat occasionally? Or just let it run.
        if step_size < 1e-5:
             break

    return best_centers, best_r

def construct():
    t_start = time.time()
    deadline = t_start + 145.0 # 5s safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # List of initializations
    inits = [
        ("grid_42", _initial_grid_jittered(seed=42)),
        ("grid_123", _initial_grid_jittered(seed=123)),
        ("fcc_123", _initial_fcc_like(seed=123)),
        ("fcc_456", _initial_fcc_like(seed=456)),
        ("rand_101", np.random.RandomState(101).uniform(-0.4, 0.4, size=(N, 3))),
        ("rand_202", np.random.RandomState(202).uniform(-0.4, 0.4, size=(N, 3))),
    ]
    
    # We have ~140s. 
    # Let's allocate time to each restart.
    # First, evaluate all inits.
    for name, init in inits:
        if time.time() > deadline - 10:
            break
        r = _score(init)
        if r > best_score:
            best_score = r
            best_centers = init.copy()
            
    # Now run Simulated Annealing on the best initializations.
    # We will run SA on the top 3 distinct initializations found so far, 
    # or just run on a few specific ones with dedicated time.
    
    # Let's run SA on 'grid_42', 'fcc_123', and 'rand_101' (or the best found).
    # To be safe, we run SA on the current best, and one other strong candidate.
    
    candidates_for_sa = []
    # Add the best found
    if best_centers is not None:
        candidates_for_sa.append(best_centers.copy())
    
    # Add a few diverse ones
    candidates_for_sa.append(_initial_fcc_like(seed=123))
    candidates_for_sa.append(_initial_grid_jittered(seed=42))
    candidates_for_sa.append(np.random.RandomState(101).uniform(-0.4, 0.4, size=(N, 3)))
    
    # Deduplicate candidates roughly (by score or just take first 4)
    unique_candidates = []
    seen = []
    for c in candidates_for_sa:
        # Simple check: is it very close to an existing candidate?
        is_dup = False
        for uc in unique_candidates:
            if np.linalg.norm(c - uc) < 0.1:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
        if len(unique_candidates) >= 4:
            break
            
    # Allocate time per SA run
    time_left = deadline - 5.0 - time.time()
    n_runs = len(unique_candidates)
    time_per_run = time_left / n_runs if n_runs > 0 else 0
    
    for i, init_c in enumerate(unique_candidates):
        if time.time() > deadline - 5.0:
            break
            
        # Calculate score of init
        init_score = _score(init_c)
        if init_score > best_score:
            best_score = init_score
            best_centers = init_c.copy()
            
        # Run SA
        # Pass the deadline for this specific run
        run_deadline = time.time() + time_per_run * 0.95 # Leave a small gap
        if run_deadline > deadline - 5.0:
            run_deadline = deadline - 5.0
            
        sa_centers, sa_score = _simulated_annealing(init_c, t_start, run_deadline, seed=1000+i)
        
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers.copy()
            
    # Final Polish: A quick local search on the best result to squeeze out more precision
    if best_centers is not None and time.time() < deadline - 2.0:
        # Simple gradient-free polish
        # Try small random steps and keep improvements
        current = best_centers.copy()
        current_r = best_score
        polish_deadline = deadline - 1.0
        rng = np.random.RandomState(999)
        
        while time.time() < polish_deadline:
            # Pick a random point
            i = rng.randint(N)
            orig = current[i].copy()
            # Small step
            step = 1e-4 * rng.normal(size=3)
            current[i] += step
            current[i] = np.clip(current[i], -0.49999, 0.49999)
            
            new_r = _score(current)
            if new_r > current_r:
                current_r = new_r
                if new_r > best_score:
                    best_score = new_r
                    best_centers = current.copy()
            else:
                current[i] = orig
                
    # Ensure final result is valid
    if best_centers is None:
        best_centers = _initial_grid_jittered()
        
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

if __name__ == "__main__":
    centers = construct()
    r = _score(centers)
    print(f"Score: {r}")
