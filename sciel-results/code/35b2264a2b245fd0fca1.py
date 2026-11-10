"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _ring_init(n_points, n_rings=20):
    """
    Initialize points in concentric rings.
    This creates a structure that is close to optimal for large N,
    unlike random or simple hex grids which may have poor global structure.
    """
    pts = np.zeros((n_points, 2))
    # Distribute points into rings
    # Approximate number of points per ring ~ 2*pi*r / spacing
    # For unit circle, r goes 0..1.
    # We just assign indices to rings based on area fraction.
    
    # A better heuristic: 
    # Ring k (0-indexed) has radius r_k.
    # Number of points in ring k is proportional to r_k.
    # Let's just split indices into n_rings groups.
    # Group i gets indices [start_i, end_i).
    
    # To make it look like a packing, we want denser packing near center?
    # Actually, for equal circles, density is roughly constant in area.
    # So number of points in annulus [r, r+dr] is proportional to r dr.
    
    # Let's generate radii such that area is uniform.
    # r_i = sqrt((i + 0.5) / n_points) ? No, that's for points in unit square/disk area.
    # If we assume uniform density, r_i ~ sqrt(i/N).
    # Then angular spacing ~ 1/r_i.
    
    # Let's try assigning points to rings based on sqrt distribution.
    radii = np.sqrt((np.arange(n_points) + 0.5) / n_points)
    
    # For each point, assign an angle.
    # To minimize collisions, we should stagger angles in adjacent rings.
    # But simple approach: 
    # Group by radius bins.
    
    # Simpler robust initialization:
    # Use a hexagonal lattice but "relax" the outer boundary to fit the circle better?
    # No, let's stick to the "different strategy" directive.
    # Random restarts + SA is the key.
    
    # Let's use a "swiss cheese" or random uniform distribution as base,
    # but we will run SA which is good at fixing local overlaps.
    
    # Actually, a good seed is a hex lattice clipped to the disk.
    # The previous program did this.
    # Let's try a different seed: Points on a spiral (Fibonacci spiral).
    # This ensures even distribution.
    
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    angles = np.arange(n_points) * golden_angle
    # Radius scaling: sqrt(i) gives uniform area density.
    radii = np.sqrt(np.arange(n_points) / n_points) * 0.95 # Keep slightly inside
    
    pts[:, 0] = radii * np.cos(angles)
    pts[:, 1] = radii * np.sin(angles)
    
    return pts

def _score(pts):
    """
    Calculate the radius r for a given configuration.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
    """
    n = len(pts)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    # Vectorized calculation of min distance is O(N^2) memory if done naively.
    # N=300, N^2 = 90,000. This is small.
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx**2 + dy**2)
    
    # Set diagonal to inf
    np.fill_diagonal(dist_mat, np.inf)
    
    min_pair_dist = np.min(dist_mat)
    
    # Boundary distance
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary_dist = 1.0 - np.max(radii)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    if not np.isfinite(r):
        return -1e10
    return r

def _simulated_annealing(pts, time_limit, target_score):
    """
    Run Simulated Annealing to improve the packing.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Parameters
    T = 0.05  # Initial temperature
    T_min = 1e-5
    alpha = 0.999  # Cooling rate
    step_size = 0.02 # Initial step size for moves
    
    # We want to maximize score.
    # Score is min(dist/2, boundary).
    # Moves: pick a random point, move it by a random small vector.
    
    while time.time() < time_limit:
        # Check for termination if we are close to theoretical max?
        # Theoretical max for N=300 is approx 0.05315.
        # If best_score > 0.05315, we might be done, but let's keep going to be sure.
        
        if current_score > target_score:
             # We beat the target, but keep trying to improve slightly
             pass
             
        # Pick a random point to move
        idx = np.random.randint(0, N)
        
        # Generate a random move
        # Step size decreases as T decreases
        current_step = step_size * (T / 0.05)
        
        delta = np.random.normal(0, current_step, 2)
        
        new_pts = current_pts.copy()
        new_pts[idx] += delta
        
        # Ensure new point is inside the unit circle (hard constraint for validity)
        # If it goes outside, we can either reject or project it back.
        # Projecting back changes the move magnitude, which is fine.
        r_new = np.hypot(new_pts[idx, 0], new_pts[idx, 1])
        if r_new > 1.0:
             # Project to boundary (or slightly inside)
             scale = (1.0 - 1e-9) / r_new
             new_pts[idx] *= scale
             # If we project, the move is not exactly delta, but it's a valid move.
             # Alternatively, just reject if outside. Rejection is safer for SA.
             # Let's reject if outside to keep the "random walk" property cleaner.
             # Actually, for packing, staying inside is crucial.
             # Let's reject if outside.
             continue
             
        new_score = _score(new_pts)
        
        # Acceptance criterion
        delta_score = new_score - current_score
        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            current_pts = new_pts
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Reheating strategy: if no improvement for a while, increase T
        # We track time since last improvement
        # For simplicity in this loop, we can just let it cool.
        # But to be robust, let's add a simple stagnation check.
        # If best_score hasn't changed for a long time, reheat.
        # This requires tracking time_since_improvement.
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start strategy
    # 1. Fibonacci Spiral Init
    # 2. Hex Lattice Init
    # 3. Random Uniform Init
    
    inits = []
    
    # Init 1: Fibonacci
    inits.append(_ring_init(N))
    
    # Init 2: Hex Lattice
    # Generate hex lattice points
    r_hex = 0.052
    spacing = 2.0 * r_hex
    k = int(1.0 / spacing) + 2
    pts_hex = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - r_hex:
                pts_hex.append((x, y))
    if len(pts_hex) >= N:
        pts_hex = np.array(pts_hex)
        dists = np.hypot(pts_hex[:, 0], pts_hex[:, 1])
        idx = np.argsort(dists, kind="stable")[:N]
        inits.append(pts_hex[idx])
    else:
        # Fallback to random if hex fails
        inits.append(np.random.uniform(-1, 1, (N, 2)) * 0.8)

    # Init 3: Random Uniform
    # Use rejection sampling to keep inside unit circle
    pts_rand = np.random.uniform(-1, 1, (N, 2))
    radii = np.hypot(pts_rand[:, 0], pts_rand[:, 1])
    mask = radii <= 1.0
    # If not enough, resample
    while np.sum(mask) < N:
        temp = np.random.uniform(-1, 1, (N, 2))
        radii_temp = np.hypot(temp[:, 0], temp[:, 1])
        mask_temp = radii_temp <= 1.0
        pts_rand[~mask] = temp[mask_temp][:N - np.sum(mask)]
        mask[~mask] = True
    inits.append(pts_rand)
    
    # Run SA on each init
    # Allocate time budget
    n_inits = len(inits)
    time_per_init = (deadline - start_time) / n_inits
    
    for i, init_pts in enumerate(inits):
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            break
            
        # Give more time to the first init, less to subsequent?
        # Or equal time.
        # Let's use equal time but stop early if we find a very good solution.
        # Target score is the best known: 0.0531542991469
        # If we beat this, we can stop earlier?
        # No, we want to maximize.
        
        # Adjust time per init dynamically
        time_budget = min(time_per_init, remaining_time)
        
        # Run SA
        # We pass a slightly higher target to encourage exploration?
        # No, SA just maximizes.
        
        res_pts, res_score = _simulated_annealing(init_pts, time_budget, best_score)
        
        if res_score > best_score:
            best_score = res_score
            best_pts = res_pts
            
        # If we are significantly better than the known best, we might be done
        # Known best: 0.0531542991469
        if best_score > 0.0531542991469 + 1e-6:
             # We have beaten the known record!
             # Use remaining time to polish this specific solution
             remaining = deadline - time.time()
             if remaining > 1.0:
                 res_pts, res_score = _simulated_annealing(best_pts, remaining, best_score)
                 if res_score > best_score:
                     best_score = res_score
                     best_pts = res_pts
             break
            
    if best_pts is None:
        # Fallback
        best_pts = inits[0]
        
    return best_pts

# EVOLVE-BLOCK-END
