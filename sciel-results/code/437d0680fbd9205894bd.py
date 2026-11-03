"""Combined program: Local Search + Exact Epigraph SLSQP Polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Generate diverse initializations
    initializations = _generate_initializations()
    
    best_r = -1
    best_centers = None
    
    for centers0 in initializations:
        if time.time() > deadline - 20.0:
            break
        
        # Phase 1: Fast Local Search (Hill Climbing)
        # This establishes the topology and gets close to the local optimum
        centers_ls = _local_search(centers0.copy(), time_limit=10.0)
        r_ls = _compute_radius(centers_ls)
        
        if r_ls > best_r:
            best_r = r_ls
            best_centers = centers_ls.copy()
            
        # Phase 2: Exact Epigraph Polish
        # This uses SLSQP to maximize r subject to exact non-overlap and boundary constraints
        if time.time() < deadline - 5.0:
            centers_polished, r_polished = _epigraph_polish(centers_ls)
            
            if r_polished > best_r:
                best_r = r_polished
                best_centers = centers_polished
        
        # Phase 3: Final Micro-Polish (Optional, if time permits)
        # A second pass of local search with very small steps to ensure no tiny improvements are missed
        if time.time() < deadline - 2.0 and best_centers is not None:
            centers_final = _local_search(best_centers.copy(), time_limit=3.0, initial_step=0.001)
            r_final = _compute_radius(centers_final)
            if r_final > best_r:
                best_r = r_final
                best_centers = centers_final

    if best_centers is None:
        return initializations[0]

    return best_centers

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
        
    # Wall distances
    wall_dists = np.min(np.array([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ]), axis=0)
    min_wall = np.min(wall_dists)
    
    # Pair distances
    if n > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        # Avoid sqrt for speed if possible, but we need min dist
        # We only need the minimum, so we can just take min of sqrt
        # For N=50, 1225 pairs, sqrt is cheap.
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = 0.5

    return min(min_wall, min_pair)

def _local_search(centers, time_limit=10.0, initial_step=0.01):
    """
    Perform local search (hill climbing) to improve the packing.
    Moves circles to increase the minimum distance to neighbors and walls.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    step_size = initial_step
    no_improve_count = 0
    start_time = time.time()
    n = len(centers)
    
    # Pre-compute pair indices for efficiency? 
    # For N=50, O(N^2) per step is fine.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Random order to avoid bias
        indices = np.random.permutation(n)
        
        for idx in indices:
            orig_center = centers[idx].copy()
            
            # Find current bottleneck for this circle
            wall_dist = 0.5 - np.abs(orig_center)
            current_min_wall = np.min(wall_dist)
            
            # Dist to other circles
            diffs = centers - orig_center
            dists_sq = np.sum(diffs**2, axis=1)
            dists_sq[idx] = np.inf
            current_min_pair_sq = np.min(dists_sq)
            current_min_pair = np.sqrt(current_min_pair_sq) / 2.0
            
            current_local_min = min(current_min_wall, current_min_pair)
            
            # Determine direction to move
            # If wall is bottleneck, move away from wall
            # If pair is bottleneck, move away from closest neighbor
            
            if current_min_wall < current_min_pair:
                # Move away from wall
                # Determine which wall
                if orig_center[0] > 0:
                    direction = np.array([-1, 0])
                elif orig_center[0] < 0:
                    direction = np.array([1, 0])
                elif orig_center[1] > 0:
                    direction = np.array([0, -1])
                elif orig_center[1] < 0:
                    direction = np.array([0, 1])
                else:
                    direction = np.array([1, 1]) # Corner case, pick any
            else:
                # Move away from closest neighbor
                closest_idx = np.argmin(dists_sq)
                direction = orig_center - centers[closest_idx]
                norm = np.linalg.norm(direction)
                if norm > 1e-9:
                    direction /= norm
                else:
                    direction = np.array([1.0, 0.0])
            
            # Try a few random perturbations around the suggested direction
            best_local_min = current_local_min
            best_new_center = orig_center
            
            # Adaptive number of trials
            num_trials = 20
            
            for _ in range(num_trials):
                noise = np.random.normal(0, 1, 2)
                move = direction * step_size + noise * step_size * 0.5
                
                new_center = orig_center + move
                new_center = np.clip(new_center, -0.5, 0.5)
                
                # Calculate new local min
                new_wall_dist = 0.5 - np.abs(new_center)
                new_min_wall = np.min(new_wall_dist)
                
                new_diffs = centers - new_center
                new_dists_sq = np.sum(new_diffs**2, axis=1)
                new_dists_sq[idx] = np.inf
                new_min_pair = np.sqrt(np.min(new_dists_sq)) / 2.0
                
                new_local_min = min(new_min_wall, new_min_pair)
                
                if new_local_min > best_local_min + 1e-12:
                    best_local_min = new_local_min
                    best_new_center = new_center
            
            if best_local_min > current_local_min + 1e-12:
                centers[idx] = best_new_center
                improved = True
        
        if not improved:
            no_improve_count += 1
            step_size *= 0.9
            if no_improve_count > 50:
                break
        else:
            no_improve_count = 0
            
        current_r = _compute_radius(centers)
        if current_r > best_r:
            best_r = current_r
            
    return centers

def _epigraph_polish(centers):
    """
    Polish using SLSQP with an epigraph formulation.
    Maximize r subject to:
    1. ||c_i - c_j||^2 >= (2r)^2
    2. 0.5 - |c_ik| >= r
    """
    n = len(centers)
    # Variables: [x1, y1, ..., xN, yN, r]
    x0 = np.concatenate([centers.flatten(), [_compute_radius(centers)]])
    
    # Bounds for coordinates
    bounds = [(-0.5, 0.5)] * (2 * n)
    # Bound for r (must be positive)
    bounds.append((1e-6, 0.5))
    
    constraints = []
    
    # Pair constraints: ||c_i - c_j||^2 - 4r^2 >= 0
    # We can generate these on the fly to save memory, or pre-define
    # For N=50, 1225 constraints. Defining them all might be slow to create.
    # Let's define a generic constraint function that checks all pairs.
    # SLSQP expects a list of constraint dicts.
    # It is more efficient to define a single constraint function that returns
    # an array of violations for all pairs? No, SLSQP usually handles
    # individual constraints better for debugging, but for speed, a vectorized
    # constraint is better if supported.
    # scipy.optimize.minimize with method='SLSQP' accepts a list of constraints.
    # Each constraint is a dict {'type': 'ineq', 'fun': lambda x: ...}
    # The fun should return >= 0.
    
    # To avoid creating 1225+100 lambda functions, we can create a few
    # aggregate constraints? No, that breaks the linearity/convexity assumptions
    # if not done carefully.
    
    # Alternative: Use the fact that we only need to satisfy the constraints.
    # We will create constraints for pairs and walls.
    
    # Wall constraints: 0.5 - |x_i| - r >= 0 and 0.5 - |y_i| - r >= 0
    # This is non-differentiable at x_i=0.
    # To handle this robustly, we can split x_i into x_i_pos and x_i_neg?
    # Or just rely on SLSQP's numerical gradient which handles kinks okay-ish.
    # Let's try the direct approach first.
    
    def wall_constraint_x(i):
        def fun(x):
            xi = x[i*2]
            r = x[-1]
            # 0.5 - |xi| - r
            return 0.5 - abs(xi) - r
        return {'type': 'ineq', 'fun': fun}
        
    def wall_constraint_y(i):
        def fun(x):
            yi = x[i*2+1]
            r = x[-1]
            return 0.5 - abs(yi) - r
        return {'type': 'ineq', 'fun': fun}
        
    def pair_constraint(i, j):
        def fun(x):
            ci = x[i*2:i*2+2]
            cj = x[j*2:j*2+2]
            r = x[-1]
            dist_sq = np.sum((ci - cj)**2)
            return dist_sq - 4*r*r
        return {'type': 'ineq', 'fun': fun}
        
    # This creates ~1325 constraints. This might be slow for SLSQP to set up.
    # Let's try a different approach:
    # Use a penalty method or just stick to the soft-min but with higher beta?
    # No, the directive says "SLSQP epigraph... avoiding approximation error".
    
    # To make it faster, we can reduce the number of constraints by only
    # including "active" or "near-active" constraints?
    # SLSQP doesn't support dynamic constraint sets easily in a single call.
    
    # Let's try to implement the constraints as a single vectorized function
    # that returns an array of values, and use a wrapper?
    # SLSQP does not natively support vectorized inequality constraints in a single dict
    # in the same way L-BFGS-B does. It expects a list.
    
    # Given the time limit, creating 1300 lambdas might be acceptable if done once.
    # But calling them repeatedly is slow.
    
    # Let's optimize the constraint functions.
    
    cons = []
    
    # Walls
    for i in range(n):
        cons.append(wall_constraint_x(i))
        cons.append(wall_constraint_y(i))
        
    # Pairs
    for i in range(n):
        for j in range(i+1, n):
            cons.append(pair_constraint(i, j))
            
    # Objective: Maximize r -> Minimize -r
    def objective(x):
        return -x[-1]
        
    try:
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if result.success:
            polished_centers = result.x[:2*n].reshape(n, 2)
            r_polished = result.x[-1]
            # Verify with true radius
            r_true = _compute_radius(polished_centers)
            if r_true >= result.x[-1] - 1e-9:
                return polished_centers, r_true
        else:
            # If SLSQP fails, return original
            return centers, _compute_radius(centers)
            
    except Exception:
        return centers, _compute_radius(centers)
        
    return centers, _compute_radius(centers)

def _generate_initializations():
    """Generate multiple diverse initial configurations."""
    inits = []

    # Grid-based initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    inits.append(grid.copy())

    # Random initialization with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        rng = np.random.default_rng(seed)
        rand = rng.uniform(-0.45, 0.45, size=(N, 2))
        inits.append(rand)

    # Hexagonal-like packing initialization
    hex_init = _hexagonal_init()
    inits.append(hex_init)

    # Perturbed grid
    rng = np.random.default_rng(777)
    perturbed_grid = grid.copy()
    perturbed_grid += rng.normal(0, 0.02, size=perturbed_grid.shape)
    # Clip to bounds
    perturbed_grid = np.clip(perturbed_grid, -0.49, 0.49)
    inits.append(perturbed_grid)

    return inits

def _hexagonal_init():
    """Create a hexagonal close-packed initialization."""
    centers = []
    # Approximate spacing for hexagonal packing of 50 circles
    best_spacing = None
    best_count = 0

    for spacing in np.arange(0.06, 0.12, 0.001):
        row_h = spacing * np.sqrt(3) / 2
        y = -0.5 + spacing / 2
        row = 0
        while y <= 0.5 - spacing / 2 and len(centers) < N:
            x_offset = 0 if row % 2 == 0 else spacing / 2
            x = -0.5 + spacing / 2 + x_offset
            while x <= 0.5 - spacing / 2 and len(centers) < N:
                centers.append([x, y])
                x += spacing
            y += row_h
            row += 1

        if len(centers) >= N:
            if best_count < N or (best_count == N and best_spacing is None):
                best_spacing = spacing
                best_count = N
                break
        elif len(centers) > best_count:
            best_count = len(centers)
            best_spacing = spacing

    if len(centers) < N:
        # Fill remaining with random
        rng = np.random.default_rng(999)
        while len(centers) < N:
            centers.append(rng.uniform(-0.45, 0.45, size=2).tolist())
    else:
        centers = centers[:N]

    return np.array(centers)

# EVOLVE-BLOCK-END
