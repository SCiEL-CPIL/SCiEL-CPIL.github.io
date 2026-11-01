"""
Packing 50 equal circles in a unit square using gradient-based repulsion
with simulated annealing. Directly maximizes minimum distance (the actual
scoring objective) rather than doing feasibility checks at fixed radii.
"""
import numpy as np
import time

N = 50
HALF_SIDE = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to each wall
    dists_to_edge = np.min(np.array([
        HALF_SIDE - np.abs(centers[:, 0]),
        HALF_SIDE - np.abs(centers[:, 1])
    ]), axis=1)
    if np.any(dists_to_edge < 0):
        return 0.0
    min_edge_dist = np.min(dists_to_edge)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, min_edge_dist)

def _min_dist_and_grad(centers):
    """
    Compute the minimum pairwise distance (or edge distance) and its gradient
    w.r.t. all centers. Returns (min_val, gradient_array of shape (N,2)).
    We maximize min_val; gradient points in direction of increasing min_val.
    """
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N,N,2)
    dists = np.sqrt(np.sum(diff**2, axis=2))  # (N,N)
    np.fill_diagonal(dists, np.inf)
    
    # Edge distances
    edge_dists = np.min(np.array([
        HALF_SIDE - np.abs(centers[:, 0]),
        HALF_SIDE - np.abs(centers[:, 1])
    ]), axis=1)  # (N,)
    
    # Find the minimum value and whether it's from pair or edge
    min_pair_val = np.min(dists)
    min_edge_val = np.min(edge_dists)
    
    if min_pair_val < 2.0 * min_edge_val:
        # The bottleneck is a pair
        min_val = min_pair_val / 2.0
        # Find which pair
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = idx
        if i == j:
            return 0.0, np.zeros((N, 2))
        d = centers[i] - centers[j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            # Degenerate: push apart randomly
            grad = np.zeros((N, 2))
            rng = np.random.default_rng()
            direction = rng.normal(size=2)
            direction /= (np.linalg.norm(direction) + 1e-15)
            grad[i] = direction
            grad[j] = -direction
            return min_val, grad
        # Gradient of min_pair_dist/2 w.r.t. centers[i] is +d/dist, w.r.t. centers[j] is -d/dist
        # To maximize, we move i in +d/dist direction and j in -d/dist direction
        unit = d / dist
        grad = np.zeros((N, 2))
        grad[i] = unit
        grad[j] = -unit
        return min_val, grad
    else:
        # The bottleneck is an edge
        min_val = min_edge_val
        # Find which center and which edge
        idx = np.argmin(edge_dists)
        c = centers[idx]
        grad = np.zeros((N, 2))
        # Determine which edge is closest
        dl = HALF_SIDE - c[0]  # left
        dr = HALF_SIDE + c[0]  # right
        db = HALF_SIDE - c[1]  # bottom
        dt = HALF_SIDE + c[1]  # top
        min_edge = min(dl, dr, db, dt)
        if min_edge == dl:
            grad[idx, 0] = 1.0  # push right
        elif min_edge == dr:
            grad[idx, 0] = -1.0  # push left
        elif min_edge == db:
            grad[idx, 1] = 1.0  # push up
        else:
            grad[idx, 1] = -1.0  # push down
        return min_val, grad

def _hex_lattice_initial():
    """
    Generate a hexagonal lattice arrangement of N points in [-0.5, 0.5]^2.
    Uses rows with alternating offsets for hexagonal packing.
    """
    # For 50 circles, try various row counts
    # Hexagonal: rows of ~8-9 points
    best_centers = None
    best_score = -1.0
    
    for n_rows in range(5, 11):
        n_cols = int(np.ceil(N / n_rows))
        if n_rows * n_cols < N:
            continue
        
        # Create hex lattice
        # Spacing
        dx = 1.0 / n_cols  # horizontal spacing
        dy = dx * np.sqrt(3) / 2  # vertical spacing for hex
        
        # Center the lattice
        total_width = (n_cols - 1) * dx
        total_height = (n_rows - 1) * dy
        
        x_offset = -total_width / 2
        y_offset = -total_height / 2
        
        points = []
        for row in range(n_rows):
            offset = 0.0
            if row % 2 == 1:
                offset = dx / 2
            for col in range(n_cols):
                x = x_offset + col * dx + offset
                y = y_offset + row * dy
                points.append([x, y])
        
        points = np.array(points[:N])
        
        # Clip to valid region
        points = np.clip(points, -HALF_SIDE + 0.01, HALF_SIDE - 0.01)
        
        score = _compute_radius(points)
        if score > best_score:
            best_score = score
            best_centers = points.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers

def _gradient_ascent_with_annealing(centers, time_budget=120.0):
    """
    Perform gradient ascent on the minimum distance with simulated annealing
    to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current = centers.copy()
    current_val = _compute_radius(current)
    best = current.copy()
    best_val = current_val
    
    # Annealing parameters
    T_start = 0.02
    T_end = 1e-7
    n_steps_max = 200000
    
    rng = np.random.default_rng(42)
    
    step = 0
    while time.time() < deadline - 5.0:
        step += 1
        
        # Temperature schedule: exponential cooling
        frac = min(1.0, step / n_steps_max)
        T = T_start * (T_end / T_start) ** frac
        
        # Compute gradient
        min_val, grad = _min_dist_and_grad(current)
        
        # Gradient ascent step
        lr = 0.005 * (1.0 + T / T_start * 0.5)
        new = current + lr * grad
        
        # Add noise (annealing perturbation)
        noise = rng.normal(0, T * 0.01, size=new.shape)
        new += noise
        
        # Project back into valid region [-0.5+r, 0.5-r] approximately
        # We need to keep centers in [-0.5, 0.5]
        new = np.clip(new, -HALF_SIDE + 1e-6, HALF_SIDE - 1e-6)
        
        # Evaluate
        new_val = _compute_radius(new)
        
        # Accept if better, or with probability exp((new_val - current_val)/T)
        if new_val >= current_val:
            current = new
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best = current.copy()
        else:
            delta = new_val - current_val
            if delta > -T * 0.5:  # Only accept worse solutions if not too bad
                accept_prob = np.exp(delta / max(T, 1e-10))
                if rng.random() < accept_prob:
                    current = new
                    current_val = new_val
        
        # Every 10000 steps, check if we should restart from best with small perturbation
        if step % 50000 == 0:
            # Small kick to escape local minima
            kick = rng.normal(0, 0.005, size=current.shape)
            current = np.clip(current + kick, -HALF_SIDE + 1e-6, HALF_SIDE - 1e-6)
            current_val = _compute_radius(current)
    
    return best, best_val

def _grid_initial():
    """Simple grid fallback."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def construct():
    """
    Main function: find best packing of 50 equal circles in unit square.
    Uses gradient-based repulsion with simulated annealing.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin below 150s kill
    
    # Start with hexagonal lattice
    initial = _hex_lattice_initial()
    init_r = _compute_radius(initial)
    
    # Run gradient ascent with annealing
    best_centers, best_r = _gradient_ascent_with_annealing(initial, time_budget=DEADLINE - start_time - 5.0)
    
    # If result is invalid or terrible, fall back to grid
    final_r = _compute_radius(best_centers)
    if final_r < 1e-4:
        best_centers = _grid_initial()
        final_r = _compute_radius(best_centers)
    
    return best_centers
