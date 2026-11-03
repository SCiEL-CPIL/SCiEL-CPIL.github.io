import numpy as np
import time

N = 50

def compute_radius(X):
    """
    Computes the maximum radius of non-overlapping circles centered at X
    that fit inside the unit circle.
    Returns -inf if any circle is outside the unit circle.
    """
    # Check if any point is outside the unit circle
    # |c_i| <= 1
    dists_to_origin = np.linalg.norm(X, axis=1)
    max_dist = np.max(dists_to_origin)
    if max_dist > 1.0:
        return -np.inf
    
    # Boundary constraint radius
    r_boundary = 1.0 - max_dist
    
    # Pairwise distances
    # X is (N, 2)
    # dist[i, j] = ||X[i] - X[j]||
    # We only need the minimum distance between distinct pairs
    # Using broadcasting
    # diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # More efficient:
    # Squared distances
    x2 = np.sum(X**2, axis=1)
    # dist^2 = x2_i + x2_j - 2 * X_i . X_j
    dot = X @ X.T
    dist_sq = x2[:, np.newaxis] + x2[np.newaxis, :] - 2 * dot
    # Clip small negative values due to floating point errors
    dist_sq = np.maximum(dist_sq, 0)
    
    # Set diagonal to infinity so we don't pick 0 distance
    np.fill_diagonal(dist_sq, np.inf)
    
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r_overlap = min_dist / 2.0
    
    return min(r_boundary, r_overlap)

def sa_optimize(X0, time_budget):
    """
    Simulated Annealing to maximize the packing radius.
    """
    X = X0.copy()
    current_radius = compute_radius(X)
    best_X = X.copy()
    best_radius = current_radius
    
    # Parameters
    T = 0.1  # Initial temperature
    cooling_rate = 0.9999
    step_size = 0.1
    start_time = time.time()
    
    # Adaptive step size? Maybe keep it simple first.
    
    while (time.time() - start_time) < time_budget:
        # Propose a move
        idx = np.random.randint(N)
        # Gaussian perturbation
        delta = np.random.normal(0, step_size, 2)
        X_new = X.copy()
        X_new[idx] += delta
        
        # Check if it's valid (inside circle)
        # If not, we can still evaluate, but compute_radius returns -inf
        new_radius = compute_radius(X_new)
        
        # Acceptance criterion
        if new_radius > current_radius:
            # Better, always accept
            X = X_new
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_X = X.copy()
        else:
            # Worse, accept with probability exp((new - current) / T)
            # Note: we are maximizing, so delta = new - current is negative
            delta = new_radius - current_radius
            if delta < 0:
                # Avoid overflow/underflow issues with exp
                # If T is very small, exp(delta/T) is 0
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    X = X_new
                    current_radius = new_radius
        
        # Cool down
        T *= cooling_rate
        # If T gets too small, the algorithm might get stuck.
        # We can reheat or just stop.
        if T < 1e-8:
            break
            
    return best_X, best_radius

def init_hex():
    # Hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def init_rings():
    # Ring based initialization
    # 1 center
    points = [np.array([0.0, 0.0])]
    remaining = N - 1
    ring_idx = 1
    while remaining > 0:
        # Number of points in ring k is approx 6k
        # Adjust to fit remaining
        pts_in_ring = min(6 * ring_idx, remaining)
        # Radius of ring
        # Approximate radius to fit 6k points: 2*pi*r / (6k) ~ 2r (diameter of circle)
        # r ~ 6k / (2*pi) * r_max? No.
        # Let's just distribute evenly in angle.
        # Radius needs to be such that they don't overlap.
        # Approx radius for ring k: r_k ~ k * (2/sqrt(3)) * r_unit?
        # Let's use a heuristic: r_k = k * 0.15
        r = ring_idx * 0.15
        angles = np.linspace(0, 2 * np.pi, pts_in_ring, endpoint=False)
        for a in angles:
            points.append(np.array([r * np.cos(a), r * np.sin(a)]))
        remaining -= pts_in_ring
        ring_idx += 1
        
    X = np.array(points[:N])
    # Randomize slightly to break symmetry
    X += np.random.normal(0, 0.01, X.shape)
    return X

def construct():
    start_time = time.time()
    total_time = 150.0
    safety_margin = 10.0
    available_time = total_time - safety_margin
    
    best_X = None
    best_radius = -np.inf
    
    # Strategies:
    # 1. Hex lattice
    # 2. Ring initialization
    # 3. Random initialization
    
    inits = []
    
    # Add hex
    X_hex = init_hex()
    inits.append(X_hex)
    
    # Add rings
    X_rings = init_rings()
    inits.append(X_rings)
    
    # Add a few random ones
    for _ in range(3):
        # Random points inside unit circle
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        X_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(X_rand)
    
    # Time per init
    time_per_init = available_time / len(inits)
    
    for i, X0 in enumerate(inits):
        if time.time() - start_time > available_time:
            break
            
        # Run SA
        X_opt, r_opt = sa_optimize(X0, time_per_init)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_X = X_opt
        
        # If we have a very good result, we might stop early?
        # The known best is ~0.1258.
        # If we exceed that, we are doing great.
        if best_radius > 0.1258:
             # We found a solution better than the known best? Unlikely, but possible.
             # Or just close to it.
             pass

    # Final polish?
    # If we have time left, we can run a short SA on the best solution found
    # with a smaller step size.
    remaining_time = available_time - (time.time() - start_time)
    if remaining_time > 5.0 and best_X is not None:
        # Refine the best solution
        X_refined, r_refined = sa_optimize(best_X, remaining_time)
        if r_refined > best_radius:
            best_radius = r_refined
            best_X = X_refined
            
    return best_X

# EVOLVE-BLOCK-END
