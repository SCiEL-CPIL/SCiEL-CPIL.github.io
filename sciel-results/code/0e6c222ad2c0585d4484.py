import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s kill
    
    # 1. Initialization: Hexagonal Lattice
    # We want to fit N points. 
    # Approximate area per circle in hex packing: pi * r^2 / (2 * r^2 * sqrt(3)/2) ? 
    # Let's just generate a large hex grid and pick the N closest to center.
    
    # Estimate spacing
    # Area of unit circle = pi. N circles.
    # Approx radius r ~ sqrt(pi / (N * sqrt(3))) 
    # r ~ sqrt(3.14 / (300 * 1.732)) ~ sqrt(3.14 / 520) ~ sqrt(0.006) ~ 0.077
    # Best known is 0.053. So spacing is larger.
    # Let's use a grid with spacing s.
    # Number of points in hex grid of radius R_grid is ~ pi * R_grid^2 / (s^2 * sqrt(3)/2)
    # We want ~ 300 points.
    
    # Try different spacings to get close to N points
    def get_hex_points(spacing):
        points = []
        # Range of indices
        # Max radius in grid ~ sqrt(N) * spacing
        max_k = int(np.sqrt(N) * 2) + 5
        for i in range(-max_k, max_k + 1):
            for j in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) == 0:
            return np.zeros((0,2))
        # Sort by distance to center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        return points[idx]

    # Find a good initial spacing
    best_P = None
    best_N_pts = 0
    # Search for spacing that gives ~N points
    # r ~ 0.053, so diameter ~ 0.106. Spacing ~ 0.106.
    # Let's try a range
    for s in np.linspace(0.05, 0.15, 20):
        P = get_hex_points(s)
        if len(P) >= N:
            # Take the N closest to center
            P = P[:N]
            if len(P) > best_N_pts:
                best_N_pts = len(P)
                best_P = P
            # We want exactly N, so if we have more, we take the best N.
            # But we want the spacing to be such that the N-th point is near the boundary.
            # Actually, just taking the first N from a denser grid is a good start.
            # Let's pick the spacing that yields the largest radius for the N points.
            # We will evaluate the radius of P[:N] for each candidate s.
    
    # Let's do a more precise initialization
    # We'll try a few spacings and pick the one with the best initial radius
    candidates = []
    for s in np.linspace(0.05, 0.12, 10):
        P_full = get_hex_points(s)
        if len(P_full) >= N:
            P_init = P_full[:N]
            # Calculate radius
            # Vectorized radius calculation
            diff = P_init[:, np.newaxis, :] - P_init[np.newaxis, :, :]
            dist_mat = np.linalg.norm(diff, axis=2)
            min_dist = np.min(dist_mat[np.triu_indices(N, k=1)])
            min_bound = np.min(1 - np.linalg.norm(P_init, axis=1))
            r_init = min(min_dist / 2.0, min_bound)
            candidates.append((r_init, P_init))
            
    if not candidates:
        # Fallback: random points
        P = np.random.rand(N, 2) * 2 - 1
        P = P[np.linalg.norm(P, axis=1) <= 1.0]
        while len(P) < N:
            p = np.random.rand(2) * 2 - 1
            if np.linalg.norm(p) <= 1.0:
                P = np.vstack([P, p])
        P = P[:N]
    else:
        # Pick the best initial configuration
        candidates.sort(key=lambda x: x[0], reverse=True)
        P = candidates[0][1].copy()

    current_radius = 0.0
    # Recalculate current radius properly
    def calc_radius(P):
        if len(P) < 2:
             return 0.0
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        min_dist = np.min(dist_mat[np.triu_indices(N, k=1)])
        min_bound = np.min(1 - np.linalg.norm(P, axis=1))
        return min(min_dist / 2.0, min_bound)
        
    current_radius = calc_radius(P)
    best_radius = current_radius
    best_P = P.copy()

    # Simulated Annealing
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.9995
    
    # Step size scaling
    base_step = 0.05
    
    while time.time() - start_time < time_limit:
        # Check if we should stop cooling
        if T < T_min:
            break
            
        # Pick a random point
        i = np.random.randint(N)
        
        # Generate random step
        # Scale step with T or current radius?
        # Let's scale with current radius to be adaptive
        step_scale = base_step * (current_radius / 0.05) 
        # Cap step scale
        step_scale = np.clip(step_scale, 0.001, 0.1)
        
        delta = np.random.normal(0, step_scale, size=2)
        
        P_new = P.copy()
        P_new[i] += delta
        
        # Check boundary constraint
        if np.linalg.norm(P_new[i]) > 1.0:
            # Reject move
            T *= cooling_rate
            continue
            
        # Calculate new radius
        # Optimization: We only changed point i.
        # We can recalculate full radius for simplicity, N=300 is small.
        new_radius = calc_radius(P_new)
        
        # Acceptance criterion
        if new_radius > current_radius:
            P = P_new
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_P = P.copy()
        else:
            # Metropolis criterion
            # Probability of accepting a worse move
            # delta_r = new_radius - current_radius (negative)
            # prob = exp(delta_r / T)
            # If T is very small, this is 0.
            if T > 0:
                prob = np.exp((new_radius - current_radius) / T)
                if np.random.rand() < prob:
                    P = P_new
                    current_radius = new_radius
            
        # Cool down
        T *= cooling_rate

    # Polish Phase: Deterministic local search
    # Try to improve by small moves in various directions
    # We have remaining time
    remaining_time = time_limit - (time.time() - start_time)
    
    if remaining_time > 5.0:
        # Use a smaller step size and no randomness, or very small randomness
        # Let's do a "hill climbing" on the best_P found so far
        P = best_P.copy()
        current_radius = best_radius
        
        step_sizes = [0.01, 0.005, 0.001, 0.0005]
        
        for ss in step_sizes:
            improved = True
            while improved and (time.time() - start_time) < time_limit - 2.0:
                improved = False
                # Shuffle order to avoid bias
                indices = np.random.permutation(N)
                for i in indices:
                    # Try 8 directions + random
                    directions = np.array([
                        [1, 0], [-1, 0], [0, 1], [0, -1],
                        [1, 1], [1, -1], [-1, 1], [-1, -1]
                    ]) / np.sqrt(2)
                    # Add a random direction
                    rand_dir = np.random.normal(0, 1, size=2)
                    rand_dir /= np.linalg.norm(rand_dir)
                    directions = np.vstack([directions, rand_dir])
                    
                    best_move_dir = None
                    best_move_r = current_radius
                    
                    for d in directions:
                        P_try = P.copy()
                        P_try[i] += d * ss
                        if np.linalg.norm(P_try[i]) > 1.0:
                            continue
                        r_try = calc_radius(P_try)
                        if r_try > best_move_r:
                            best_move_r = r_try
                            best_move_dir = d
                            
                    if best_move_dir is not None:
                        P[i] += best_move_dir * ss
                        current_radius = best_move_r
                        if current_radius > best_radius:
                            best_radius = current_radius
                            best_P = P.copy()
                        improved = True

    return best_P

# EVOLVE-BLOCK-END
