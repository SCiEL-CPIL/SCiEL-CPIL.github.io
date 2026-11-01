import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_score(points):
    """
    Compute the score (radius) for a given set of points.
    Score = min(min_pairwise_distance / 2, min_distance_to_boundary)
    """
    if len(points) < 2:
        return 0.0
    
    # Calculate pairwise distances
    # To avoid memory issues for large N, we do it in chunks, but N=50 is small
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            dists.append(np.linalg.norm(points[i] - points[j]))
    
    min_pair_dist = min(dists) if dists else 0.0
    
    # Calculate distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary_dist = 1.0 - np.max(radii)
    
    # The radius of the circles is half the minimum distance between centers
    # AND must fit inside the container
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Ensure validity: circles must be inside and disjoint
    # If r <= 0, it's invalid or degenerate
    return r

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal lattice
    # We generate a hexagonal grid and select the N points closest to the center
    # This provides a good starting point with non-overlapping circles.
    
    def generate_hex_lattice(N_target):
        spacing = 2.0 / (np.sqrt(N_target) + 1.0)
        points = []
        # Generate points in a range large enough to cover the unit circle
        range_k = int(1.0 / spacing) + 3
        for j in range(-range_k, range_k + 1):
            for i in range(-range_k, range_k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N_target:
            # Fallback to random if lattice is too sparse (unlikely for N=50)
            return np.random.uniform(-1, 1, (N_target, 2))
        
        # Select N_target points closest to center
        radii = np.linalg.norm(points, axis=1)
        indices = np.argsort(radii)
        return points[indices[:N_target]]

    current_points = generate_hex_lattice(N)
    best_points = current_points.copy()
    best_score = _compute_score(best_points)
    
    # 2. Simulated Annealing
    # We will iteratively perturb points and accept changes based on score improvement
    # or probabilistically to escape local minima.
    
    T = 0.1  # Initial temperature
    cooling_rate = 0.995
    min_T = 1e-6
    
    # We use a simple move: pick a random point and displace it by a small random vector
    # Or, pick a pair and adjust distance. Displacement is simpler.
    
    while T > min_T:
        if time.time() - start_time > timeout:
            break
            
        # Generate a new candidate
        new_points = current_points.copy()
        
        # Randomly select a few points to move (e.g., 1 or 2)
        num_moves = np.random.choice([1, 2], p=[0.8, 0.2])
        indices_to_move = np.random.choice(N, num_moves, replace=False)
        
        for idx in indices_to_move:
            # Random displacement
            angle = np.random.uniform(0, 2 * np.pi)
            magnitude = np.random.uniform(0, T * 2.0)
            new_points[idx] += magnitude * np.array([np.cos(angle), np.sin(angle)])
            
            # Constrain to inside unit circle (soft constraint handled by score, 
            # but hard projection helps stability)
            r = np.linalg.norm(new_points[idx])
            if r > 0.99:
                new_points[idx] = new_points[idx] * (0.99 / r)
        
        new_score = _compute_score(new_points)
        
        # Acceptance criterion
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            
        # Metropolis criterion
        delta = new_score - _compute_score(current_points)
        if delta > 0:
            current_points = new_points
        else:
            # Accept with probability exp(delta / T)
            # Since scores are small, we scale delta for temperature
            # Actually, using the raw score difference is standard for max problems
            # We want to maximize, so if new < old, delta is negative.
            # P(accept) = exp(delta / T)
            if np.random.rand() < np.exp(delta / T):
                current_points = new_points
                
        T *= cooling_rate
        
        # Optional: Restart if stuck? 
        # For now, simple SA.
        
    # Final Polish: Local optimization using gradient-free Nelder-Mead or simple hill climbing
    # Since we have time left, let's do a quick local refinement on the best points found
    
    if time.time() - start_time < timeout - 10:
        # Simple local search: try small moves on each point individually
        points = best_points.copy()
        score = best_score
        step_size = 0.01
        improved = True
        while improved and time.time() - start_time < timeout - 5:
            improved = False
            for i in range(N):
                original = points[i].copy()
                # Try 8 directions
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        points[i] = original + np.array([dx, dy])
                        # Check boundary
                        if np.linalg.norm(points[i]) > 1.0:
                            continue
                        new_s = _compute_score(points)
                        if new_s > score:
                            score = new_s
                            improved = True
                            best_points = points.copy()
                            best_score = score
                # Reset if no improvement in this point's local search
                if not improved:
                    points[i] = original
        best_points = points if score > best_score else best_points
        if score > best_score:
            best_score = score

    return best_points

# EVOLVE-BLOCK-END
