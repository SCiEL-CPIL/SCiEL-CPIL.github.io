import numpy as np

def construct():
    N = 32
    
    # Helper to resolve collisions by pushing circles apart
    # Returns modified centers
    def resolve_collisions(centers, max_iter=100):
        centers = centers.copy()
        for _ in range(max_iter):
            moved = False
            # Check all pairs
            for i in range(N):
                for j in range(i + 1, N):
                    d = centers[i] - centers[j]
                    dist = np.linalg.norm(d)
                    # If overlapping (dist < min possible radius sum? 
                    # We don't know radii yet. 
                    # Instead, we enforce a minimum distance. 
                    # Let's assume we want to keep them separated by some epsilon 
                    # or just separate them if they are too close.
                    # Better strategy: 
                    # The "score" function will determine radii. 
                    # To make the score function meaningful, we should ensure 
                    # circles are not *too* close. 
                    # Let's push them apart if dist < 0.1 (arbitrary threshold) 
                    # or just always push if they are very close.
                    # Actually, a better way: 
                    # We want to maximize the minimum distance. 
                    # So we should push them apart if dist < min_dist_threshold.
                    # But what is the threshold?
                    # Let's just push them apart if they are extremely close to avoid singularities, 
                    # but rely on the optimization to spread them.
                    # Wait, if we don't enforce disjointness, the "radius" calculation 
                    # r_i = min(d_wall, min_j d_ij / 2) ensures disjointness by definition 
                    # IF we interpret it as "maximum radius such that no overlap occurs".
                    # If r_i = d_ij/2 and r_j = d_jk/2, they might still overlap if d_ij is small?
                    # No, if r_i = d_ij/2, then r_i + r_j <= d_ij + r_j. 
                    # If r_j is also d_ij/2, then r_i+r_j = d_ij. They touch. 
                    # So the formula r_i = min(d_wall_i, 0.5 * min_{j!=i} d_ij) 
                    # guarantees that the circles are disjoint (touching at most).
                    # So we don't need to resolve collisions! 
                    # The "radius" is derived from the positions.
                    # The optimization is to find positions that maximize the sum of these derived radii.
                    # This is a much simpler and more robust formulation.
                    pass
            # Since we don't need to resolve collisions if radii are defined by local min distance,
            # this function is effectively a no-op for the logic, but we keep the structure 
            # in case we want to add a "hard" constraint later.
            # For now, we just return centers as is.
            return centers

    # Calculate score based on positions
    # r_i = min( dist_to_wall_i, 0.5 * min_{j != i} dist(i, j) )
    # This ensures disjoint interiors.
    def calc_score(centers):
        radii = np.zeros(N)
        for i in range(N):
            # Wall distance
            dx = min(centers[i, 0], 1.0 - centers[i, 0])
            dy = min(centers[i, 1], 1.0 - centers[i, 1])
            d_wall = min(dx, dy)
            
            # Nearest neighbor distance
            min_dist = float('inf')
            for j in range(N):
                if i == j:
                    continue
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
            
            # Radius is half the distance to the nearest neighbor, capped by wall distance
            r_i = min(d_wall, 0.5 * min_dist)
            radii[i] = r_i
            
        return np.sum(radii), radii

    # Initialization: Hexagonal lattice with jitter
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Add jitter
                x += np.random.uniform(-0.02, 0.02)
                y += np.random.uniform(-0.02, 0.02)
                centers.append([x, y])
                
    centers = np.array(centers)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Local Search
    rng = np.random.default_rng(42)
    current_score, current_radii = calc_score(centers)
    best_score = current_score
    best_centers = centers.copy()
    best_radii = current_radii.copy()
    
    T = 0.01
    cooling = 0.999
    steps = 8000
    move_size = 0.05
    
    for step in range(steps):
        # Pick a random circle
        i = rng.integers(0, N)
        
        # Generate move
        dx = rng.uniform(-move_size, move_size)
        dy = rng.uniform(-move_size, move_size)
        
        old_x = centers[i, 0]
        old_y = centers[i, 1]
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Keep inside box
        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        
        centers[i, 0] = new_x
        centers[i, 1] = new_y
        
        # Calculate new score
        new_score, new_radii = calc_score(centers)
        
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            # Accept
            current_score = new_score
            current_radii = new_radii
            
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
                best_radii = current_radii.copy()
        else:
            # Reject
            centers[i, 0] = old_x
            centers[i, 1] = old_y
            
        T *= cooling
        
        # Slightly reduce move size
        move_size *= 0.9999
        if move_size < 0.001:
            move_size = 0.001

    # Final result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_centers[i, 0], best_centers[i, 1], best_radii[i]]
        
    return result
