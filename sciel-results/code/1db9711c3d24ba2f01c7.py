# EVOLVE-BLOCK-START
import numpy as np

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a Fibonacci sphere initialization followed by local optimization to maximize the minimum distance.
    """
    n = 14
    d = 3
    
    # 1. Initialize points on a unit sphere using Fibonacci sphere distribution
    # This provides a good uniform spread
    points = np.zeros((n, d))
    golden_angle = np.pi * (3. - np.sqrt(5.))
    for i in range(n):
        z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
        radius = np.sqrt(1 - z*z)
        theta = golden_angle * i
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points[i] = [x, y, z]
        
    # Normalize to unit sphere (Fibonacci sphere is already on unit sphere, but just in case)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]

    # 2. Local Optimization to maximize minimum distance (Tammes Problem approximation)
    # We will iterate, finding the closest pair and pushing them apart.
    # We also need to keep points on the sphere? 
    # Actually, for max ratio in 3D, points might move off the sphere slightly to allow better packing?
    # No, for fixed max distance, the optimal set for min distance usually lies on the boundary of the convex hull.
    # For N=14, the optimal configuration is likely on a sphere.
    
    # Let's do a simple hill climbing on the sphere surface.
    # Objective: Maximize min pairwise distance.
    
    def min_dist(pts):
        # Calculate pairwise distances
        # Using scipy is allowed, but let's stick to numpy for simplicity and speed if possible
        # N=14 is small, 91 pairs.
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                dists.append(np.linalg.norm(pts[i] - pts[j]))
        return min(dists), dists

    # Optimization Loop
    # We will try to push the closest pair apart.
    # Since points are on a sphere, moving them apart means moving them in the direction of the vector between them, projected onto the tangent plane.
    
    for _ in range(1000):
        min_d, _ = min_dist(points)
        
        # Find the pair with minimum distance
        min_pair = (0, 1)
        current_min = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d_ij = np.linalg.norm(points[i] - points[j])
                if d_ij < current_min:
                    current_min = d_ij
                    min_pair = (i, j)
        
        i, j = min_pair
        pi = points[i]
        pj = points[j]
        
        # Vector from i to j
        v = pj - pi
        
        # We want to move i away from j and j away from i.
        # Direction for i: -v (away from j)
        # Direction for j: v (away from i)
        
        # Project these directions onto the tangent plane at pi and pj respectively.
        # Tangent at pi is perpendicular to pi.
        # tangent_i = v - (v dot pi) * pi  (Note: v is pj-pi, so direction from i to j is v. 
        # We want to move i in direction -v. 
        # Let dir_i = -v.
        # Project dir_i onto tangent plane of pi:
        # t_i = dir_i - (dir_i dot pi) * pi
        
        dir_i = -v
        t_i = dir_i - np.dot(dir_i, pi) * pi
        
        dir_j = v
        t_j = dir_j - np.dot(dir_j, pj) * pj
        
        # Normalize tangent vectors
        norm_t_i = np.linalg.norm(t_i)
        norm_t_j = np.linalg.norm(t_j)
        
        if norm_t_i < 1e-9 or norm_t_j < 1e-9:
            continue
            
        t_i = t_i / norm_t_i
        t_j = t_j / norm_t_j
        
        # Step size
        # Adaptive step size based on current min distance? 
        # Small step to ensure stability.
        step = 0.01
        
        new_pi = pi + step * t_i
        new_pj = pj + step * t_j
        
        # Re-normalize to sphere
        new_pi = new_pi / np.linalg.norm(new_pi)
        new_pj = new_pj / np.linalg.norm(new_pj)
        
        # Check if this move improves the global minimum distance
        # We need to check if the new min distance is greater than the old one.
        # To be efficient, we can just compute the new min distance for the whole set.
        
        points[i] = new_pi
        points[j] = new_pj
        
        new_min_d, _ = min_dist(points)
        
        if new_min_d <= min_d:
            # Revert
            points[i] = pi
            points[j] = pj
            # If stuck, try a smaller step or random perturbation?
            # For now, just break or continue with smaller step?
            # Let's try a smaller step if we revert.
            # But for simplicity, if we revert, we might be at a local max.
            # Let's add a small random perturbation to escape local minima.
            if _ > 900:
                idx = np.random.randint(0, n)
                points[idx] += np.random.randn(3) * 0.01
                points[idx] /= np.linalg.norm(points[idx])

    return points

# EVOLVE-BLOCK-END
