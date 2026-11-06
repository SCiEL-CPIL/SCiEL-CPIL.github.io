import numpy as np
import time

N = 50

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm

def _spherical_distance(a, b):
    # Angular distance between two unit vectors
    dot = np.clip(np.sum(a * b), -1.0, 1.0)
    return np.arccos(dot)

def _covering_radius(points):
    # Compute the covering radius by checking the circumradius of the convex hull facets
    # For simplicity and speed, we approximate by sampling points on the sphere
    # and finding the maximum distance to the nearest point.
    # A more accurate method would involve computing the Voronoi diagram, but this is complex.
    # Instead, we use a dense sampling of the sphere.
    
    # Generate a dense set of points on the sphere
    num_samples = 10000
    phi = np.random.uniform(0, 2 * np.pi, num_samples)
    theta = np.arccos(np.random.uniform(-1, 1, num_samples))
    samples = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # Compute distances from each sample to each point
    # To speed up, we compute the dot products
    dot_products = samples @ points.T
    dot_products = np.clip(dot_products, -1.0, 1.0)
    distances = np.arccos(dot_products)
    
    # Find the minimum distance for each sample
    min_distances = np.min(distances, axis=1)
    
    # The covering radius is the maximum of these minimum distances
    return np.max(min_distances)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave a safety margin
    
    # Initialize with a Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Optimization loop
    step_size = 0.01
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Compute the gradient of the covering radius with respect to each point
        # This is complex, so we use a heuristic: move each point towards the centroid
        # of its Voronoi cell.
        
        # For each point, find the samples that are closest to it
        dot_products = points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        distances = np.arccos(dot_products)
        
        # For each point, move it towards the direction that reduces the maximum distance
        # to its neighbors.
        for i in range(N):
            # Find the neighbors
            neighbor_distances = distances[i]
            neighbor_distances[i] = np.inf  # Exclude self
            
            # Move towards the centroid of the neighbors
            # This is a simplified version of Lloyd's algorithm
            # A more accurate method would be to compute the gradient of the objective function
            # For now, we use a small random perturbation and check if it improves the radius
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                # Random perturbation
                delta = np.random.normal(0, step_size, 3)
                new_point = _normalize(points[i] + delta)
                
                # Update the point
                old_point = points[i]
                points[i] = new_point
                
                # Check if the radius improved
                new_radius = _covering_radius(points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = points.copy()
                else:
                    # Revert the change
                    points[i] = old_point
                    break
        
        # Gradually reduce the step size
        step_size *= 0.99
        
        # Check for convergence
        if step_size < 1e-6:
            break
    
    return best_points
