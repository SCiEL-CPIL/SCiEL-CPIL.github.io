import numpy as np

def construct():
    N = 16
    
    # Define the basis vectors for the triangular lattice
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3.0)/2.0])
    
    # Generate a grid of lattice points
    # We need enough points to choose the closest 16
    # A range of -10 to 10 is sufficient
    points = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            p = i * v1 + j * v2
            points.append(p)
            
    points = np.array(points)
    
    # Calculate distance from the origin for each point
    # The origin is a lattice point, so it's included
    dists = np.linalg.norm(points, axis=1)
    
    # Get indices of the 16 closest points to the origin
    # argsort returns indices that would sort the array
    idx = np.argsort(dists)[:N]
    
    # Select the points
    selected_points = points[idx]
    
    # To ensure the configuration is as compact as possible, 
    # we might want to center the selected points around their centroid.
    # However, the set of closest points to the origin is already centered 
    # around the origin (which is one of the points). 
    # Centering around the centroid might slightly reduce the diameter 
    # if the shape is not perfectly symmetric around the origin, 
    # but for a cluster of closest points, the origin is a good center.
    
    # Let's center around the centroid of the selected points to be safe
    centroid = np.mean(selected_points, axis=0)
    selected_points = selected_points - centroid
    
    return selected_points
