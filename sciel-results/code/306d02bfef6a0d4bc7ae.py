import numpy as np
from scipy.spatial import cKDTree
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    k = 10
    step_size = 0.8
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
            
        tree = cKDTree(pts)
        # Query k+1 nearest neighbors (including self)
        dists, indices = tree.query(pts, k=k+1)
        
        # Exclude self (first column)
        knn_indices = indices[:, 1:]
        
        # Centroid of neighbors
        neighbors = pts[knn_indices]
        centroids = np.mean(neighbors, axis=1)
        
        # Normalize
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_pts = centroids / norms
        
        # Update
        pts = (1 - step_size) * pts + step_size * new_pts
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
    return pts
