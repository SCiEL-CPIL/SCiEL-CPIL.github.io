import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _generate_icosahedron(level=9):
    """
    Generate a high-resolution icosahedral grid.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi,  0.],
        [ 1,  phi,  0.],
        [-1, -phi,  0.],
        [ 1, -phi,  0.],
        [ 0., -1,  phi],
        [ 0.,  1,  phi],
        [ 0., -1, -phi],
        [ 0.,  1, -phi],
        [ phi,  0., -1],
        [ phi,  0.,  1],
        [-phi,  0., -1],
        [-phi,  0.,  1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    for i in range(level):
        new_faces = []
        mids = {}
        for f in faces:
            a, b, c = f
            def mid(p1, p2):
                key = (min(p1, p2), max(p1, p2))
                if key not in mids:
                    v = verts[p1] + verts[p2]
                    v /= np.linalg.norm(v)
                    mids[key] = len(verts)
                    verts = np.vstack([verts, v])
                return mids[key]
            
            ab = mid(a, b)
            bc = mid(b, c)
            ca = mid(c, a)
            
            new_faces.append((a, ab, ca))
            new_faces.append((b, bc, ab))
            new_faces.append((c, ca, bc))
            new_faces.append((ab, bc, ca))
        faces = new_faces
        
    return verts

def _compute_covering_radius_grid(points, grid):
    """
    Compute the covering radius using the grid.
    """
    dots = points @ grid.T
    max_dots = np.max(dots, axis=0)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)
    return np.degrees(np.max(angles))

def construct():
    # Generate grid
    grid = _generate_icosahedron(level=9)
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = points.copy()
    best_radius = _compute_covering_radius_grid(points, grid)
    
    # Optimization loop
    max_iter = 2000
    step_size = 0.1
    alpha = 5.0 # Weighting parameter for far points
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute dots
        dots = points @ grid.T
        dots = np.clip(dots, -1.0, 1.0)
        
        # Distances
        dists = np.arccos(dots)
        
        # Find nearest point for each grid point
        nearest_idx = np.argmin(dists, axis=0)
        
        # Compute weights
        dists_nearest = dists[nearest_idx, np.arange(grid.shape[0])]
        weights = np.exp(alpha * dists_nearest)
        
        # Accumulate weighted centroids using bincount
        centroids = np.zeros((N, 3))
        weights_sum = np.zeros(N)
        
        # contributions = weights[:, np.newaxis] * grid
        # This is large (524k x 3).
        # We can do it component by component to save memory if needed, but 524k*3*8 bytes = 12MB. Fine.
        
        contrib = weights[:, np.newaxis] * grid
        for k in range(3):
            centroids[:, k] = np.bincount(nearest_idx, weights=contrib[:, k], minlength=N)
            
        weights_sum = np.bincount(nearest_idx, weights=weights, minlength=N)
        
        # Avoid division by zero
        weights_sum[weights_sum == 0] = 1
        
        centroids = centroids / weights_sum[:, np.newaxis]
        
        # Normalize centroids
        cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        cent_norms[cent_norms == 0] = 1
        centroids = centroids / cent_norms
        
        # Update points
        dots_pc = np.einsum('ij,ij->i', points, centroids)
        tangent = centroids - dots_pc[:, np.newaxis] * points
        
        tan_norms = np.linalg.norm(tangent, axis=1, keepdims=True)
        tan_norms[tan_norms == 0] = 1
        tangent = tangent / tan_norms
        
        new_points = points + step_size * tangent
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check improvement
        current_radius = _compute_covering_radius_grid(new_points, grid)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            step_size *= 0.99
        else:
            step_size *= 0.9
            if step_size < 0.001:
                points += rng.normal(0, 0.01, points.shape)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                current_radius = _compute_covering_radius_grid(points, grid)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
                    step_size = 0.1
                else:
                    break
                    
    return best_points
