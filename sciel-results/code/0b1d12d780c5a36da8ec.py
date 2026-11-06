import numpy as np
from scipy.spatial import Delaunay

N = 130

# EVOLVE-BLOCK-START
def construct():
    # 1. Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    pts = np.column_stack([x, y, z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    # 2. Local Search to minimize covering radius
    # We iteratively find the worst Voronoi cell (triangle in Delaunay triangulation)
    # and move its vertices towards its circumcenter.
    
    max_iter = 2000
    step_size = 0.1
    
    for _ in range(max_iter):
        try:
            tri = Delaunay(pts)
            facets = tri.simplices
            
            # Vectorized circumcenter calculation
            A = pts[facets[:, 0]]
            B = pts[facets[:, 1]]
            C = pts[facets[:, 2]]
            
            e1 = B - A
            e2 = C - A
            normals = np.cross(e1, e2)
            
            # Normalize normals to get circumcenter directions
            norms_n = np.linalg.norm(normals, axis=1, keepdims=True)
            # Avoid division by zero
            norms_n[norms_n == 0] = 1.0
            centers = normals / norms_n
            
            # Calculate radii (angular distance from center to A)
            dots = np.sum(centers * A, axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            radii = np.arccos(dots)
            
            # Find max radius facet
            max_idx = np.argmax(radii)
            max_radius = radii[max_idx]
            
            # If max radius is very small, we might be converged
            # Best known is ~11.316 degrees ~ 0.1975 radians
            if max_radius < 0.1975:
                break
                
            # Move vertices of the worst facet towards its circumcenter
            facet = facets[max_idx]
            center = centers[max_idx]
            
            for idx in facet:
                pts[idx] = (1 - step_size) * pts[idx] + step_size * center
                pts[idx] /= np.linalg.norm(pts[idx])
                
        except Exception:
            # Perturb if stuck or error
            pts += np.random.normal(0, 1e-5, size=(N, 3))
            norms = np.linalg.norm(pts, axis=1)
            pts = pts / norms[:, np.newaxis]
            continue

    # Final normalization
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    return pts
# EVOLVE-BLOCK-END
