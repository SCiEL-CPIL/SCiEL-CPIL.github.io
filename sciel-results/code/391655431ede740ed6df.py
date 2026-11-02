import numpy as np
import scipy.spatial
import time

N = 130

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def compute_circumcenters(points):
        """
        Compute circumcenters and circumradii for the convex hull facets.
        """
        hull = scipy.spatial.ConvexHull(points)
        simplices = hull.simplices
        num_facets = len(simplices)
        centers = np.zeros((num_facets, 3))
        radii = np.zeros(num_facets)
        
        for idx, simplex in enumerate(simplices):
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                centers[idx] = (p1 + p2 + p3) / 3.0
                centers[idx] /= np.linalg.norm(centers[idx])
                radii[idx] = 0.0
                continue
            n = n / n_norm
            
            d = np.dot(n, p1)
            
            A = np.array([n, p1 - p2, p1 - p3])
            b = np.array([d, 0, 0])
            
            try:
                C = np.linalg.solve(A, b)
                C_norm = np.linalg.norm(C)
                C = C / C_norm
                cos_R = np.dot(C, p1)
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                centers[idx] = C
                radii[idx] = R
            except np.linalg.LinAlgError:
                centers[idx] = (p1 + p2 + p3) / 3.0
                centers[idx] /= np.linalg.norm(centers[idx])
                radii[idx] = 0.0
        
        return centers, radii, simplices

    # Optimization loop
    num_iterations = 2000
    initial_alpha = 0.2
    final_alpha = 0.0001
    decay_rate = (final_alpha / initial_alpha) ** (1.0 / (num_iterations - 1))
    
    for it in range(num_iterations):
        centers, radii, simplices = compute_circumcenters(points)
        
        # Weight by exponential of radius squared to focus on worst holes
        # This mimics a "soft" minimax approach
        sigma = 0.1
        w_facets = np.exp((radii / sigma) ** 2)
        w_facets = np.clip(w_facets, 0, 1e6)
        
        targets = np.zeros_like(points)
        weights = np.zeros(N)
        
        for idx, simplex in enumerate(simplices):
            R = radii[idx]
            if R < 1e-6:
                continue
            w = w_facets[idx]
            C = centers[idx]
            for k in range(3):
                i = simplex[k]
                targets[i] += w * C
                weights[i] += w
        
        alpha = initial_alpha * (decay_rate ** it)
        
        for i in range(N):
            if weights[i] > 1e-10:
                target = targets[i] / weights[i]
                points[i] = points[i] + alpha * (target - points[i])
                norm = np.linalg.norm(points[i])
                if norm > 1e-10:
                    points[i] /= norm
                else:
                    points[i] = np.random.randn(3)
                    points[i] /= np.linalg.norm(points[i])
                    
    return points
