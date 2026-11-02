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

    # Phase 1: Variational Energy Minimization (Riesz s-energy, s=1)
    # This promotes uniform distribution by repelling points.
    num_energy_iters = 3000
    alpha_start = 0.05
    alpha_end = 0.0001
    decay = (alpha_end / alpha_start) ** (1.0 / (num_energy_iters - 1))
    
    for it in range(num_energy_iters):
        # Compute pairwise distances and forces
        # E = sum_{i<j} 1/||xi-xj||
        # grad_i E = sum_{j!=i} (xi - xj) / ||xi - xj||^3
        
        # Vectorized computation of gradients
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Force magnitude: 1/dist^3
        # We want sum over j != i of (xi - xj) / dist^3
        # Note: diff[i,j] is xi - xj.
        # We need to zero out the diagonal in the sum.
        
        inv_dist_cubed = 1.0 / (dist ** 3)
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        # grad[i] = sum_j diff[i,j] * inv_dist_cubed[i,j]
        grads = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
        
        # Project to tangent space
        # grad_tangent = grad - (grad . xi) * xi
        dot_prod = np.sum(grads * points, axis=1)
        grads_tangent = grads - dot_prod[:, np.newaxis] * points
        
        # Update
        alpha = alpha_start * (decay ** it)
        points = points - alpha * grads_tangent
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms

    # Phase 2: Refine using Covering Radius Gradient (similar to original but shorter)
    # This targets the specific metric: minimax angular distance.
    num_refine_iters = 1000
    alpha_start = 0.05
    alpha_end = 0.0001
    decay = (alpha_end / alpha_start) ** (1.0 / (num_refine_iters - 1))
    
    for it in range(num_refine_iters):
        centers, radii, simplices = compute_circumcenters(points)
        
        # Weight by exponential of radius squared
        sigma = 0.05
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
        
        alpha = alpha_start * (decay ** it)
        
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
