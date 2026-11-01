"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci lattice."""
    points = np.zeros((n, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x, y, z]
    return points

def construct():
    # 1. Initialize with Fibonacci Lattice
    points = _fibonacci_sphere(N)
    
    # 2. Prepare test points for covering radius approximation
    # Use a dense set of test points to approximate the max distance to nearest point.
    # 5000 test points is a good balance between accuracy and speed.
    n_test = 5000
    test_points = _fibonacci_sphere(n_test)
    
    # Pre-compute test points in a format suitable for distance calculation
    # We will compute distances from each test point to each of the N configuration points.
    
    p = 50  # L-p norm power, approximates max
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize points to ensure they stay on the sphere during optimization steps
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Calculate squared distances from all test points to all configuration points
        # dist_sq[i, j] = ||test_points[i] - pts[j]||^2 = 2 - 2 * dot(test_points[i], pts[j])
        # Using dot products is more stable and faster.
        dots = test_points @ pts.T  # shape (n_test, N)
        dist_sq = 2.0 - 2.0 * dots
        
        # Clip to avoid negative values due to floating point errors
        dist_sq = np.maximum(dist_sq, 0.0)
        
        # Angular distance squared is approx dist_sq for small angles, but we want angular distance.
        # Actually, minimizing Euclidean distance on the sphere is equivalent to minimizing angular distance.
        # We want the max angular distance.
        # Let's use the squared Euclidean distance as a proxy, or convert to angular.
        # cos(d) = 1 - dist_sq/2. d = arccos(1 - dist_sq/2).
        # For optimization, it's often better to optimize a smooth function of the distances.
        # Let's use the sum of exp(dist_sq * lambda) or L-p norm of dist_sq.
        
        # L-p norm of the squared distances
        # This emphasizes the largest distances
        max_dist_sq = np.max(dist_sq, axis=1) # shape (n_test,)
        # To make it differentiable and focus on the max, use a soft-max or L-p norm
        # L-p norm: (sum(max_dist_sq^p))^(1/p)
        # But we want the global max over all test points.
        # So we take the max over test points of the nearest distance.
        # Let d_i be the squared distance from test point i to its nearest neighbor.
        # We want to minimize max(d_i).
        # Approximate max(d_i) by (sum(d_i^p))^(1/p) for large p.
        
        # d_i is the min over j of dist_sq[i,j]
        # Wait, dist_sq[i,j] is dist between test i and config j.
        # The nearest distance for test point i is min_j dist_sq[i,j].
        # So d_i = min_j dist_sq[i,j].
        
        # We want to minimize max_i d_i.
        # Since min is not differentiable, and max is not differentiable, this is tricky.
        # However, the "covering radius" is defined by the Voronoi cells.
        # The point of maximum distance is the pole of the Voronoi cell with the largest circumradius.
        
        # Alternative objective: Sum of exp(-beta * d_i) where d_i is the *safety margin*?
        # No, we want to *increase* the minimum distance.
        
        # Let's use the "Repulsive" potential on the *test points*? No.
        
        # Let's stick to the Convex Hull approach but use a gradient-based method on the vertices.
        # The current program uses Nelder-Mead. I will use L-BFGS-B on a function that
        # computes the convex hull and returns the max circumradius.
        # But L-BFGS-B requires a gradient.
        
        # Since computing the gradient of the convex hull circumradius is complex,
        # let's use a different differentiable proxy:
        # Minimize the sum of the squared distances from the *centroids of the hull facets* 
        # to their vertices? No.
        
        # Let's go back to the "Max-Nearest-Neighbor" idea but handle non-differentiability.
        # We can use the fact that for the optimal configuration, the covering radius is determined
        # by a few facets.
        
        # Actually, a very effective simple differentiable objective for covering is:
        # Minimize the variance of the nearest-neighbor distances? No.
        
        # Let's use the "Energy" of the test points being repelled from the configuration?
        
        # Let's try a different differentiable objective:
        # Minimize the sum of the squared distances from each test point to the nearest configuration point,
        # weighted by a large power to emphasize the worst-case.
        # F = sum_i (d_i)^p
        # d_i = min_j ||t_i - c_j||^2
        # This is still non-differentiable at the min.
        
        # However, if we assume the assignment of test points to nearest neighbors is fixed (which it is
        # locally), we can compute gradients.
        
        # Let's implement a "Fixed Assignment" Gradient Descent.
        # 1. Compute nearest neighbor for each test point.
        # 2. Compute gradient of sum(d_i^p) with respect to c_j, assuming assignments are fixed.
        # 3. Update c_j.
        
        # This is a valid algorithmic approach: "Gradient Descent with Fixed Assignments".
        
        return 0.0 # Placeholder, we will implement the loop manually or use a wrapper
        
    # Since implementing a custom gradient with fixed assignments inside scipy.optimize is tricky,
    # I will write a manual gradient descent loop which is distinct from the Nelder-Mead approach.
    
    points = points.copy()
    
    # Use a smaller number of test points for speed in the loop, e.g., 1000
    test_points = _fibonacci_sphere(1000)
    
    p = 4.0 # Power for the loss function
    lr = 0.01 # Learning rate
    n_iter = 200
    
    for it in range(n_iter):
        # Normalize points
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Compute distances
        dots = test_points @ points.T
        dist_sq = 2.0 - 2.0 * dots
        dist_sq = np.maximum(dist_sq, 0.0)
        
        # Find nearest neighbor index for each test point
        nn_idx = np.argmin(dist_sq, axis=1)
        
        # Compute loss: Sum of (dist_sq[nn])^p
        # This emphasizes the larger distances
        d_nn = dist_sq[np.arange(len(test_points)), nn_idx]
        loss = np.sum(d_nn ** p)
        
        # Compute gradient
        # d_loss/d_c_j = sum_{i: nn_idx[i]==j} p * (d_nn[i])^(p-1) * d(dist_sq[i,j])/d_c_j
        # dist_sq[i,j] = 2 - 2 * t_i . c_j
        # d(dist_sq)/d_c_j = -2 * t_i
        # So grad_c_j = -2 * p * sum_{i in S_j} (d_nn[i])^(p-1) * t_i
        
        grads = np.zeros_like(points)
        for i in range(len(test_points)):
            j = nn_idx[i]
            w = p * (d_nn[i]) ** (p - 1)
            grads[j] += -2.0 * w * test_points[i]
            
        # Update points
        points = points - lr * grads
        
        # Project to sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay learning rate
        lr *= 0.99
        
    # Final refinement with Nelder-Mead on the exact covering radius (using convex hull)
    # This ensures we are on the precise manifold
    
    def _covering_radius_exact(pts):
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            for simplex in hull.simplices:
                verts = pts[simplex]
                if len(verts) == 3:
                    e1 = verts[1] - verts[0]
                    e2 = verts[2] - verts[0]
                    pole = np.cross(e1, e2)
                    pn = np.linalg.norm(pole)
                    if pn < 1e-12: continue
                    pole /= pn
                    cos_t = np.clip(np.dot(verts[0], pole), -1, 1)
                    max_rad = max(max_rad, np.arccos(cos_t))
            return max_rad
        except:
            return np.pi

    def objective_final(x):
        pts = x.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        return _covering_radius_exact(pts)
        
    res = minimize(objective_final, points.flatten(), method='Nelder-Mead', options={'maxiter': 100})
    points = res.x.reshape(N, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
