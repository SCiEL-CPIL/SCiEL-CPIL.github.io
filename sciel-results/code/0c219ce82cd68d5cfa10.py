"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math

def _covering_radius(points):
    """Compute the covering radius (in radians) as the max circumradius of convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                continue
            C = C_plane / norm_C
            cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_r:
                max_r = R
        return max_r
    except Exception:
        return np.pi

def _repulsion_energy(points, k=2.0, cutoff=0.5):
    """
    Calculate an energy function that penalizes large gaps.
    We use a sum of inverse distance powers, but truncated or modified to focus on the worst-case.
    A better proxy for covering radius minimization is minimizing the max distance.
    However, gradient descent on max is hard. 
    We can use a 'soft' max or sum of exp.
    Let's use a repulsive force model: points repel each other.
    Energy = sum_{i,j} 1/d_ij^k. Minimizing this spreads points out.
    But to specifically target covering radius, we should also attract points to the 'holes'.
    
    Alternative: Direct gradient descent on the covering radius is difficult due to non-differentiability.
    Instead, we use a two-step process:
    1. Greedy initialization (Farthest Point Sampling).
    2. Local optimization using the circumcenters of the largest facets.
       Move the points of the largest facet towards its circumcenter to shrink it.
    """
    pass

def _greedy_init(n):
    """
    Initialize points using Farthest-Point Sampling.
    This ensures a good initial distribution without spiral artifacts.
    """
    points = np.zeros((n, 3))
    # Start with a random point
    rng = np.random.default_rng(12345)
    points[0] = rng.normal(size=3)
    points[0] /= np.linalg.norm(points[0])
    
    # Distances to nearest existing point
    min_dists = np.full(n, np.inf)
    
    for i in range(1, n):
        # Calculate distance from each candidate point to the nearest existing point
        # We need to find the point on the sphere (or from a set of candidates) that maximizes min_dist.
        # To be efficient, we can sample many candidates and pick the best.
        # Or, we can use a grid. A Fibonacci spiral of candidates is a good grid.
        
        # Generate a large set of candidate points (Fibonacci spiral)
        M = 5000
        idx = np.arange(M) + 0.5
        z = 1.0 - 2.0 * idx / M
        phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
        r = np.sqrt(1.0 - z * z)
        candidates = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Compute distances from all candidates to all existing points
        # points shape: (i, 3), candidates shape: (M, 3)
        # dist shape: (M, i)
        # We want min_dist for each candidate
        # d^2 = |c|^2 + |p|^2 - 2 c.p = 1 + 1 - 2 c.p = 2 - 2 c.p
        # min_dist is max of (2 - 2 c.p) -> min of (c.p)
        
        # Use broadcasting
        # candidates: (M, 3), points[:i]: (i, 3)
        dots = candidates @ points[:i].T
        min_dots = np.min(dots, axis=1)
        max_dot = np.max(min_dots) # This is wrong. We want to maximize the minimum distance.
        # Distance d = arccos(dot). Max distance <=> Min dot.
        # So we want to minimize the maximum dot product? No.
        # For a candidate c, d_min = min_j d(c, p_j).
        # d(c, p_j) = arccos(c.p_j).
        # Minimizing d_min is hard. Maximizing d_min is the goal.
        # d_min is large when c.p_j is small for all j.
        # So we want to minimize max_j (c.p_j).
        
        max_dots = np.max(dots, axis=1)
        best_idx = np.argmin(max_dots)
        
        points[i] = candidates[best_idx]
        
    return points

def _local_optimize_hull(points, iterations=200, step_size=0.05):
    """
    Local optimization by moving points of the largest facet towards its circumcenter.
    This directly reduces the covering radius.
    """
    points = points.copy()
    
    for _ in range(iterations):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        facet_radii = []
        facet_centers = []
        
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                facet_radii.append(0.0)
                facet_centers.append(np.zeros(3))
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                facet_radii.append(0.0)
                facet_centers.append(np.zeros(3))
                continue
            C = C_plane / norm_C
            cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            facet_radii.append(R)
            facet_centers.append(C)
            
        facet_radii = np.array(facet_radii)
        max_idx = np.argmax(facet_radii)
        max_r = facet_radii[max_idx]
        
        if max_r < 0.197: # ~11.3 degrees
            break
            
        simplex = hull.simplices[max_idx]
        idx1, idx2, idx3 = simplex
        C = facet_centers[max_idx]
        
        # Move the three points towards the circumcenter
        # Use a small step to avoid breaking the hull structure too much
        for idx in [idx1, idx2, idx3]:
            p = points[idx]
            # Move towards C
            new_p = p + step_size * (C - p)
            new_p /= np.linalg.norm(new_p)
            points[idx] = new_p
            
    return points

def _gradient_descent_repulsion(points, iterations=500, learning_rate=0.01):
    """
    Use repulsion to spread points evenly, then refine.
    Energy = sum_{i!=j} 1/d_ij^2.
    Gradient of energy w.r.t p_i:
    E = sum_{j!=i} (1 - 2 p_i.p_j + p_j.p_j)^-1/2 ... wait, d_ij = sqrt(2 - 2 p_i.p_j).
    E = sum_{j!=i} (2 - 2 p_i.p_j)^-k/2.
    dE/dp_i = sum_{j!=i} (-k/2) (2 - 2 p_i.p_j)^(-k/2 - 1) * (-2 p_j)
             = k sum_{j!=i} p_j (2 - 2 p_i.p_j)^(-k/2 - 1)
    
    We project the gradient onto the tangent space of the sphere.
    grad_tangent = grad - (grad . p_i) p_i
    p_i_new = p_i - lr * grad_tangent
    p_i_new /= ||p_i_new||
    """
    points = points.copy()
    k = 2.0
    
    for it in range(iterations):
        grads = np.zeros_like(points)
        
        # Compute pairwise distances squared
        # d2[i,j] = 2 - 2 p_i.p_j
        dots = points @ points.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0) # Avoid division by zero, though we exclude i=j
        
        # We need sum_{j!=i} p_j * (d2_ij)^(-k/2 - 1)
        # weights[i,j] = (d2[i,j])^(-k/2 - 1)
        # Set diagonal to 0
        weights = np.zeros_like(d2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        # Calculate weights only where mask is true
        # d2 can be small, leading to large weights. Clip d2 to avoid inf.
        d2_safe = np.clip(d2, 1e-6, None)
        weights[mask] = d2_safe[mask] ** (-k/2.0 - 1.0)
        
        # grads[i] = k * sum_j weights[i,j] * p_j
        grads = k * (weights @ points)
        
        # Project to tangent space
        for i in range(N):
            p_i = points[i]
            # Remove radial component
            radial_comp = np.dot(grads[i], p_i)
            grad_tan = grads[i] - radial_comp * p_i
            
            # Update
            new_p = p_i - learning_rate * grad_tan
            norm = np.linalg.norm(new_p)
            if norm < 1e-12:
                continue
            points[i] = new_p / norm
            
    return points

def construct():
    # 1. Greedy Initialization
    points = _greedy_init(N)
    
    # 2. Global Repulsion Optimization to smooth out the distribution
    points = _gradient_descent_repulsion(points, iterations=300, learning_rate=0.005)
    
    # 3. Local Hull Facet Optimization to minimize covering radius
    points = _local_optimize_hull(points, iterations=500, step_size=0.02)
    
    # 4. Final Polish
    points = _local_optimize_hull(points, iterations=200, step_size=0.005)
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
