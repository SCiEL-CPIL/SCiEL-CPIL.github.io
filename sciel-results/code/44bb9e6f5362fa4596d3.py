"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _covering_radius(pts):
    """Compute the covering radius (in radians) of a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point. We approximate this by checking the circumradii
    of the spherical triangles formed by the Delaunay triangulation of the points.
    For a better approximation without explicit triangulation, we can use the fact that
    the covering radius is determined by the "worst" Voronoi cell.
    
    A robust numerical check: sample the sphere densely and find the max distance to nearest point.
    However, this is slow. A better proxy is the max circumradius of the convex hull facets.
    We will use the convex hull method as it is exact for the "empty cap" condition.
    """
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
        
        # Find circumcenter of the spherical triangle (a,b,c)
        # u is perpendicular to (a-b) and (a-c)
        u = np.cross(a - b, a - c)
        u_norm = np.linalg.norm(u)
        
        if u_norm < 1e-10:
            continue
            
        u = u / u_norm
        
        # Ensure u is on the same side as the facet normal (outward)
        normal = np.cross(b - a, c - a)
        if np.dot(u, normal) < 0:
            u = -u
        
        # Compute circumradius
        dot_val = np.clip(np.dot(u, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        
        if r > max_circumradius:
            max_circumradius = r
    
    return max_circumradius


def _repulsive_gradient(pts, power=2.0):
    """Compute the gradient of the repulsive energy sum(1/d^power) projected onto tangent spaces.
    
    Energy E = sum_{i<j} d_ij^{-power}
    Gradient w.r.t x_i: 
    dE/dx_i = -power * sum_{j!=i} d_ij^{-power-1} * (d d_ij / d x_i)
    d d_ij / d x_i = -x_j / sin(d_ij)
    So dE/dx_i = power * sum_{j!=i} x_j / (d_ij^{power+1} * sin(d_ij))
    
    We project this gradient onto the tangent plane at x_i:
    g_i_tangent = g_i - (g_i . x_i) x_i
    """
    pts = np.asarray(pts)
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Compute all pairwise dot products
    dots = np.dot(pts, pts.T)
    # Mask diagonal to avoid division by zero
    mask = ~np.eye(n, dtype=bool)
    
    # Clip to avoid arccos issues
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    dists = np.arccos(dots)
    
    # Compute sin(dists)
    sins = np.sin(dists)
    
    # Avoid division by zero
    eps = 1e-10
    safe_dists = np.where(dists < eps, eps, dists)
    safe_sins = np.where(sins < eps, eps, sins)
    
    # Term: x_j / (d_ij^{power+1} * sin(d_ij))
    # We want to sum over j for each i.
    # Let's construct a matrix M where M[i,j] = 1 / (d_ij^{power+1} * sin(d_ij))
    # Then grad[i] = power * sum_j M[i,j] * x_j
    
    # Create the weight matrix
    # Note: dists[i,i] is 0, so we mask it out.
    weights = np.zeros_like(dists)
    np.fill_diagonal(weights, 0.0)
    
    # Calculate weights for valid entries
    valid = mask
    weights[valid] = 1.0 / (safe_dists[valid] ** (power + 1) * safe_sins[valid])
    
    # Gradient contribution from repulsion
    # grad[i] = power * (weights @ pts)[i]
    rep_grad = power * (weights @ pts)
    
    # Project onto tangent space
    for i in range(n):
        radial_comp = np.dot(rep_grad[i], pts[i])
        grad[i] = rep_grad[i] - radial_comp * pts[i]
        
    return grad


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius.
    
    Uses vectorized projected gradient descent on a repulsive potential.
    """
    best_cr = np.inf
    best_pts = None
    
    # Set random seed for reproducibility if needed, but we want to explore
    # We will run multiple restarts with different initializations
    
    for restart in range(20):
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Random initialization
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Optimize
        lr = 0.05
        n_iter = 2000
        momentum = 0.9
        v = np.zeros_like(pts)
        
        prev_cr = np.inf
        patience = 100
        no_improve = 0
        
        for it in range(n_iter):
            # Compute gradient
            grad = _repulsive_gradient(pts, power=2.0)
            
            # Momentum update
            v = momentum * v - lr * grad
            pts_new = pts + v
            
            # Normalize to unit sphere
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            pts_new = pts_new / norms
            
            # Check for coincident points (optional, gradient handles it mostly but can explode)
            # If points get too close, the gradient becomes huge. 
            # The repulsive force 1/d^2 pushes them apart strongly.
            
            pts = pts_new
            
            # Compute covering radius periodically to check improvement
            if it % 10 == 0:
                cr = _covering_radius(pts)
                if cr < best_cr:
                    best_cr = cr
                    best_pts = pts.copy()
                
                if cr < prev_cr - 1e-6:
                    prev_cr = cr
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve > patience:
                    break
                if cr < 0.32: # ~18.3 degrees
                    break
                    
    # Final local refinement using Nelder-Mead on the best points
    # We flatten the points to a 1D vector for scipy.optimize
    # Constraint: norm of each triplet is 1.
    # We can use penalty method or just rely on the previous result if it's good enough.
    # Given the time limit, a few Nelder-Mead steps on the flattened vector might be risky.
    # Instead, we do a few more iterations of the gradient descent with very small step size.
    
    if best_pts is not None:
        pts = best_pts.copy()
        lr = 0.001
        for it in range(500):
            grad = _repulsive_gradient(pts, power=2.0)
            pts_new = pts - lr * grad
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            pts_new = pts_new / norms
            pts = pts_new
            
            if it % 50 == 0:
                cr = _covering_radius(pts)
                if cr < best_cr:
                    best_cr = cr
                    best_pts = pts.copy()

    return best_pts
# EVOLVE-BLOCK-END
