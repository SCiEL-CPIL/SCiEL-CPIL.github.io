"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _pairwise_dot_products(pts):
    """Compute all pairwise dot products."""
    return pts @ pts.T


def _covering_radius_degrees(pts):
    """
    Estimate the covering radius in degrees.
    The covering radius is half the maximum angular distance between any two
    nearest neighbors in the Voronoi sense, but a simpler proxy is:
    For each point, find its nearest neighbor, compute the angular distance,
    and take half of the maximum such distance as an estimate.
    
    Actually, a better estimate: the covering radius is the maximum over all
    points on the sphere of the distance to the nearest config point. This is
    equal to the maximum circumradius of the spherical Voronoi cells.
    
    A practical estimate: for each point i, find the 3 nearest neighbors
    (since on a sphere, the Voronoi cell is bounded by arcs to nearest neighbors).
    The circumradius of the spherical triangle formed by point i and its two
    nearest neighbors that bound the largest cell... this is complex.
    
    Simpler: use the fact that the covering radius is at least half the max
    nearest-neighbor distance. For a good estimate, we can compute for each
    point the maximum angular distance to any of its Voronoi vertices.
    
    For simplicity and speed, let's use: for each point, find its nearest
    neighbor, compute angular distance d_i. The covering radius is approximately
    max_i(d_i / 2) but this underestimates. A better bound: the covering radius
    is at most max_i(d_i / 2) * something.
    
    Actually, let's just compute the maximum over all pairs of the angular
    distance, and note that the covering radius is related to the largest gap.
    
    For the optimization, we'll use a different objective: maximize the minimum
    pairwise angular distance (which is equivalent to minimizing the maximum
    nearest-neighbor distance, which is a proxy for covering radius).
    """
    # Compute pairwise angular distances
    dot = _pairwise_dot_products(pts)
    # Clip for numerical stability
    dot = np.clip(dot, -1.0, 1.0)
    # Angular distances
    angles = np.arccos(dot)
    # For each point, find the nearest neighbor (excluding self)
    np.fill_diagonal(angles, np.pi)
    min_angles = np.min(angles, axis=1)
    # The covering radius is roughly half the maximum nearest-neighbor distance
    # But actually, for a set of points, the covering radius is the maximum
    # distance from any point on the sphere to its nearest config point.
    # This is equal to the maximum circumradius of the spherical Voronoi cells.
    # A good lower bound is max_i(min_angles[i] / 2).
    # A good upper bound involves the actual Voronoi cells.
    # For optimization purposes, we'll use max(min_angles) as a proxy.
    return np.degrees(np.max(min_angles))


def _objective(x):
    """
    Objective function to minimize: negative of the minimum pairwise dot product.
    Minimizing this is equivalent to maximizing the minimum pairwise dot product,
    which maximizes the minimum angular separation.
    
    Wait, we want to maximize the minimum angular separation, which means
    maximizing the minimum angle, which means minimizing the maximum dot product
    between distinct points? No.
    
    Angular distance = arccos(dot). Maximizing angular distance means minimizing
    dot product. So we want to minimize the maximum dot product between distinct
    points.
    
    Actually, let's think again. We want points to be as far apart as possible.
    The minimum angular separation is min_{i!=j} arccos(p_i . p_j).
    To maximize this, we minimize the maximum of p_i . p_j over i!=j.
    
    So the objective should be: maximize the minimum angle = minimize the max dot product.
    """
    pts = _project_to_sphere(x)
    dot = _pairwise_dot_products(pts)
    # Set diagonal to -1 so it doesn't affect the max
    np.fill_diagonal(dot, -1.0)
    # We want to minimize the maximum off-diagonal dot product
    return np.max(dot)


def _gradient(x):
    """
    Gradient of the objective: max over i!=j of p_i . p_j, where p = x / ||x||.
    
    Let f = max_{i!=j} p_i . p_j. Let (i*, j*) be the pair achieving the max.
    Then df/dx = (dp_i*/dx) . p_j* + p_i* . (dp_j*/dx)
    
    dp_i/dx = (I - p_i p_i^T) / ||x_i|| for the i-th component... this is getting
    complicated. Let's use a simpler approach: use scipy's minimize with
    numerical gradients, or use a projected gradient method.
    
    Actually, let's use a different formulation. We'll use a soft-max penalty
    on the closest neighbors as suggested in the directive.
    """
    pts = _project_to_sphere(x)
    dot = _pairwise_dot_products(pts)
    np.fill_diagonal(dot, -1.0)
    
    # Use a soft-max to approximate the max
    beta = 50.0
    exp_dot = np.exp(beta * dot)
    # Zero out diagonal
    np.fill_diagonal(exp_dot, 0.0)
    
    # Soft-max of the dot products
    log_sum_exp = np.log(np.sum(exp_dot, axis=1))
    weights = exp_dot / np.sum(exp_dot, axis=1, keepdims=True)
    
    # The soft-max value
    soft_max = log_sum_exp / beta
    
    # Gradient with respect to pts
    # d(soft_max)/d(dot_ij) = weights[i,j]
    # d(dot_ij)/d(pts_i) = pts_j
    # d(dot_ij)/d(pts_j) = pts_i
    
    grad_pts = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i != j:
                grad_pts[i] += weights[i, j] * pts[j]
                grad_pts[j] += weights[i, j] * pts[i]
    
    # Chain rule: d(pts)/d(x) where pts = x / ||x||
    # d(pts_i)/d(x_i) = (I - pts_i pts_i^T) / ||x_i||
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    
    grad_x = np.zeros_like(x)
    for i in range(N):
        n_i = norms[i, 0]
        proj = np.eye(3) - np.outer(pts[i], pts[i])
        grad_x[i] = (proj @ grad_pts[i]) / n_i
    
    return grad_x


def construct():
    """
    Construct 130 points on the unit sphere using projected gradient ascent
    to maximize the minimum pairwise angular separation.
    """
    # Start with a Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Use scipy's minimize with L-BFGS-B for efficiency
    # But we need to handle the projection. Let's use a penalty approach instead.
    
    # Alternative: use a simple projected gradient descent with multiple restarts
    best_x = x0.copy()
    best_score = _covering_radius_degrees(best_x)
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if restart == 0:
            x = x0.copy()
        else:
            # Random initialization
            x = rng.normal(size=(N, 3))
            x = _project_to_sphere(x)
        
        # Projected gradient ascent: we want to minimize the max dot product
        # So we do gradient descent on the objective
        step_size = 0.01
        for it in range(2000):
            grad = _gradient(x)
            x_new = x - step_size * grad
            x_new = _project_to_sphere(x_new)
            
            # Adaptive step size
            if it % 100 == 0:
                score = _covering_radius_degrees(x_new)
                if score < best_score:
                    best_score = score
                    best_x = x_new.copy()
            
            x = x_new
        
        score = _covering_radius_degrees(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
