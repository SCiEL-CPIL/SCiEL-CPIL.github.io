"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _normalize(v):
    """Normalize rows of a (N,3) array to unit length."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return v / norms

def _covering_radius(pts, n_test=20000):
    """
    Estimate the covering radius (in degrees) of a set of points on the sphere.
    Uses a dense random sampling of the sphere plus the vertices and edge midpoints
    of the convex hull for a tighter estimate.
    """
    pts = _normalize(pts)
    N = len(pts)
    
    # Compute pairwise dot products
    # For each test point, find the max dot product (min angular distance)
    
    # Test points: random on sphere + vertices
    rng = np.random.default_rng(42)
    # Fibonacci-like dense sampling for test points
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Also add the points themselves
    test_pts = np.vstack([test_pts, pts])
    
    # Compute dot products: (n_test+N, N)
    dots = test_pts @ pts.T
    max_dots = np.max(dots, axis=1)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    min_angles = np.arccos(max_dots)
    return np.max(min_angles) * 180.0 / np.pi

def _soft_max_covering(pts, beta):
    """
    Differentiable soft-max approximation of the covering radius.
    For each point on a fixed grid, compute the soft-min of angular distances.
    """
    pts = _normalize(pts)
    N = len(pts)
    
    # Use a moderately sized test grid for gradient computation
    n_test = 5000
    rng = np.random.default_rng(123)
    # Fixed test points for consistent gradients
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Angular distances: arccos(dot(test, pt))
    dots = test_pts @ pts.T  # (n_test, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (n_test, N)
    
    # Soft-min over N for each test point: -soft_max(-angles)
    # soft_max(x, beta) = (1/beta) * log(sum(exp(beta*x)))
    neg_angles = -angles
    # For numerical stability
    max_neg = np.max(neg_angles, axis=1, keepdims=True)
    exp_neg = np.exp(beta * (neg_angles - max_neg))
    soft_max_neg = (1.0 / beta) * (np.log(np.sum(exp_neg, axis=1)) + beta * np.max(neg_angles, axis=1))
    soft_min_angles = -soft_max_neg
    
    return np.max(soft_min_angles) * 180.0 / np.pi

def _soft_max_covering_grad(pts, beta):
    """
    Compute the soft-max covering radius and its gradient w.r.t. pts.
    """
    pts = _normalize(pts)
    N = len(pts)
    
    n_test = 5000
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = test_pts @ pts.T  # (n_test, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (n_test, N)
    
    # Soft-min: we want d/dpts of max_j softmin_i angle(j,i)
    # First, soft-min over i for each j
    neg_angles = -angles
    max_neg = np.max(neg_angles, axis=1, keepdims=True)
    exp_neg = np.exp(beta * (neg_angles - max_neg))
    sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
    probs = exp_neg / sum_exp  # (n_test, N) - weights for soft-min
    
    # soft_min_angles[j] = -1/beta * (log(sum_exp[j]) + beta*max_neg[j])
    # d(soft_min_angles[j])/d(angles[j,i]) = -probs[j,i]
    # d(angles[j,i])/d(pts[i]) = -test_pts[j] / sqrt(1 - dots[j,i]^2)  (in tangent space)
    
    # For each test point j, the gradient contribution to pts[i] is:
    # sum_j [ -probs[j,i] * (-test_pts[j] / sqrt(1-dots^2)) ]
    # But we only need the max over j of soft_min_angles[j], so we need to find which j is max
    # and use a soft version.
    
    # Simpler: use the max test point (or soft-max over test points)
    # For now, use the argmax test point for a cleaner gradient
    sm_angles = - (1.0 / beta) * (np.log(sum_exp[:, 0]) + beta * np.max(neg_angles, axis=1))
    
    # Find the test point with max soft_min_angle
    j_star = np.argmax(sm_angles)
    
    # Gradient for pts[i]: sum over j of indicator(j==j_star) * (-probs[j,i]) * (-test_pts[j]/sqrt(1-dots^2))
    # = probs[j_star, i] * test_pts[j_star] / sqrt(1 - dots[j_star,i]^2)
    
    sqrt_term = np.sqrt(np.maximum(1e-12, 1.0 - dots[j_star] ** 2))
    grad = np.zeros_like(pts)
    for i in range(N):
        grad[i] = probs[j_star, i] * test_pts[j_star] / sqrt_term[i]
    
    # Project gradient to tangent space: grad - (grad . pts) * pts
    dot_grad_pts = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - dot_grad_pts * pts
    
    # Convert to degrees
    value = sm_angles[j_star] * 180.0 / np.pi
    grad_tangent = grad_tangent * 180.0 / np.pi
    
    return value, grad_tangent

def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lloyd_iteration(pts, n_subsample=20000):
    """
    One Lloyd's iteration: move each point to the centroid of its Voronoi cell.
    Uses random sampling to approximate Voronoi cell centroids.
    """
    pts = _normalize(pts)
    N = len(pts)
    
    rng = np.random.default_rng(7)
    # Sample points on sphere
    theta = rng.uniform(0, 2 * np.pi, n_subsample)
    cos_phi = rng.uniform(-1, 1, n_subsample)
    sin_phi = np.sqrt(1 - cos_phi ** 2)
    test_pts = np.column_stack([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
    
    # Assign each test point to nearest pts
    dots = test_pts @ pts.T
    nearest = np.argmax(dots, axis=1)
    
    # Compute centroids
    centroids = np.zeros((N, 3))
    counts = np.zeros(N)
    for i in range(N):
        mask = nearest == i
        if np.sum(mask) > 0:
            centroids[i] = np.sum(test_pts[mask], axis=0)
            counts[i] = np.sum(mask)
    
    # Normalize centroids
    for i in range(N):
        norm = np.linalg.norm(centroids[i])
        if norm > 1e-12:
            centroids[i] /= norm
        else:
            centroids[i] = pts[i]
    
    return _normalize(centroids)

def _simulated_annealing(pts, n_iter=200, T_start=0.5, T_end=0.01, n_test=10000):
    """
    Simple simulated annealing on the sphere.
    """
    pts = _normalize(pts)
    N = len(pts)
    rng = np.random.default_rng(99)
    
    # Precompute test points
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def _quick_covering(p):
        p = _normalize(p)
        dots = test_pts @ p.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        min_angles = np.arccos(max_dots)
        return np.max(min_angles) * 180.0 / np.pi
    
    current = _normalize(pts.copy())
    current_cost = _quick_covering(current)
    best = current.copy()
    best_cost = current_cost
    
    for it in range(n_iter):
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Pick a random point and perturb it
        idx = rng.integers(0, N)
        # Random tangent direction
        rand_dir = rng.normal(size=3)
        rand_dir = rand_dir - np.dot(rand_dir, current[idx]) * current[idx]
        rand_norm = np.linalg.norm(rand_dir)
        if rand_norm < 1e-12:
            continue
        rand_dir /= rand_norm
        
        step = 0.1
        new_pt = current[idx] + step * rand_dir
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        new_pts = current.copy()
        new_pts[idx] = new_pt
        
        new_cost = _quick_covering(new_pts)
        
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = new_pts
            current_cost = new_cost
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_points(N)
    best_pts = pts.copy()
    best_score = _covering_radius(pts)
    
    # Phase 1: Lloyd's iterations
    for it in range(30):
        if time.time() > deadline - 30:
            break
        pts = _lloyd_iteration(pts)
        score = _covering_radius(pts)
        if score < best_score:
            best_pts = pts.copy()
            best_score = score
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 20:
        pts = _simulated_annealing(pts, n_iter=300, T_start=0.3, T_end=0.01, n_test=15000)
        score = _covering_radius(pts)
        if score < best_score:
            best_pts = pts.copy()
            best_score = score
    
    # Phase 3: More Lloyd's iterations
    pts = best_pts.copy()
    for it in range(30):
        if time.time() > deadline - 10:
            break
        pts = _lloyd_iteration(pts)
        score = _covering_radius(pts)
        if score < best_score:
            best_pts = pts.copy()
            best_score = score
    
    # Phase 4: Gradient-based refinement using L-BFGS-B
    if time.time() < deadline - 5:
        # Flatten pts for optimization
        x0 = best_pts.flatten()
        
        def _objective(x):
            p = _normalize(x.reshape(N, 3))
            return _soft_max_covering(p, beta=50.0)
        
        def _grad(x):
            p = _normalize(x.reshape(N, 3))
            val, grad = _soft_max_covering_grad(p, beta=50.0)
            return grad.flatten()
        
        try:
            result = minimize(_objective, x0, jac=_grad, method='L-BFGS-B',
                             options={'maxiter': 100, 'gtol': 1e-8})
            if result.success or result.fun < best_score:
                new_pts = _normalize(result.x.reshape(N, 3))
                score = _covering_radius(new_pts)
                if score < best_score:
                    best_pts = new_pts
                    best_score = score
        except Exception:
            pass
    
    # Final validation: ensure no coincident points and not in a hemisphere
    best_pts = _normalize(best_pts)
    
    # Check for coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                # Separate them slightly
                best_pts[j] += 1e-4 * np.random.randn(3)
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Check if all points are in a closed hemisphere
    # If so, reflect some points
    # Simple check: if max dot product with any axis is < 1 for all, we're likely fine
    # More robust: check if there exists a plane through origin with all points on one side
    # For now, just ensure spread
    if time.time() < deadline - 1:
        # One more Lloyd iteration
        best_pts = _lloyd_iteration(best_pts)
        best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
