import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.pi, np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.pi, np.array([]), np.array([])
        
    # Extract vertices for all facets
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        norms[~valid] = 1.0
        n[~valid] = [0, 0, 1]
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    max_radius = np.max(radii)
    return max_radius, radii, simplices

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    best_points = None
    best_radius = np.pi

    # Initializations
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Create a grid of test points Y for the soft-max objective
    M = 3000
    j = np.arange(M) + 0.5
    zy = 1.0 - 2.0 * j / M
    phiy = np.pi * (1.0 + np.sqrt(5.0)) * j
    ry = np.sqrt(1.0 - zy * zy)
    Y = np.column_stack([ry * np.cos(phiy), ry * np.sin(phiy), zy])

    # Main Optimization Loop: Soft-Max Gradient Descent
    points = init_fib.copy()
    
    # Parameters
    sigma = 0.15
    rho = 0.15
    lr = 0.005
    decay = 0.98
    
    while time.time() - start_time < time_limit * 0.5:
        # Compute distances D (M x N)
        C = Y @ points.T
        C = np.clip(C, -1.0, 1.0)
        D = np.arccos(C)
        
        # Soft-min for each y
        D_min = np.min(D, axis=1, keepdims=True)
        exp_term = np.exp(-(D - D_min) / sigma)
        sum_exp = np.sum(exp_term, axis=1, keepdims=True)
        W = exp_term / sum_exp
        
        # Soft-max for y
        D_soft = D_min.squeeze(1) - sigma * np.log(sum_exp.squeeze(1))
        D_soft_max = np.max(D_soft)
        exp_term_y = np.exp((D_soft - D_soft_max) / rho)
        sum_exp_y = np.sum(exp_term_y)
        V = exp_term_y / sum_exp_y
        
        # Gradient
        C_safe = np.clip(C, -1.0 + 1e-6, 1.0 - 1e-6)
        denom = np.sqrt(1.0 - C_safe * C_safe)
        
        Y_exp = Y[:, np.newaxis, :]
        W_exp = W[:, :, np.newaxis]
        denom_exp = denom[:, :, np.newaxis]
        
        G = W_exp * (-Y_exp / denom_exp)
        
        V_exp = V[:, np.newaxis, np.newaxis]
        
        grad = np.sum(V_exp * G, axis=0)
        
        # Update points
        points -= lr * grad
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Anneal
        sigma *= decay
        rho *= decay
        
        if sigma < 0.01:
            sigma = 0.01
        if rho < 0.01:
            rho = 0.01

    # Polish Phase: Convex Hull Optimization with Nesterov Momentum
    alpha = 0.05
    no_improve = 0
    velocity = np.zeros_like(points)
    
    while time.time() - start_time < time_limit - 3.0:
        max_radius, radii, simplices = compute_covering_radius_and_facets(points)
        
        if len(simplices) == 0:
            break
            
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 30:
                alpha *= 0.95
                if alpha < 1e-5:
                    break

        # Find top 5 worst facets and push their vertices
        top_k = min(5, len(radii))
        worst_indices = np.argsort(radii)[-top_k:]
        
        for max_idx in worst_indices:
            simplex = simplices[max_idx]
            i_idx, j_idx, k_idx = simplex
            
            a = points[i_idx]
            b = points[j_idx]
            c = points[k_idx]
            
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            u = n / norm_n
            centroid = (a + b + c) / 3.0
            if np.dot(u, centroid) < 0:
                u = -u
            
            # Move vertices towards u with momentum
            step_size = alpha * radii[max_idx]
            
            grad_i = u - a
            grad_j = u - b
            grad_k = u - c
            
            velocity[i_idx] = 0.9 * velocity[i_idx] + step_size * grad_i
            velocity[j_idx] = 0.9 * velocity[j_idx] + step_size * grad_j
            velocity[k_idx] = 0.9 * velocity[k_idx] + step_size * grad_k
            
            points[i_idx] += velocity[i_idx]
            points[j_idx] += velocity[j_idx]
            points[k_idx] += velocity[k_idx]
        
        # Normalize
        points /= np.linalg.norm(points, axis=1, keepdims=True)

    if best_points is None:
        best_points = points

    return best_points
