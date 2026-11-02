import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

N = 130

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_from_hull(points):
    """
    Compute the exact covering radius from the convex hull.
    For each triangular facet, compute the circumradius of the triangle
    and return the maximum.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        
        # Compute side lengths
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)
        
        # Check for degenerate triangles
        if ab < 1e-12 or bc < 1e-12 or ca < 1e-12:
            continue
        
        # Circumradius formula: R = abc / (4 * Area)
        cross = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(cross)
        
        if area < 1e-12:
            continue
        
        R = (ab * bc * ca) / (4.0 * area)
        max_radius = max(max_radius, R)
    
    return max_radius

def _compute_gradient(X, Y, beta, penalty_weight=100.0):
    """
    Computes the gradient of the soft-max covering radius objective + sphere penalty.
    X: (N, 3) array of points on the sphere.
    Y: (M, 3) array of test points on the sphere.
    beta: temperature parameter for soft-max.
    """
    N_pts = X.shape[0]
    M_pts = Y.shape[0]
    
    # Normalize X to ensure they are on the sphere during gradient calculation
    # (L-BFGS-B might drift slightly, but we penalize it)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Compute dot products: (M, N)
    # Y @ X_norm.T
    dots = Y @ X_norm.T
    
    # Clip for numerical stability in arccos
    dots = np.clip(dots, -1.0, 1.0)
    
    # Find nearest neighbor index for each test point y
    # Max dot product corresponds to min angular distance
    nn_idx = np.argmax(dots, axis=1) # (M,)
    
    # Get the dot products for the nearest neighbors
    dots_nn = dots[np.arange(M_pts), nn_idx] # (M,)
    
    # Angular distances
    angles = np.arccos(dots_nn) # (M,)
    
    # Soft-max weights
    # J = (1/beta) * log(sum(exp(beta * angles)))
    # grad J = sum(w_y * grad angle_y)
    # w_y = exp(beta * angle_y) / sum(exp(beta * angle_y))
    
    # Numerical stability for exp
    max_angle = np.max(angles)
    exp_angles = np.exp(beta * (angles - max_angle))
    sum_exp = np.sum(exp_angles)
    weights = exp_angles / sum_exp # (M,)
    
    # Gradient of angle with respect to x_j
    # d/dx arccos(y . x) = -(y - (y.x)x) / sqrt(1 - (y.x)^2)
    # This is only non-zero for j = nn_idx(y)
    
    grads = np.zeros_like(X)
    
    # We need to accumulate gradients for each x_j
    # For each y, the gradient is applied to X[nn_idx[y]]
    
    # Vectorized calculation of the gradient vector for each y
    # y: (M, 3), x_nn: (M, 3)
    x_nn = X_norm[nn_idx]
    
    # y_dot_x = dots_nn (M,)
    # proj_y_on_x = y_dot_x * x_nn
    # diff = y - proj_y_on_x
    # norm_diff = sqrt(1 - y_dot_x^2)
    
    y_dot_x = dots_nn
    diff = Y - y_dot_x[:, np.newaxis] * x_nn
    norm_diff = np.sqrt(np.clip(1.0 - y_dot_x**2, 1e-12, 1.0))
    
    # grad_angle = -diff / norm_diff
    grad_angles = -diff / norm_diff[:, np.newaxis] # (M, 3)
    
    # Weighted gradients
    weighted_grads = weights[:, np.newaxis] * grad_angles # (M, 3)
    
    # Scatter-add to grads
    # np.add.at is slow, but M is not huge (e.g., 5000). 
    # Alternatively, use bincount or loop if M is small.
    # For M=5000, N=130, a loop in Python is too slow. 
    # We can use np.add.at or a matrix multiplication trick if indices were unique, but they are not.
    # np.add.at is acceptable for M=5000.
    np.add.at(grads, nn_idx, weighted_grads)
    
    # Add sphere constraint penalty gradient
    # Penalty P = penalty_weight * sum(|x_i|^2 - 1)^2
    # dP/dx_i = 4 * penalty_weight * (|x_i|^2 - 1) * x_i
    norms_sq = np.sum(X**2, axis=1)
    penalty_grad = 4.0 * penalty_weight * (norms_sq - 1.0)[:, np.newaxis] * X
    
    return grads + penalty_grad

def _objective(X, Y, beta, penalty_weight=100.0):
    """
    Computes the soft-max covering radius + sphere penalty.
    """
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    dots = Y @ X_norm.T
    dots = np.clip(dots, -1.0, 1.0)
    nn_idx = np.argmax(dots, axis=1)
    dots_nn = dots[np.arange(len(Y)), nn_idx]
    angles = np.arccos(dots_nn)
    
    max_angle = np.max(angles)
    exp_angles = np.exp(beta * (angles - max_angle))
    soft_max = (1.0 / beta) * np.log(np.sum(exp_angles)) + max_angle
    
    norms_sq = np.sum(X**2, axis=1)
    penalty = penalty_weight * np.sum((norms_sq - 1.0)**2)
    
    return soft_max + penalty

def _lloyd_step(X, Y, n_iter=30):
    """
    Perform Lloyd refinement using Voronoi cells.
    For each point, compute the centroid of its Voronoi cell (approximated
    by the mean of test points that are closest to it) and move toward it.
    """
    X = X.copy()
    for _ in range(n_iter):
        # Normalize
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        # Assign each test point to nearest X point
        dots = Y @ X.T
        nn_idx = np.argmax(dots, axis=1)
        
        # Compute centroids
        new_X = np.zeros_like(X)
        counts = np.zeros(N)
        for i in range(N):
            mask = (nn_idx == i)
            if np.any(mask):
                new_X[i] = np.mean(Y[mask], axis=0)
                counts[i] = np.sum(mask)
        
        # Normalize and update
        for i in range(N):
            if counts[i] > 0:
                norm = np.linalg.norm(new_X[i])
                if norm > 1e-12:
                    new_X[i] = new_X[i] / norm
                else:
                    new_X[i] = X[i]
            else:
                new_X[i] = X[i]
        
        X = new_X
    
    return X

def _exact_local_optimize(X, Y, max_iter=50):
    """
    Exact local optimization using Nelder-Mead on the hull-based covering radius.
    """
    def obj_func(x_flat):
        X_curr = x_flat.reshape(N, 3)
        X_curr = X_curr / np.linalg.norm(X_curr, axis=1, keepdims=True)
        return _covering_radius_from_hull(X_curr)
    
    x0 = X.flatten()
    
    try:
        res = minimize(
            obj_func,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if res.fun < obj_func(x0):
            X_result = res.x.reshape(N, 3)
            X_result = X_result / np.linalg.norm(X_result, axis=1, keepdims=True)
            return X_result
    except Exception:
        pass
    
    return X

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Generate test points
    M_test = 20000
    Y = _fibonacci_sphere(M_test)
    
    # Initial guess: Fibonacci spiral
    X = _fibonacci_sphere(N)
    
    # Parameters
    beta = 50.0
    penalty_weight = 100.0
    
    def obj_func(x_flat):
        X_curr = x_flat.reshape(N, 3)
        return _objective(X_curr, Y, beta, penalty_weight)
        
    def jac_func(x_flat):
        X_curr = x_flat.reshape(N, 3)
        return _compute_gradient(X_curr, Y, beta, penalty_weight).flatten()
        
    x0 = X.flatten()
    
    best_x = X.copy()
    best_val = _covering_radius_from_hull(X)
    
    # Stage 1: Soft-max gradient descent with L-BFGS-B
    try:
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            res = minimize(
                obj_func, 
                x0, 
                jac=jac_func, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000}
            )
            
            X_candidate = res.x.reshape(N, 3)
            X_candidate = X_candidate / np.linalg.norm(X_candidate, axis=1, keepdims=True)
            val_candidate = _covering_radius_from_hull(X_candidate)
            
            if val_candidate < best_val:
                best_x = X_candidate
                best_val = val_candidate
    except Exception:
        pass
    
    # Stage 2: Lloyd refinement
    try:
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            X_lloyd = _lloyd_step(best_x, Y, n_iter=30)
            val_lloyd = _covering_radius_from_hull(X_lloyd)
            
            if val_lloyd < best_val:
                best_x = X_lloyd
                best_val = val_lloyd
    except Exception:
        pass
    
    # Stage 3: Exact local polish with Nelder-Mead
    try:
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            X_polished = _exact_local_optimize(best_x, Y, max_iter=50)
            val_polished = _covering_radius_from_hull(X_polished)
            
            if val_polished < best_val:
                best_x = X_polished
                best_val = val_polished
    except Exception:
        pass
    
    # Final normalization
    X_final = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    
    return X_final

# EVOLVE-BLOCK-END
